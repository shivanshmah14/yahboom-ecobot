"""
Train the plastic / paper / none classifier (transfer learning on MobileNetV3-large, a CNN).
    pip install torch torchvision timm
    python train_classifier.py --data dataset [--val other_dataset] [--epochs 10]
Saves model.pt (used by:  python ecobot.py --model model.pt)

--val: a SEPARATE folder (same class sub-folders) photographed on another day / place / lighting.
       Without it, the validation split comes from the same photos, so accuracy looks better
       than it will be on the robot (burst frames of one object are near-duplicates).
"""
import argparse
import random
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

ARCH = "mobilenetv3_large_100"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--val", default=None)
    ap.add_argument("--test", default=None, help="held-out folder, evaluated once at the end")
    ap.add_argument("--weights", default=None, help="local ImageNet weights file (otherwise timm downloads them)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="model.pt")
    ap.add_argument("--finetune", action="store_true", help="also train the pretrained layers (slower, needs more photos)")
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (testing only)")
    args = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.35, 0.35, 0.35, 0.03),   # lighting changes on the real floor
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    val_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                 transforms.Normalize(MEAN, STD)])

    full_train = datasets.ImageFolder(args.data, train_tf)
    classes = full_train.classes
    print("classes:", classes, "images per class:",
          {c: sum(1 for _, y in full_train.samples if y == i) for i, c in enumerate(classes)})
    if args.val:
        val_ds = datasets.ImageFolder(args.val, val_tf)
        assert val_ds.classes == classes, "val folder must have the same class folders"
        train_ds = full_train
    else:
        idx = list(range(len(full_train)))
        random.shuffle(idx)
        cut = int(0.8 * len(idx))
        train_ds = Subset(full_train, idx[:cut])
        val_ds = Subset(datasets.ImageFolder(args.data, val_tf), idx[cut:])
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch, num_workers=args.workers)

    if args.weights:
        model = timm.create_model(ARCH, pretrained=False, num_classes=1000)
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
        model.reset_classifier(len(classes))
    else:
        model = timm.create_model(ARCH, pretrained=not args.no_pretrained, num_classes=len(classes))
    if not args.finetune:                                   # keep the pretrained image features frozen
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith(("classifier", "conv_head"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr if not args.finetune else args.lr / 3)
    loss_fn = nn.CrossEntropyLoss()

    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        if not args.finetune:                               # frozen layers: do not let BatchNorm drift
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        for x, y in train_dl:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        model.eval()
        correct = total = 0
        conf = torch.zeros(len(classes), len(classes), dtype=torch.long)
        with torch.no_grad():
            for x, y in val_dl:
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += len(y)
                for t, p in zip(y, pred):
                    conf[t, p] += 1
        acc = correct / max(total, 1)
        print(f"epoch {ep}/{args.epochs}  validation accuracy {acc:.1%}")
        if acc >= best:
            best = acc
            best_conf = conf.clone()
            torch.save({"state": model.state_dict(), "classes": classes, "arch": ARCH}, args.out)
    print(f"\nbest validation accuracy {best:.1%}  -> saved {args.out}")
    print("confusion matrix (rows = true, columns = predicted), order:", classes)
    print(best_conf)
    if args.test:
        model.load_state_dict(torch.load(args.out, map_location="cpu")["state"])
        model.eval()
        test_ds = datasets.ImageFolder(args.test, val_tf)
        conf = torch.zeros(len(classes), len(classes), dtype=torch.long)
        with torch.no_grad():
            for x, y in DataLoader(test_ds, batch_size=args.batch, num_workers=args.workers):
                for t, p in zip(y, model(x).argmax(1)):
                    conf[t, p] += 1
        print("\nTEST set (never used for training or model selection), order:", classes)
        print(conf)
        print(f"test accuracy {conf.diag().sum().item() / conf.sum().item():.1%}")
        for i, c in enumerate(classes):
            tp = conf[i, i].item()
            print(f"  {c}: recall {tp / max(conf[i].sum().item(), 1):.1%}  precision {tp / max(conf[:, i].sum().item(), 1):.1%}")


if __name__ == "__main__":
    main()