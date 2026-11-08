"""
Turn TrashNet into the folder layout the trainer expects.
    python prepare_trashnet.py PATH_TO/dataset-resized  [--out .]
(TrashNet: https://github.com/garythung/trashnet  ->  data/dataset-resized.zip, unzip it first.)

Mapping:  plastic -> plastic | paper + cardboard -> paper | glass + metal + trash -> none
Writes dataset/ (80%), dataset_val/ (10%) and dataset_test/ (10%), split at random with a fixed seed.
Your own robot photos can be copied into dataset/<class>/ afterwards.
Note: TrashNet has several photos of the same object, so test accuracy is a little optimistic.
"""
import argparse
import os
import random
import shutil

MAP = {"plastic": "plastic", "paper": "paper", "cardboard": "paper",
       "glass": "none", "metal": "none", "trash": "none"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    random.seed(0)
    items = {"plastic": [], "paper": [], "none": []}
    for folder, cls in MAP.items():
        d = os.path.join(args.src, folder)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                items[cls].append((os.path.join(d, f), f"{folder}_{f}"))
    for cls, files in items.items():
        random.shuffle(files)
        n = len(files)
        cuts = {"dataset": files[:int(.8 * n)], "dataset_val": files[int(.8 * n):int(.9 * n)],
                "dataset_test": files[int(.9 * n):]}
        for split, fl in cuts.items():
            dst = os.path.join(args.out, split, cls)
            os.makedirs(dst, exist_ok=True)
            for src, name in fl:
                shutil.copy(src, os.path.join(dst, name))
        print(cls, {k: len(v) for k, v in cuts.items()})


if __name__ == "__main__":
    main()