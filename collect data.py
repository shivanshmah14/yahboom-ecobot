"""
Collect training photos for the plastic / paper classifier.
    python collect_data.py [--camera 0] [--out dataset]

Hold ONE object inside the green square (fill most of it).  Keys (click the window first):
    p / a / n   save 1 photo as  plastic / paper / none   (none = empty floor, hands, other junk)
    P / A / N   (capital) save a burst of 10 photos - rotate/tilt the object while it saves
    q           quit
Aim for 150+ per class, with different objects, angles, distances and lighting.
Photos are saved as 224x224 crops in dataset/<class>/.
"""
import argparse
import os
import time
import cv2

KEYS = {"p": "plastic", "a": "paper", "n": "none"}


def count(out):
    return {c: len(os.listdir(os.path.join(out, c))) for c in KEYS.values()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default="dataset")
    args = ap.parse_args()
    for c in KEYS.values():
        os.makedirs(os.path.join(args.out, c), exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    burst, burst_cls, last = 0, None, 0.0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        side = int(min(h, w) * 0.5)
        x0, y0 = (w - side) // 2, (h - side) // 2
        crop = frame[y0:y0 + side, x0:x0 + side]

        def save(cls):
            img = cv2.resize(crop, (224, 224))
            name = f"{int(time.time() * 1000)}.jpg"
            cv2.imwrite(os.path.join(args.out, cls, name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if burst > 0 and time.time() - last > 0.15:
            save(burst_cls)
            burst -= 1
            last = time.time()

        view = frame.copy()
        cv2.rectangle(view, (x0, y0), (x0 + side, y0 + side), (0, 255, 0), 2)
        n = count(args.out)
        cv2.putText(view, "plastic:%d  paper:%d  none:%d" % (n["plastic"], n["paper"], n["none"]),
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(view, "p/a/n = 1 photo   P/A/N = burst of 10   q = quit",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("collect", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        ch = chr(k) if 0 < k < 128 else ""
        if ch.lower() in KEYS:
            if ch.islower():
                save(KEYS[ch])
            else:
                burst, burst_cls = 10, KEYS[ch.lower()]
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()