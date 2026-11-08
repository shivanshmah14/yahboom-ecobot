"""
HSV calibration tool.  python calibrate.py [--camera 0]

Three windows open at once - nothing to switch:
  "yellow sliders" : sliders for the yellow range
  "blue sliders"   : sliders for the blue range
  "preview"        : camera | yellow mask | blue mask
Move each slider set until ONLY your item/sheet is white in its mask.
Click the preview window, then:  s = save to hsv_ranges.json    q = quit
"""
import argparse
import json
import traceback
import cv2
import numpy as np
from ecobot import Config, HSV_FILE

NAMES = ["H low", "H high", "S low", "S high", "V low", "V high"]
MAXV = [179, 179, 255, 255, 255, 255]


def to_values(r):
    return [r["lower"][0], r["upper"][0], r["lower"][1], r["upper"][1], r["lower"][2], r["upper"][2]]


def to_range(v):
    return {"lower": [v[0], v[2], v[4]], "upper": [v[1], v[3], v[5]]}


def make_sliders(win, r):
    cv2.namedWindow(win)
    for n, m, v in zip(NAMES, MAXV, to_values(r)):
        cv2.createTrackbar(n, win, v, m, lambda _: None)


def read_sliders(win):
    return to_range([cv2.getTrackbarPos(n, win) for n in NAMES])


def run(camera):
    cfg = Config.load()
    wins = {"yellow": "yellow sliders", "blue": "blue sliders"}
    for color, win in wins.items():
        make_sliders(win, cfg.hsv[color])
    cv2.namedWindow("preview")
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("Could not open the camera.")
        return
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame failed.")
            break
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        panels = [frame.copy()]
        for color, win in wins.items():
            cfg.hsv[color] = read_sliders(win)
            r = cfg.hsv[color]
            mask = cv2.inRange(hsv, np.array(r["lower"]), np.array(r["upper"]))
            panels.append(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        for p, t in zip(panels, ["camera", "yellow mask", "blue mask"]):
            cv2.putText(p, t, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        both = np.hstack(panels)
        scale = 1200 / both.shape[1]                    # fit on screen
        cv2.imshow("preview", cv2.resize(both, None, fx=scale, fy=scale))

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), ord("Q")):
            break
        if k in (ord("s"), ord("S")):
            with open(HSV_FILE, "w") as f:
                json.dump(cfg.hsv, f, indent=2)
            print("Saved", HSV_FILE, cfg.hsv)
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()
    try:
        run(args.camera)
    except Exception:
        traceback.print_exc()          # show the real error instead of silently closing
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()