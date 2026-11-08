"""
EcoBot AI - colour-based waste sorting robot (Yahboom ROSMASTER).

Structure (one class per job, so each part can be replaced or tested alone):
  Config          - all tunable numbers in one place (+ HSV ranges from hsv_ranges.json)
  ColorDetector   - OpenCV HSV masks -> list of Blob objects (items and drop zones)
  ItemClassifier  - decides "plastic" / "paper" for an item.  ColorClassifier is the
                    stand-in; replace it with your ML model (see MLClassifier below).
  Drive           - wraps Rosmaster_Lib; falls back to printing in --dry-run
  Brain           - the state machine: SEARCH -> APPROACH -> CARRY -> DEPOSIT
                    -> BACK_OFF -> TURN_AROUND -> SEARCH

Run:
  python ecobot.py --dry-run --show      # laptop test, no robot, shows camera window
  python ecobot.py --dry-run --show --model model.pt --finder floor
                                         # REAL plastic/paper: find anything not floor, ML decides
  python ecobot.py --show                # on the robot, with a display attached
  python ecobot.py                       # on the robot, headless
Press q in the window (or Ctrl+C) to stop; the robot is always stopped on exit.
With --show and --model: each box shows what the model thinks (label + confidence). If the LARGEST
box is wrong, press p (it is plastic), a (paper) or n (not litter) to save it as a training photo
in dataset/<class>/ - then retrain.
"""
import argparse
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np

HSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hsv_ranges.json")


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass
class Config:
    # HSV ranges (OpenCV: H 0-179, S 0-255, V 0-255). Starting points only -
    # run calibrate.py with YOUR items, sheets and room lighting.
    hsv: dict = field(default_factory=lambda: {
        "yellow": {"lower": [20, 100, 100], "upper": [35, 255, 255]},
        "blue":   {"lower": [100, 100, 60], "upper": [130, 255, 255]},
    })
    # Blob sizes as a FRACTION of the frame area (works at any camera resolution)
    item_min_frac: float = 0.002     # smaller = noise
    item_max_frac: float = 0.08      # bigger = it's a drop-zone sheet, not an item
    zone_min_frac: float = 0.10      # a sheet must fill at least this much of the frame
    zone_arrive_frac: float = 0.35   # sheet this big = we are at the sheet
    capture_y_frac: float = 0.80     # item centre below this image row = in the sweeper
    # Colour-independent item finder ("floor" mode, used with the ML model)
    finder: str = "color"            # "color" = yellow/blue blobs, "floor" = anything unlike the floor
    floor_thresh: float = 25.0       # Lab colour distance from the floor that counts as "not floor"
    roi_top_frac: float = 0.35       # ignore the top part of the image (wall / horizon)
    ignore_bottom_frac: float = 0.0  # ignore the bottom part (e.g. your own sweeper in view)
    # Item colour -> waste type -> drop-zone colour
    zone_for = {"plastic": "yellow", "paper": "blue"}
    # Speeds (m/s and rad/s) - keep SLOW at first
    approach_speed: float = 0.12
    carry_speed: float = 0.10
    steer_gain: float = 0.8          # rad/s per unit of normalised error (-1..1)
    search_spin: float = 0.5
    carry_spin: float = 0.3
    # Timings (seconds)
    lost_timeout: float = 1.5        # lose the item this long -> back to SEARCH
    carry_timeout: float = 30.0      # can't find the sheet -> give up
    deposit_time: float = 1.0
    backoff_time: float = 1.5
    turn_speed: float = 0.8
    turn_time: float = 3.9           # ~ pi / turn_speed for a 180 degree turn; tune on floor

    @classmethod
    def load(cls, path=HSV_FILE):
        cfg = cls()
        if os.path.exists(path):
            with open(path) as f:
                cfg.hsv.update(json.load(f))
        return cfg


# ----------------------------------------------------------------------------
# Vision
# ----------------------------------------------------------------------------
@dataclass
class Blob:
    color: str
    cx: int
    cy: int
    area: float          # in pixels
    bbox: tuple          # x, y, w, h
    text: str = ""       # what the classifier thought (shown on screen)


def crop_for(frame, blob, pad=0.15):
    """The picture the classifier sees (and what the s-key training photos use)."""
    x, y, w, h = blob.bbox
    px, py = int(w * pad), int(h * pad)
    H, W = frame.shape[:2]
    return frame[max(0, y - py):min(H, y + h + py), max(0, x - px):min(W, x + w + px)]


class ColorDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.kernel = np.ones((5, 5), np.uint8)
        self.kernel_big = np.ones((9, 9), np.uint8)
        self.last_floor_mask = None       # for the --mask debug window

    # --- colour masks -------------------------------------------------------
    def _color_mask(self, frame, color):
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        r = self.cfg.hsv[color]
        mask = cv2.inRange(hsv, np.array(r["lower"]), np.array(r["upper"]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)    # remove specks
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)   # fill holes

    def _to_blobs(self, mask, color, frame_area, min_frac, max_frac=1.0):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if min_frac <= area / frame_area <= max_frac:
                m = cv2.moments(c)
                if m["m00"] == 0:
                    continue
                out.append(Blob(color, int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]),
                                area, cv2.boundingRect(c)))
        return sorted(out, key=lambda b: b.area, reverse=True)

    def _blobs(self, frame, color, min_frac, max_frac=1.0):
        h, w = frame.shape[:2]
        return self._to_blobs(self._color_mask(frame, color), color, float(h * w), min_frac, max_frac)

    # --- colour-independent finder: "anything that is not floor" -------------
    def _floor_blobs(self, frame, min_frac, max_frac=1.0):
        cfg = self.cfg
        h, w = frame.shape[:2]
        top = int(cfg.roi_top_frac * h)
        bottom = int((1.0 - cfg.ignore_bottom_frac) * h)
        lab = cv2.cvtColor(cv2.GaussianBlur(frame, (7, 7), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
        # floor colour = median colour of the region of interest (floor fills most of it)
        ref = np.median(lab[top:bottom:4, ::4].reshape(-1, 3), axis=0)
        dist = np.linalg.norm(lab - ref, axis=2)
        mask = (dist > cfg.floor_thresh).astype(np.uint8) * 255
        mask[:top] = 0
        mask[bottom:] = 0
        # the drop-zone sheets are not items: blank out big yellow / blue areas (and their edges)
        for color in ("yellow", "blue"):
            cnts, _ = cv2.findContours(self._color_mask(frame, color), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) / (h * w) >= cfg.zone_min_frac:
                    cv2.drawContours(mask, [c], -1, 0, thickness=cv2.FILLED)
                    cv2.drawContours(mask, [c], -1, 0, thickness=15)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_big)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_big)
        self.last_floor_mask = mask
        return self._to_blobs(mask, "item", float(h * w), min_frac, max_frac)

    # --- what Brain calls -----------------------------------------------------
    def find_items(self, frame):
        """Candidate items. Colour mode: small yellow/blue blobs. Floor mode: anything not floor."""
        if self.cfg.finder == "floor":
            return self._floor_blobs(frame, self.cfg.item_min_frac, self.cfg.item_max_frac)
        items = []
        for color in ("yellow", "blue"):
            items += self._blobs(frame, color, self.cfg.item_min_frac, self.cfg.item_max_frac)
        return sorted(items, key=lambda b: b.area, reverse=True)

    def find_color(self, frame, color):
        """Every blob of one kind, any size (used while tracking a locked item)."""
        if color == "item":
            return self._floor_blobs(frame, self.cfg.item_min_frac)
        return self._blobs(frame, color, self.cfg.item_min_frac)

    def find_zones(self, frame, color):
        """Only LARGE blobs of one colour = the drop-zone sheet."""
        return self._blobs(frame, color, self.cfg.zone_min_frac)


class ItemClassifier:
    """Interface: look at a detected item, return 'plastic', 'paper' or None."""
    def classify(self, frame, blob: Blob) -> Optional[str]:
        raise NotImplementedError


class ColorClassifier(ItemClassifier):
    """Stand-in: yellow item = plastic, blue item = paper."""
    def classify(self, frame, blob):
        return {"yellow": "plastic", "blue": "paper"}.get(blob.color)


class MLClassifier(ItemClassifier):
    """
    Uses the model trained by train_classifier.py (MobileNetV3-large CNN, via the timm library).
    Looks at the crop around a detected blob and returns 'plastic', 'paper' or None
    (None = the model says 'none' or is not confident enough).
    Use with  --finder floor  so items are located as "anything that is not floor", not by colour.
    """
    def __init__(self, model_path="model.pt", min_conf=0.7, pad=0.15):
        import torch
        import timm
        ckpt = torch.load(model_path, map_location="cpu")
        self.classes = ckpt["classes"]
        self.torch = torch
        self.model = timm.create_model(ckpt.get("arch", "mobilenetv3_large_100"), pretrained=False,
                                       num_classes=len(self.classes))
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()
        self.min_conf, self.pad = min_conf, pad
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def classify(self, frame, blob):
        crop = crop_for(frame, blob, self.pad)
        if crop.size == 0:
            return None
        img = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = ((img - self.mean) / self.std).transpose(2, 0, 1)
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(self.torch.from_numpy(img)[None]), 1)[0]
        conf, idx = probs.max(0)
        label = self.classes[int(idx)]
        blob.text = f"{label} {float(conf):.2f}"
        return label if label in ("plastic", "paper") and float(conf) >= self.min_conf else None


# ----------------------------------------------------------------------------
# Robot drive
# ----------------------------------------------------------------------------
class Drive:
    MAX_V, MAX_W = 0.25, 1.5     # safety clamps

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.bot = None
        self._last_print = 0.0
        if not dry_run:
            from Rosmaster_Lib import Rosmaster      # only exists on the robot
            self.bot = Rosmaster()
            self.bot.create_receive_threading()

    def move(self, vx, vz):
        vx = max(-self.MAX_V, min(self.MAX_V, vx))
        vz = max(-self.MAX_W, min(self.MAX_W, vz))
        if self.bot:
            self.bot.set_car_motion(vx, 0, vz)       # vx fwd m/s, vy sideways, vz rad/s (+ = left)
        elif time.monotonic() - self._last_print > 0.5:
            print(f"  [dry-run] vx={vx:+.2f} vz={vz:+.2f}")
            self._last_print = time.monotonic()

    def stop(self):
        if self.bot:
            self.bot.set_car_motion(0, 0, 0)


# ----------------------------------------------------------------------------
# State machine
# ----------------------------------------------------------------------------
class State(Enum):
    SEARCH = auto()
    APPROACH = auto()
    CARRY = auto()
    DEPOSIT = auto()
    BACK_OFF = auto()
    TURN_AROUND = auto()


class Brain:
    def __init__(self, cfg: Config, detector: ColorDetector, classifier: ItemClassifier):
        self.cfg, self.det, self.clf = cfg, detector, classifier
        self.state = State.SEARCH
        self.label = None            # 'plastic' / 'paper' while an item is locked
        self.item_color = None
        self.last_xy = None
        self.t_state = 0.0
        self.t_seen = 0.0
        self.debug = []              # blobs to draw on screen

    def _go(self, state, now):
        print(f"[{state.name}]" + (f" carrying {self.label}" if self.label else ""))
        self.state, self.t_state = state, now

    @staticmethod
    def _error(blob, width):
        return (blob.cx - width / 2) / (width / 2)    # -1 (left) .. +1 (right)

    def step(self, frame, now):
        """One camera frame in -> (forward speed, turn speed) out."""
        cfg, h, w = self.cfg, frame.shape[0], frame.shape[1]
        s = self.state
        self.debug = []

        if s is State.SEARCH:
            items = self.det.find_items(frame)
            self.debug = items
            labels = [(it, self.clf.classify(frame, it)) for it in items[:5]]   # largest 5 get a label on screen
            for it, label in labels:
                if label:
                    self.label, self.item_color = label, it.color
                    self.last_xy, self.t_seen = (it.cx, it.cy), now
                    self._go(State.APPROACH, now)
                    return 0.0, 0.0
            return 0.0, cfg.search_spin

        if s is State.APPROACH:
            blobs = self.det.find_color(frame, self.item_color)
            self.debug = blobs
            if not blobs:
                if now - self.t_seen > cfg.lost_timeout:
                    self.label = None
                    self._go(State.SEARCH, now)
                return 0.0, 0.0
            # stay locked on the blob nearest to where the item was last seen
            lx, ly = self.last_xy
            it = min(blobs, key=lambda b: (b.cx - lx) ** 2 + (b.cy - ly) ** 2)
            self.last_xy, self.t_seen = (it.cx, it.cy), now
            it.text = f"locked: {self.label}"
            err = self._error(it, w)
            if it.cy > cfg.capture_y_frac * h and abs(err) < 0.3:
                self._go(State.CARRY, now)
                return 0.0, 0.0
            return cfg.approach_speed, -cfg.steer_gain * err

        if s is State.CARRY:
            zones = self.det.find_zones(frame, cfg.zone_for[self.label])
            self.debug = zones
            if now - self.t_state > cfg.carry_timeout:
                print("  could not find the drop zone - giving up")
                self._go(State.BACK_OFF, now)
                return 0.0, 0.0
            if not zones:
                return 0.0, cfg.carry_spin
            z = zones[0]
            if z.area / (w * h) >= cfg.zone_arrive_frac:
                self._go(State.DEPOSIT, now)
                return 0.0, 0.0
            return cfg.carry_speed, -cfg.steer_gain * self._error(z, w)

        if s is State.DEPOSIT:                       # push the item onto the sheet
            if now - self.t_state > cfg.deposit_time:
                self._go(State.BACK_OFF, now)
                return 0.0, 0.0
            return cfg.carry_speed, 0.0

        if s is State.BACK_OFF:
            if now - self.t_state > cfg.backoff_time:
                self._go(State.TURN_AROUND, now)
                return 0.0, 0.0
            return -0.15, 0.0

        if s is State.TURN_AROUND:
            if now - self.t_state > cfg.turn_time:
                self.label = self.item_color = None
                self._go(State.SEARCH, now)
                return 0.0, 0.0
            return 0.0, cfg.turn_speed

        return 0.0, 0.0


# ----------------------------------------------------------------------------
def save_crop(frame, blob, cls, out="dataset"):
    """Save the classifier's view of a blob as a training photo (used by the p / a / n keys)."""
    os.makedirs(os.path.join(out, cls), exist_ok=True)
    img = cv2.resize(crop_for(frame, blob), (224, 224))
    path = os.path.join(out, cls, f"live_{int(time.time() * 1000)}.jpg")
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def draw_debug(frame, brain):
    for b in brain.debug:
        x, y, w, h = b.bbox
        col = {"yellow": (0, 255, 255), "blue": (255, 0, 0)}.get(b.color, (0, 255, 0))
        cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
        if b.text:
            cv2.putText(frame, b.text, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.putText(frame, brain.state.name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="show camera window with detections")
    ap.add_argument("--model", default=None, help="use the ML classifier, e.g. --model model.pt")
    ap.add_argument("--finder", choices=["color", "floor"], default="color",
                    help="how items are found: color = yellow/blue test items, floor = anything not floor (needs --model)")
    ap.add_argument("--min-conf", type=float, default=0.7, help="ML confidence needed (raise to 0.85 to cut false detections)")
    ap.add_argument("--floor-thresh", type=float, default=None, help="floor mode: lower (e.g. 12) = more sensitive, higher (e.g. 40) = fewer false boxes")
    ap.add_argument("--mask", action="store_true", help="floor mode: also show the black/white floor-difference picture")
    ap.add_argument("--dry-run", action="store_true", help="do not drive; print commands")
    args = ap.parse_args()

    if args.finder == "floor" and not args.model:
        ap.error("--finder floor needs --model model.pt (something must decide plastic vs paper)")
    cfg = Config.load()
    cfg.finder = args.finder
    if args.floor_thresh is not None:
        cfg.floor_thresh = args.floor_thresh
    classifier = MLClassifier(args.model, min_conf=args.min_conf) if args.model else ColorClassifier()
    brain = Brain(cfg, ColorDetector(cfg), classifier)
    drive = Drive(dry_run=args.dry_run)
    cap = cv2.VideoCapture(args.camera)
    print("EcoBot running. q / Ctrl+C to stop.")
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Camera frame failed - stopping.")
                break
            vx, vz = brain.step(frame, time.monotonic())
            drive.move(vx, vz)
            if args.show:
                raw = frame.copy()                       # clean copy for training photos
                draw_debug(frame, brain)
                cv2.imshow("EcoBot", frame)
                if args.mask and brain.det.last_floor_mask is not None:
                    cv2.imshow("floor mask", brain.det.last_floor_mask)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), ord("Q"), 27):
                    break
                cls = {ord("p"): "plastic", ord("a"): "paper", ord("n"): "none"}.get(k)
                if cls and brain.debug:                  # teach it: label the LARGEST box on screen
                    print("saved", save_crop(raw, brain.debug[0], cls))
    except KeyboardInterrupt:
        pass
    finally:
        drive.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()