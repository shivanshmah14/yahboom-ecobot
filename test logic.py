"""Hardware-free test of the detector + state machine using synthetic frames and a fake clock.
Run: python test_logic.py"""
import numpy as np
import cv2
from ecobot import Config, ColorDetector, ColorClassifier, Brain, State

H, W = 480, 640
YEL, BLU = (0, 255, 255), (255, 0, 0)      # BGR


def frame(*rects):
    f = np.full((H, W, 3), 60, np.uint8)   # grey "floor"
    for (x, y, w, h, c) in rects:
        cv2.rectangle(f, (x, y), (x + w, y + h), c, -1)
    return f


def new_brain():
    cfg = Config()
    return Brain(cfg, ColorDetector(cfg), ColorClassifier()), cfg


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name


b, cfg = new_brain()
t = 0.0
# 1 empty floor -> spin to search
vx, vz = b.step(frame(), t)
check("empty floor: spins", b.state is State.SEARCH and vz != 0 and vx == 0)
# 2 a big yellow sheet alone is NOT an item
vx, vz = b.step(frame((100, 100, 300, 250, YEL)), t)
check("big yellow sheet ignored during search", b.state is State.SEARCH)
# 3 small yellow item right of centre -> approach, turn right (negative vz)
vx, vz = b.step(frame((450, 200, 40, 40, YEL)), t)
check("yellow item locked as plastic", b.state is State.APPROACH and b.label == "plastic")
vx, vz = b.step(frame((450, 200, 40, 40, YEL)), t + 0.1)
check("steers right and drives forward", vx > 0 and vz < 0)
# 4 item low and centred -> captured
vx, vz = b.step(frame((300, 410, 40, 40, YEL)), t + 0.2)
check("item in sweeper -> CARRY", b.state is State.CARRY)
# 5 carrying, blue sheet visible must be ignored (needs yellow)
vx, vz = b.step(frame((50, 50, 250, 200, BLU)), t + 0.3)
check("carrying plastic ignores blue sheet", vz > 0 and vx == 0)
# 6 yellow sheet left -> drive towards it, turn left (positive vz)
vx, vz = b.step(frame((20, 100, 220, 200, YEL)), t + 0.4)
check("steers to yellow sheet", vx > 0 and vz > 0)
# 7 huge yellow sheet -> deposit
vx, vz = b.step(frame((0, 0, 600, 400, YEL)), t + 0.5)
check("sheet fills view -> DEPOSIT", b.state is State.DEPOSIT)
# 8 timed states advance with the clock
b.step(frame(), t + 0.5 + cfg.deposit_time + 0.1)
check("-> BACK_OFF", b.state is State.BACK_OFF)
b.step(frame(), t + 3.2)
check("-> TURN_AROUND", b.state is State.TURN_AROUND)
b.step(frame(), t + 7.5)
check("-> SEARCH again, item cleared", b.state is State.SEARCH and b.label is None)

# paper goes to blue
b, cfg = new_brain()
b.step(frame((300, 200, 40, 40, BLU)), 0)
check("blue item = paper", b.label == "paper")
b.step(frame((300, 410, 40, 40, BLU)), 0.1)
b.step(frame((20, 100, 220, 200, YEL)), 0.2)
vx, vz = b.step(frame((20, 100, 220, 200, YEL)), 0.3)
check("paper ignores yellow sheet", vx == 0 and vz > 0)

# give up if no sheet ever appears
b, cfg = new_brain()
b.step(frame((300, 200, 40, 40, YEL)), 0)
b.step(frame((300, 410, 40, 40, YEL)), 0.1)
b.step(frame(), 0.2 + cfg.carry_timeout)
check("carry timeout -> BACK_OFF", b.state is State.BACK_OFF)

# item lost while approaching
b, cfg = new_brain()
b.step(frame((300, 200, 40, 40, YEL)), 0)
b.step(frame(), 0.1 + cfg.lost_timeout)
check("lost item -> SEARCH", b.state is State.SEARCH)



# ---------------- floor mode (real plastic / paper, colour-independent) ----------------
class StubML:
    """Pretends to be the ML model: says 'paper' for dark-brown blobs, 'plastic' for anything else."""
    def classify(self, frame, blob):
        x, y, w, h = blob.bbox
        return "paper" if frame[y + h // 2, x + w // 2][2] > 100 and frame[y + h // 2, x + w // 2][0] < 80 else "plastic"


def floor_frame(*rects):
    rng = np.random.default_rng(0)
    f = np.full((H, W, 3), (110, 120, 130), np.int16) + rng.integers(-6, 7, (H, W, 3))
    f = np.clip(f, 0, 255).astype(np.uint8)                     # noisy grey-brown floor
    cv2.rectangle(f, (0, 0), (W, 120), (230, 230, 230), -1)      # bright wall band at the top
    for (x, y, w, h, c) in rects:
        cv2.rectangle(f, (x, y), (x + w, y + h), c, -1)
    return f


def new_floor_brain():
    cfg = Config()
    cfg.finder = "floor"
    return Brain(cfg, ColorDetector(cfg), StubML()), cfg


b, cfg = new_floor_brain()
vx, vz = b.step(floor_frame(), 0)
check("floor mode: empty noisy floor + wall = no false item", b.state is State.SEARCH)
b.step(floor_frame((20, 150, 560, 300, YEL)), 0.05)
check("floor mode: big yellow sheet is not an item", b.state is State.SEARCH)
# an item that is NOT yellow or blue (reddish-brown) is found
b.step(floor_frame((450, 280, 50, 50, (30, 60, 150))), 0.1)
check("floor mode: uncoloured item found and classified", b.state is State.APPROACH and b.label == "paper")
vx, vz = b.step(floor_frame((450, 280, 50, 50, (30, 60, 150))), 0.2)
check("floor mode: steers right towards it", vx > 0 and vz < 0)
b.step(floor_frame((300, 410, 50, 40, (30, 60, 150))), 0.3)
check("floor mode: item low in view -> CARRY (paper)", b.state is State.CARRY and b.label == "paper")
vx, vz = b.step(floor_frame((20, 150, 300, 250, BLU)), 0.4)
check("floor mode: paper goes to the BLUE sheet", vx > 0 and vz > 0)
# item lying on top of the wall band only must be ignored
b, cfg = new_floor_brain()
b.step(floor_frame((100, 30, 60, 60, (30, 60, 150))), 0)
check("floor mode: things above the horizon line ignored", b.state is State.SEARCH)
print("\nAll logic tests passed.")