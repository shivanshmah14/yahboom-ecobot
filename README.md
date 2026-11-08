# Yahboom Garbage Bot

A Python project for a Yahboom garbage-sorting robot using OpenCV-based color detection and a simple state machine.

## Features
- Color calibration tool for yellow and blue object ranges
- Logic test for the state machine without hardware
- Dry-run mode that prints drive commands instead of moving the robot
- Camera-based detection and sorting flow for plastic and paper

## Files
- `ecobot.py` — main robot logic and state machine
- `calibrate.py` — HSV calibration interface for tuning colors
- `test logic.py` — logic-only validation test
- `detector.py` — simplified color detector prototype

## Setup
Install the required packages:

```bash
pip install opencv-python numpy
```

## Run tests
```bash
python "test logic.py"
```

## Calibrate colors
```bash
python calibrate.py
```

Use:
- `y` / `b` to switch between yellow and blue
- `s` to save HSV values
- `q` to quit

## Dry run
```bash
python ecobot.py --dry-run --show
```

This prints motion commands instead of driving the robot.

## Notes
This project is intended for laptop testing first, then for robot testing with the Yahboom platform once the calibration is correct.
