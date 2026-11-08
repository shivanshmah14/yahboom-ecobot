import cv2
import numpy as np

# Initialize video capture from the default camera (0)
cap = cv2.VideoCapture(0)

# Expanded HSV color ranges to catch wider shades (darker, lighter, less saturated)
YELLOW_LOWER = np.array([15, 40, 40])
YELLOW_UPPER = np.array([35, 255, 255])

BLUE_LOWER = np.array([90, 40, 40])
BLUE_UPPER = np.array([140, 255, 255])

print("Starting color detector with expanded ranges. Press 'q' to quit.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert BGR frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create color masks
        yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
        blue_mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

        # Find contours for yellow (plastic)
        yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in yellow_contours:
            if cv2.contourArea(c) > 500:  # Minimum area filter
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, "Plastic (Yellow)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Find contours for blue (paper)
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in blue_contours:
            if cv2.contourArea(c) > 500:  # Minimum area filter
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "Paper (Blue)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Show the live camera feed with detections highlighted
        cv2.imshow("Garbage Sorting Detector", frame)

        # Press 'q' or 'Q' to exit the video window
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()