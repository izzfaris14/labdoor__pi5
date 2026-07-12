import cv2
import time
import os
from picamera2 import Picamera2
from edge_impulse_linux.image import ImageImpulseRunner
from gpiozero import OutputDevice

modelPath = "/home/defaultpi/attire-detector/ITPAttire.eim"

# --- GATEKEEPING CONFIGURATION ---
REQUIRED_ATTIRE = {"covered top", "pants"}
INAPPROPRIATE_ATTIRE = {"uncovered top", "shorts", "uncovered shoes"}
CONFIDENCE_THRESHOLD = 0.60  # 60% confidence required to count a detection

# --- HARDWARE CONFIGURATION ---
RELAY_PIN = 17
door_lock = OutputDevice(RELAY_PIN, active_high=True, initial_value=False)
DOOR_OPEN_DURATION = 5.0  # How long the door stays unlocked in seconds

# --- CAPTURE CONFIGURATION ---
VIOLATION_COOLDOWN = 5.0  # Wait 5 seconds before taking another picture of a violator
VIOLATION_DIR = "/home/defaultpi/attire-detector/violations"

# Create the directory for violation images if it doesn't exist
if not os.path.exists(VIOLATION_DIR):
    os.makedirs(VIOLATION_DIR)


def main():
    runner = ImageImpulseRunner(modelPath)
    modelInfo = runner.init()
    print("Model initialised successfully.")

    print("Binding physical CSI interface via Picamera2 DMA...")
    picam2 = Picamera2()

    cameraConfig = picam2.create_video_configuration(main={"size": (640, 480), "format": "BGR888"})
    picam2.configure(cameraConfig)
    picam2.start()

    print("Camera active. Rendering to local LCD. Press 'q' to quit.")

    # State variables
    door_is_unlocked = False
    unlock_timestamp = 0.0
    last_capture_time = 0.0

    try:
        while True:
            frameRgb = picam2.capture_array()
            features, cropped = runner.get_features_from_image(frameRgb)
            inferenceResult = runner.classify(features)

            # Convert to BGR for OpenCV display and saving
            displayFrame = cv2.cvtColor(frameRgb, cv2.COLOR_RGB2BGR)

            current_frame_detections = set()

            if "bounding_boxes" in inferenceResult["result"]:
                for bb in inferenceResult["result"]["bounding_boxes"]:
                    if bb["value"] > CONFIDENCE_THRESHOLD:
                        label = bb["label"]
                        current_frame_detections.add(label)

                        # Determine box color: Red (0, 0, 255) for inappropriate, Green (0, 255, 0) otherwise
                        if label in INAPPROPRIATE_ATTIRE:
                            box_color = (0, 0, 255)
                        else:
                            box_color = (0, 255, 0)

                        x = int((bb["x"] / cropped.shape[1]) * 640)
                        y = int((bb["y"] / cropped.shape[0]) * 480)
                        w = int((bb["width"] / cropped.shape[1]) * 640)
                        h = int((bb["height"] / cropped.shape[0]) * 480)

                        cv2.rectangle(displayFrame, (x, y), (x + w, y + h), box_color, 2)

                        label_text = f'{label} {bb["value"]:.0%}'
                        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                        cv2.rectangle(displayFrame, (x, y - th - 10), (x + tw, y), box_color, -1)
                        # Text is drawn in black (0,0,0) for contrast
                        cv2.putText(displayFrame, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # --- GATEKEEPING & CAPTURE LOGIC ---

            # 1. Check for any violations
            # Use intersection to see if any items in the frame match the inappropriate list
            active_violations = INAPPROPRIATE_ATTIRE.intersection(current_frame_detections)

            if active_violations:
                current_time = time.time()
                # Only capture an image if the cooldown period has passed
                if (current_time - last_capture_time) >= VIOLATION_COOLDOWN:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"{VIOLATION_DIR}/violation_{timestamp}.jpg"

                    # Save the frame (this includes the drawn red bounding boxes)
                    cv2.imwrite(filename, displayFrame)
                    print(f"ALERT: Violation detected {active_violations}. Image saved to {filename}")

                    last_capture_time = current_time

            # 2. Door Control Logic
            criteria_met = REQUIRED_ATTIRE.issubset(current_frame_detections)

            # We only grant access if ALL required items are present AND NO violations are detected
            if criteria_met and not active_violations and not door_is_unlocked:
                print("Access Granted! Required attire detected.")
                door_lock.on()
                door_is_unlocked = True
                unlock_timestamp = time.time()

            # Timer to lock the door again
            if door_is_unlocked and (time.time() - unlock_timestamp) >= DOOR_OPEN_DURATION:
                print("Door locking automatically.")
                door_lock.off()
                door_is_unlocked = False

            # Visual indicator for door status on the screen
            status_color = (0, 255, 0) if door_is_unlocked else (0, 0, 255)
            status_text = "DOOR UNLOCKED" if door_is_unlocked else "DOOR LOCKED"
            cv2.putText(displayFrame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            cv2.imshow('Lab Attire Scanner', displayFrame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        door_lock.off()
        picam2.stop()
        cv2.destroyAllWindows()
        runner.stop()


if __name__ == "__main__":
    main()