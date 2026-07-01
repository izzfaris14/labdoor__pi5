import cv2
import time
from picamera2 import Picamera2
from edge_impulse_linux.image import ImageImpulseRunner
from gpiozero import OutputDevice

modelPath = "/home/defaultpi/attire-detector/ITPAttire.eim"

# --- GATEKEEPING CONFIGURATION ---
# Replace with the exact label strings Linus used in Edge Impulse
REQUIRED_ATTIRE = {"covered top", "pants"}
CONFIDENCE_THRESHOLD = 0.60  # 60% confidence required to count a detection

# --- HARDWARE CONFIGURATION ---
# Assuming you are using a relay connected to GPIO pin 17
# active_high=True means sending power turns the relay ON
RELAY_PIN = 17
door_lock = OutputDevice(RELAY_PIN, active_high=True, initial_value=False)
DOOR_OPEN_DURATION = 5.0  # How long the door stays unlocked in seconds


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

    # State variables for the door timer
    door_is_unlocked = False
    unlock_timestamp = 0.0

    try:
        while True:
            frameRgb = picam2.capture_array()
            features, cropped = runner.get_features_from_image(frameRgb)
            inferenceResult = runner.classify(features)

            displayFrame = cv2.cvtColor(frameRgb, cv2.COLOR_RGB2BGR)

            # Set to keep track of what we see in THIS specific frame
            current_frame_detections = set()

            if "bounding_boxes" in inferenceResult["result"]:
                for bb in inferenceResult["result"]["bounding_boxes"]:
                    if bb["value"] > CONFIDENCE_THRESHOLD:
                        # 1. Record the valid detection
                        current_frame_detections.add(bb["label"])

                        # 2. Draw the bounding boxes and text (Visuals)
                        x = int((bb["x"] / cropped.shape[1]) * 640)
                        y = int((bb["y"] / cropped.shape[0]) * 480)
                        w = int((bb["width"] / cropped.shape[1]) * 640)
                        h = int((bb["height"] / cropped.shape[0]) * 480)

                        cv2.rectangle(displayFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        label_text = f'{bb["label"]} {bb["value"]:.0%}'
                        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                        cv2.rectangle(displayFrame, (x, y - th - 10), (x + tw, y), (0, 255, 0), -1)
                        cv2.putText(displayFrame, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # --- GATEKEEPING LOGIC ---
            # Check if all required items are currently in view
            criteria_met = REQUIRED_ATTIRE.issubset(current_frame_detections)

            if criteria_met and not door_is_unlocked:
                print("Access Granted! Required attire detected.")
                door_lock.on()  # Trigger the relay to unlock
                door_is_unlocked = True
                unlock_timestamp = time.time()

            # Timer to lock the door again without freezing the camera feed
            if door_is_unlocked and (time.time() - unlock_timestamp) >= DOOR_OPEN_DURATION:
                print("Door locking automatically.")
                door_lock.off()  # Turn off relay
                door_is_unlocked = False

            # Add a visual indicator for door status on the screen
            status_color = (0, 255, 0) if door_is_unlocked else (0, 0, 255)
            status_text = "DOOR UNLOCKED" if door_is_unlocked else "DOOR LOCKED"
            cv2.putText(displayFrame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            cv2.imshow('Lab Attire Scanner', displayFrame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Cleanup hardware and software safely
        door_lock.off()
        picam2.stop()
        cv2.destroyAllWindows()
        runner.stop()


if __name__ == "__main__":
    main()