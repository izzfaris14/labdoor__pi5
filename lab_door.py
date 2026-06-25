import cv2
from picamera2 import Picamera2
from edge_impulse_linux.image import ImageImpulseRunner

modelPath = "/home/defaultpi/attire-detector/ITPAttire.eim"


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

    try:
        while True:

            frameRgb = picam2.capture_array()

            features, cropped = runner.get_features_from_image(frameRgb)
            inferenceResult = runner.classify(features)

            displayFrame = cv2.cvtColor(frameRgb, cv2.COLOR_RGB2BGR)

            if "bounding_boxes" in inferenceResult["result"]:
                for bb in inferenceResult["result"]["bounding_boxes"]:
                    if bb["value"] > 0.5:
                        x = int((bb["x"] / cropped.shape[1]) * 640)
                        y = int((bb["y"] / cropped.shape[0]) * 480)
                        w = int((bb["width"] / cropped.shape[1]) * 640)
                        h = int((bb["height"] / cropped.shape[0]) * 480)

                        cv2.rectangle(displayFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        label = f'{bb["label"]} {bb["value"]:.0%}'
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                        cv2.rectangle(displayFrame, (x, y - th - 10), (x + tw, y), (0, 255, 0), -1)
                        cv2.putText(displayFrame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                        print(f'Detected: {label}')

            cv2.imshow('Lab Attire Scanner', displayFrame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        runner.stop()


if __name__ == "__main__":
    main()
