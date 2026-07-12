from flask import Flask, request
import os
import piexif
import json
import cv2

app = Flask(__name__)

SERVER_SAVE_DIR = os.path.expanduser("~/Desktop/Received_Captures")
CROP_SAVE_DIR = os.path.expanduser("~/Desktop/Cropped_People")

os.makedirs(SERVER_SAVE_DIR, exist_ok=True)
os.makedirs(CROP_SAVE_DIR, exist_ok=True)


def parse_embedded_metadata(image_path):
    try:
        exif_data = piexif.load(image_path)
        raw_comment = exif_data["Exif"].get(0x9286, b"")
        if raw_comment.startswith(b"ASCII\x00\x00\x00"):
            json_bytes = raw_comment[8:]
            return json.loads(json_bytes.decode('utf-8'))
    except Exception as e:
        print(f"[METADATA ERROR] Failed to parse EXIF markers: {e}")
    return None


@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'image' not in request.files:
        return "Missing image payload", 400

    file = request.files['image']
    if file.filename == '':
        return "Invalid file target entry", 400

    target_save_path = os.path.join(SERVER_SAVE_DIR, file.filename)
    file.save(target_save_path)
    print(f"\n📥 [RECEIVED] Saved full image: {file.filename}")

    meta_payload = parse_embedded_metadata(target_save_path)

    if meta_payload:
        print("======== EXTRACTED SERVER METADATA ========")
        print(f"📊 Raw Keys Found: {list(meta_payload.keys())}")  # See everything the client sent
        print(f"👕 Top: {meta_payload.get('top_item')} | 👖 Bottom: {meta_payload.get('bottom_item')}")

        # Look for the coordinates key
        person_box = meta_payload.get('person_bounding_box')
        shirt_box = meta_payload.get('shirt_bounding_box')

        print(f"🎯 'person_bounding_box' values received: {person_box}")
        print(f"👔 'shirt_bounding_box' values received: {shirt_box}")
        print("===========================================")

        # FALLBACK SAFETY MATRIX: If person_bounding_box is missing or empty, try to crop just the shirt box
        chosen_box = person_box if (person_box and person_box != [0, 0, 0, 0]) else shirt_box

        if chosen_box and chosen_box != [0, 0, 0, 0]:
            try:
                full_image = cv2.imread(target_save_path)
                if full_image is not None:
                    x1, y1, x2, y2 = chosen_box
                    img_h, img_w, _ = full_image.shape

                    # Force type matching to integers to prevent array slicing errors
                    x1_crop = int(max(0, x1))
                    y1_crop = 0
                    x2_crop = int(min(img_w, x2))
                    y2_crop = int(min(img_h, y2))

                    print(f"📐 Applying crop slice matrix coordinates: [{x1_crop}:{x2_crop}, {y1_crop}:{y2_crop}]")
                    cropped_person = full_image[y1_crop:y2_crop, x1_crop:x2_crop]

                    if cropped_person.size > 0:
                        crop_filename = f"crop_{file.filename}"
                        crop_save_path = os.path.join(CROP_SAVE_DIR, crop_filename)
                        cv2.imwrite(crop_save_path, cropped_person)
                        print(f"✂️ [SUCCESS] Crop saved to: {crop_filename}")
                    else:
                        print("⚠️ [CROP ERROR] Resulting slice yielded a 0-pixel empty image surface.")
                else:
                    print("⚠️ [IMAGE ERROR] OpenCV could not open the saved image file asset.")
            except Exception as e:
                print(f"❌ [CRASH LOG] Cropping processing error: {e}")
        else:
            print("⚠️ [CROP SKIPPED] All received bounding box coordinates are either missing or [0, 0, 0, 0].")
    else:
        print("⚠️ [EXIF MISSING] No metadata payload dictionary found in this file.")

    return f"Processed", 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
