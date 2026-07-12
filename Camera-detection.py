import sys
import cv2
from ultralytics import YOLO
import time
import os
import json
import queue  # Thread-safe pipeline queue
import requests  # Clean network image streaming
from pathlib import Path

# --- EXIF METADATA INJECTION ENGINE ---
try:
    import piexif

    HAS_PIEXIF = True
except ImportError:
    HAS_PIEXIF = False
    print("[WARNING] 'piexif' library not found. Run 'pip install piexif'.")


def save_image_with_metadata(filename, frame, metadata_dict):
    """Encodes a frame to JPEG and injects a JSON metadata string into the EXIF UserComment tag."""
    success, img_encoded = cv2.imencode('.jpg', frame)
    if not success:
        return False

    img_bytes = img_encoded.tobytes()

    if not HAS_PIEXIF:
        with open(filename, 'wb') as f:
            f.write(img_bytes)
        return True

    json_meta = json.dumps(metadata_dict)
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    exif_dict["Exif"][0x9286] = b"ASCII\x00\x00\x00" + json_meta.encode('utf-8')

    try:
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, img_bytes, filename)
        return True
    except Exception as e:
        print(f"[EXIF WRITE ERROR] Fallback to direct write: {e}", flush=True)
        with open(filename, 'wb') as f:
            f.write(img_bytes)
        return False


# --- CROSS-PLATFORM PYQT/PYSIDE LAYER ---
try:
    from PyQt6 import QtWidgets, uic
    from PyQt6.QtCore import QThread, pyqtSignal, Qt
    from PyQt6.QtGui import QImage, QPixmap, QStandardItemModel, QStandardItem
except ImportError:
    from PySide6 import QtWidgets, QtUiTools
    from PySide6.QtCore import QThread, Signal as pyqtSignal, Qt
    from PySide6.QtGui import QImage, QPixmap, QStandardItemModel, QStandardItem


    class UiLoader:
        def loadUi(self, ui_file, base_instance):
            loader = QtUiTools.QUiLoader()
            ui = loader.load(ui_file, base_instance)
            for attr in dir(ui):
                if not attr.startswith('__') and not hasattr(base_instance, attr):
                    setattr(base_instance, attr, getattr(ui, attr))
            return ui


    uic = UiLoader()

# --- CHANNELS FOR NON-BLOCKING BACKGROUND NETWORK TRANSFER ---
network_queue = queue.Queue()


class NetworkWorker(QThread):
    network_status_signal = pyqtSignal(str)

    def __init__(self, server_ip, server_port=5000):
        super().__init__()
        self.server_url = f"http://{server_ip}:{server_port}/upload"
        self.running = True

    def run(self):
        print(f"[NETWORK] Background worker targeting: {self.server_url}", flush=True)
        while self.running:
            try:
                file_path = network_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if not os.path.exists(file_path):
                network_queue.task_done()
                continue

            print(f"[NETWORK] Processing item from queue: {os.path.basename(file_path)}", flush=True)
            self.network_status_signal.emit(f"Uploading {os.path.basename(file_path)}...")

            try:
                with open(file_path, 'rb') as img_file:
                    files = {'image': (os.path.basename(file_path), img_file, 'image/jpeg')}
                    response = requests.post(self.server_url, files=files, timeout=5.0)

                    if response.status_code == 200:
                        print(f"[NETWORK SUCCESS] Transferred cleanly to server: {response.text}", flush=True)
                        self.network_status_signal.emit("Server Sync: OK")
                    else:
                        print(f"[NETWORK WARNING] Server rejected file with status: {response.status_code}", flush=True)
                        self.network_status_signal.emit(f"Server Error Code: {response.status_code}")
            except Exception as net_error:
                print(f"[NETWORK CRITICAL ERROR] Connection failed: {net_error}", flush=True)
                self.network_status_signal.emit("Server Connection Offline")

            network_queue.task_done()


# --- SETUP SELECTION DIALOG WINDOW ---
class CameraSetupDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Deployment Setup")
        self.setFixedSize(360, 280)

        layout = QtWidgets.QVBoxLayout(self)

        info_label = QtWidgets.QLabel("<b>Select the CORRECT / ALLOWED outfit criteria:</b>")
        layout.addWidget(info_label)

        # Top selection group
        layout.addWidget(QtWidgets.QLabel("Chest Requirement:"))
        self.top_combo = QtWidgets.QComboBox()
        self.top_combo.addItems(["shirt", "sleeveless shirt"])
        layout.addWidget(self.top_combo)

        # Legs selection group
        layout.addWidget(QtWidgets.QLabel("Legs Requirement:"))
        self.bottom_combo = QtWidgets.QComboBox()
        self.bottom_combo.addItems(["long pants", "shorts"])
        layout.addWidget(self.bottom_combo)

        # Feet selection group
        layout.addWidget(QtWidgets.QLabel("Footwear Requirement:"))
        self.feet_combo = QtWidgets.QComboBox()
        self.feet_combo.addItems(["covered shoes", "uncovered shoes"])
        layout.addWidget(self.feet_combo)

        layout.addSpacing(15)

        # Confirm actions
        self.btn_confirm = QtWidgets.QPushButton("🚀 Initialize Active Monitoring")
        self.btn_confirm.clicked.connect(self.accept)
        layout.addWidget(self.btn_confirm)

    def get_target_config(self):
        return {
            "top": self.top_combo.currentText(),
            "bottom": self.bottom_combo.currentText(),
            "footwear": self.feet_combo.currentText()
        }


# --- BACKGROUND PROCESSING THREAD (STREAMLINED PICAMERA2) ---
class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    status_signal = pyqtSignal(str, tuple)
    file_saved_signal = pyqtSignal()

    def __init__(self, config_targets):
        super().__init__()
        self.save_dir = '/home/defaultpi/Desktop/Video_capture'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        # Config rules selected by user from the setup panel window
        self.allowed_top = config_targets["top"]
        self.allowed_bottom = config_targets["bottom"]
        self.allowed_footwear = config_targets["footwear"]

        model_path = '/home/defaultpi/sharedfolder/Faridstuff/best.onnx'
        print(f"[INIT] Loading YOLO ONNX Model into Pi CPU...", flush=True)
        self.model = YOLO(model_path)

        self.frame_count = 0
        self.inference_skip = 6
        self.cached_detections = []
        self.last_saved_hist = None
        self.similarity_threshold = 0.75

    def run(self):
        print("[CAMERA] Initializing Picamera2 Pipeline...", flush=True)
        try:
            from picamera2 import Picamera2
        except ImportError:
            print("[CRITICAL ERROR] 'picamera2' library not found inside this environment!", flush=True)
            self.status_signal.emit("Picamera2 Missing", (255, 0, 0))
            return

        try:
            picam = Picamera2()
            camera_config = picam.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
            picam.configure(camera_config)
            picam.start()
            print("[CAMERA] Picamera2 Engine Started Successfully.", flush=True)
        except Exception as cam_err:
            print(f"[CAMERA ENGINE ERROR] Failed to bind hardware: {cam_err}", flush=True)
            self.status_signal.emit("Hardware Bind Failed", (255, 0, 0))
            return

        while True:
            frame = picam.capture_array()
            if frame is None:
                QThread.msleep(2)
                continue

            self.frame_count += 1
            h, w, channel = frame.shape

            center_width_pct = 0.40
            remaining_space = 1.0 - center_width_pct
            x_start = int(w * (remaining_space / 2))
            x_end = int(w * (1.0 - (remaining_space / 2)))

            if self.frame_count % self.inference_skip == 0:
                roi_frame = frame[:, x_start:x_end]
                try:
                    results = self.model(roi_frame, conf=0.50, device='cpu', verbose=False)
                except Exception as model_err:
                    print(f"[YOLO ENGINE ERROR] ONNX Inference failed: {model_err}", flush=True)
                    continue

                new_raw_items = []
                if len(results) > 0:
                    boxes = results[0].boxes
                    for box in boxes:
                        cls_name = self.model.names[int(box.cls[0])]
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()

                        y_center = (xyxy[1] + xyxy[3]) / 2
                        cls_name_lower = cls_name.lower().strip()

                        zone = None
                        if cls_name_lower in ['shirt', 'sleeveless shirt']:
                            zone = 'top'
                        elif cls_name_lower in ['long pants', 'shorts']:
                            zone = 'bottom'
                        elif cls_name_lower in ['covered shoes', 'uncovered shoes']:
                            zone = 'footwear'

                        if zone:
                            new_raw_items.append({
                                'name': cls_name_lower, 'zone': zone, 'conf': conf,
                                'x1': int(xyxy[0] + x_start), 'y1': int(xyxy[1]),
                                'x2': int(xyxy[2] + x_start), 'y2': int(xyxy[3]), 'y_center': y_center
                            })

                self.cached_detections = new_raw_items

                people_clusters = []
                for item in self.cached_detections:
                    assigned_to_cluster = False
                    item_x_center = (item['x1'] + item['x2']) / 2

                    for cluster in people_clusters:
                        if (cluster['min_x'] - 50) <= item_x_center <= (cluster['max_x'] + 50):
                            cluster['items'].append(item)
                            cluster['min_x'] = min(cluster['min_x'], item['x1'])
                            cluster['max_x'] = max(cluster['max_x'], item['x2'])
                            assigned_to_cluster = True
                            break

                    if not assigned_to_cluster:
                        people_clusters.append({'min_x': item['x1'], 'max_x': item['x2'], 'items': [item]})

                frame_has_valid_save = False
                frame_status_messages = []

                for index, person in enumerate(people_clusters):
                    detections_sorted = sorted(person['items'], key=lambda x: x['y_center'])
                    detected_zones = [item['zone'] for item in detections_sorted]
                    detected_names = [item['name'] for item in detections_sorted]

                    has_top = 'top' in detected_zones
                    has_bottom = 'bottom' in detected_zones
                    has_footwear = 'footwear' in detected_zones

                    if not (has_top and has_bottom and has_footwear):
                        missing = []
                        if not has_top: missing.append("Top")
                        if not has_bottom: missing.append("Bottom")
                        if not has_footwear: missing.append("Footwear")
                        frame_status_messages.append(f"Person {index + 1}: Missing {', '.join(missing)}")
                    else:
                        idx_top = detected_zones.index('top')
                        idx_bottom = detected_zones.index('bottom')
                        idx_footwear = detected_zones.index('footwear')

                        if idx_top < idx_bottom < idx_footwear:
                            current_top = detected_names[idx_top]
                            current_bottom = detected_names[idx_bottom]
                            current_footwear = detected_names[idx_footwear]

                            # EVALUATION: True means perfect match, False means violation
                            is_outfit_valid = (current_top == self.allowed_top and
                                               current_bottom == self.allowed_bottom and
                                               current_footwear == self.allowed_footwear)

                            person_crop = frame[:, max(0, person['min_x'] - 15): min(w, person['max_x'] + 15)]
                            hsv_roi = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
                            current_hist = cv2.calcHist([hsv_roi], [0, 1], None, [50, 60], [0, 180, 0, 256])
                            cv2.normalize(current_hist, current_hist, 0, 1, cv2.NORM_MINMAX)

                            is_duplicate = False
                            if self.last_saved_hist is not None:
                                similarity_score = cv2.compareHist(current_hist, self.last_saved_hist,
                                                                   cv2.HISTCMP_CORREL)
                                if similarity_score >= self.similarity_threshold:
                                    is_duplicate = True

                            # --- UPDATED MECHANICS: ONLY PROCESS IF OUTSIDE EXPECTED UNIFORM ---
                            if not is_outfit_valid:
                                if not is_duplicate:
                                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                                    all_y1 = [item['y1'] for item in detections_sorted]
                                    all_y2 = [item['y2'] for item in detections_sorted]

                                    metadata_payload = {
                                        "timestamp": timestamp,
                                        "top_item": current_top,
                                        "bottom_item": current_bottom,
                                        "footwear_item": current_footwear,
                                        "is_valid_outfit": False,
                                        "person_bounding_box": [int(person['min_x']), int(min(all_y1)) if all_y1 else 0,
                                                                int(person['max_x']),
                                                                int(max(all_y2)) if all_y2 else 0],
                                        "shirt_bounding_box": next(
                                            ([it['x1'], it['y1'], it['x2'], it['y2']] for it in detections_sorted if
                                             it['zone'] == 'top'), [0, 0, 0, 0])
                                    }

                                    filename = f"{self.save_dir}/outfit_{timestamp}_{current_top}_{current_bottom}_{current_footwear}_WRONG.jpg"
                                    frame_status_messages.append(f"Person {index + 1} WRONG CLOTHES SAVED!")

                                    # Save full frame locally and hand over tracking details to server queue
                                    if save_image_with_metadata(filename, frame, metadata_payload):
                                        network_queue.put(filename)

                                    self.last_saved_hist = current_hist
                                    frame_has_valid_save = True
                                    self.file_saved_signal.emit()
                                else:
                                    frame_status_messages.append(f"Person {index + 1}: Wrong Clothes (Duplicate)")
                            else:
                                # Safe pass! Do not record, log or queue anything to server
                                frame_status_messages.append(f"Person {index + 1}: Proper attire detected.")
                        else:
                            frame_status_messages.append(f"Person {index + 1}: Vertical sequencing error.")

                if frame_status_messages:
                    self.status_signal.emit(" | ".join(frame_status_messages),
                                            (0, 255, 0) if frame_has_valid_save else (255, 255, 255))
                else:
                    self.status_signal.emit("Scanning...", (255, 255, 255))

            for det in self.cached_detections:
                cv2.rectangle(frame, (det['x1'], det['y1']), (det['x2'], det['y2']), (255, 255, 0), 2)
                cv2.putText(frame, f"{det['name']}", (det['x1'], max(det['y1'] - 7, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.line(frame, (x_start, 0), (x_start, h), (0, 165, 255), 1)
            cv2.line(frame, (x_end, 0), (x_end, h), (0, 165, 255), 1)

            bytes_per_line = channel * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
            self.change_pixmap_signal.emit(qt_image)
            QThread.msleep(1)


# --- MAIN INTERFACE APPLICATION CLASS ---
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config_targets):
        super().__init__()
        self.setWindowTitle("YOLO Smart Camera Monitor")
        self.resize(800, 600)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        self.Camera_screen = QtWidgets.QLabel("Starting Camera Feed...")
        self.Camera_screen.setStyleSheet("background-color: black; color: white;")
        try:
            self.Camera_screen.setAlignment(Qt.AlignmentFlag.AlignCenter)
        except AttributeError:
            self.Camera_screen.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.Camera_screen, stretch=3)

        right_panel = QtWidgets.QVBoxLayout()
        self.updateButton = QtWidgets.QPushButton("🔄 Refresh Files")
        self.pushButton_2 = QtWidgets.QPushButton("❌ Delete Selected")
        self.listView = QtWidgets.QListView()

        right_panel.addWidget(self.updateButton)
        right_panel.addWidget(self.listView)
        right_panel.addWidget(self.pushButton_2)
        main_layout.addLayout(right_panel, stretch=1)

        self.capture_dir = '/home/defaultpi/Desktop/Video_capture'
        self.current_selected_path = None

        self.list_model = QStandardItemModel()
        self.listView.setModel(self.list_model)
        self.listView.clicked.connect(self.display_selected_image)

        # Pass selections dynamically into parsing thread loop
        self.thread = CameraThread(config_targets)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.status_signal.connect(self.update_status)
        self.thread.file_saved_signal.connect(self.check_files)

        self.network_thread = NetworkWorker(server_ip="192.168.50.202", server_port=5000)
        self.network_thread.network_status_signal.connect(self.update_network_status)
        self.network_thread.start()

        self.updateButton.clicked.connect(self.check_files)
        self.pushButton_2.clicked.connect(self.delete_selected_image)

        self.thread.start()
        self.check_files()

    def update_image(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image)
        try:
            aspect = Qt.AspectRatioMode.KeepAspectRatio;
            transform = Qt.TransformationMode.FastTransformation
        except AttributeError:
            aspect = Qt.KeepAspectRatio;
            transform = Qt.FastTransformation
        scaled_pixmap = pixmap.scaled(self.Camera_screen.size(), aspect, transform)
        self.Camera_screen.setPixmap(scaled_pixmap)

    def update_status(self, text, color_rgb):
        print(f"[LIVE STATUS] {text}", flush=True)

    def update_network_status(self, network_text):
        print(f"[SERVER] {network_text}", flush=True)

    def check_files(self):
        folder_path = Path(self.capture_dir)
        self.list_model.clear()
        self.current_selected_path = None

        if folder_path.exists():
            all_files = [f for f in os.listdir(self.capture_dir) if
                         f.lower().startswith('outfit_') and f.lower().endswith(('.jpg', '.jpeg'))]
            all_files.sort(reverse=True)

            for filename in all_files:
                try:
                    parts = filename.split('_')
                    date_str = parts[1];
                    time_str = parts[2]
                    clothing_info = " + ".join(parts[3:]).replace('.jpg', '').replace('.jpeg', '')
                    display_text = f"📅 {date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]} | {clothing_info}"
                except IndexError:
                    display_text = filename

                item = QStandardItem(display_text)
                try:
                    role = Qt.ItemDataRole.UserRole
                except AttributeError:
                    role = Qt.UserRole
                item.setData(filename, role)
                self.list_model.appendRow(item)

    def display_selected_image(self, index):
        try:
            role = Qt.ItemDataRole.UserRole
        except AttributeError:
            role = Qt.UserRole
        raw_filename = index.data(role)
        if raw_filename:
            self.current_selected_path = os.path.join(self.capture_dir, raw_filename)

    def delete_selected_image(self):
        if hasattr(self, 'current_selected_path') and self.current_selected_path:
            if os.path.exists(self.current_selected_path):
                try:
                    os.remove(self.current_selected_path)
                    self.current_selected_path = None
                    self.check_files()
                except Exception as e:
                    print(f"[ERROR] Failed to delete file: {e}", flush=True)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # 1. Pop up configuration step first
    setup_dialog = CameraSetupDialog()
    if setup_dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
        user_config = setup_dialog.get_target_config()
        print(f"[SETUP] Monitoring initialized with allowed pattern: {user_config}", flush=True)

        # 2. Launch main GUI only if configuration is confirmed
        window = MainWindow(user_config)
        window.show()
        sys.exit(app.exec())
    else:
        print("[SETUP] Cancelled by user. Exiting application.", flush=True)
        sys.exit(0)
