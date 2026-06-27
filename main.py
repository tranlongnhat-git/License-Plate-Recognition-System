import sys
import cv2
import torch
import time
import pathlib
import warnings
import numpy as np
from collections import Counter

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog,
                             QFrame, QListWidget, QGroupBox, QSizePolicy)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import QTimer, Qt

import function.utils_rotate as utils_rotate
import function.helper as helper

pathlib.PosixPath = pathlib.WindowsPath
warnings.filterwarnings("ignore")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hệ Thống Nhận Diện Biển Số Xe Máy AI")
        self.setWindowState(Qt.WindowState.WindowMaximized)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_webcam)
        self.is_camera_on = False

        self.temp_plates = []
        self.last_sent_time = {}
        self.MAX_SCANS = 30
        self.RESEND_DELAY = 10

        self.load_models()
        self.initUI()

    def load_models(self):
        print("⏳ Đang tải mô hình, vui lòng đợi...")
        try:
            self.yolo_LP_detect = torch.hub.load('yolov5_old', 'custom', path='model/detect_plate_nano.pt',
                                                 force_reload=True, source='local')
            self.yolo_license_plate = torch.hub.load('yolov5_old', 'custom', path='model/ocr_nano.pt',
                                                     force_reload=True, source='local')
            self.yolo_license_plate.conf = 0.60
            print("✅ Mô hình đã tải xong!")
        except Exception as e:
            print(f"❌ Lỗi tải model: {e}")

    def initUI(self):
        self.setFont(QFont("Segoe UI", 10))
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        left_layout = QVBoxLayout()
        title_label = QLabel("AI LICENSE PLATE RECOGNITION SYSTEM")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #1565C0; margin-bottom: 10px;")
        left_layout.addWidget(title_label)

        btn_layout = QHBoxLayout()
        self.btn_select_img = QPushButton("📂 CHỌN ẢNH")
        self.btn_select_img.setMinimumHeight(50)
        self.btn_select_img.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_select_img.clicked.connect(self.process_image)

        self.btn_webcam = QPushButton("📹 BẬT WEBCAM")
        self.btn_webcam.setMinimumHeight(50)
        self.btn_webcam.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_webcam.clicked.connect(self.toggle_webcam)

        btn_layout.addWidget(self.btn_select_img)
        btn_layout.addWidget(self.btn_webcam)
        left_layout.addLayout(btn_layout)

        self.lbl_display = QLabel("Vui lòng chọn ảnh hoặc bật Camera")
        self.lbl_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_display.setStyleSheet(
            "background-color: #222; color: #EEE; border-radius: 8px; border: 2px solid #555;")
        self.lbl_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.lbl_display.setMinimumSize(640, 480)
        left_layout.addWidget(self.lbl_display)

        right_layout = QVBoxLayout()
        right_panel = QFrame()
        right_panel.setFixedWidth(400)
        right_panel.setStyleSheet("background-color: #F8F9FA; border-radius: 10px; border: 1px solid #DDD;")
        panel_layout = QVBoxLayout(right_panel)

        gb_result = QGroupBox("BIỂN SỐ GẦN NHẤT")
        gb_result.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        gb_result.setStyleSheet(
            "QGroupBox { border: 1px solid #CCC; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #E65100; }")
        gb_result_layout = QVBoxLayout()

        self.lbl_result = QLabel("---")
        self.lbl_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_result.setFont(QFont("Segoe UI", 36, QFont.Weight.Bold))
        self.lbl_result.setStyleSheet("color: #2E7D32; padding: 10px;")
        gb_result_layout.addWidget(self.lbl_result)
        gb_result.setLayout(gb_result_layout)
        panel_layout.addWidget(gb_result)

        gb_log = QGroupBox("NHẬT KÝ HỆ THỐNG")
        gb_log.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        gb_log.setStyleSheet(
            "QGroupBox { border: 1px solid #CCC; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #1565C0; }")
        gb_log_layout = QVBoxLayout()

        self.list_log = QListWidget()
        self.list_log.setFont(QFont("Consolas", 10))
        self.list_log.setStyleSheet("border: none; background-color: transparent;")
        gb_log_layout.addWidget(self.list_log)
        gb_log.setLayout(gb_log_layout)
        panel_layout.addWidget(gb_log)

        right_layout.addWidget(right_panel)

        main_layout.addLayout(left_layout, stretch=3)
        main_layout.addLayout(right_layout, stretch=1)

        self.apply_styles()

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #FFFFFF; }
            QPushButton {
                background-color: #FAFAFA; color: #333; 
                border: 1px solid #CCC; border-radius: 6px;
                font-family: "Segoe UI"; font-size: 14px; font-weight: bold;
            }
            QPushButton:hover { background-color: #EEE; border-color: #999; }
            QPushButton:pressed { background-color: #DDD; }
        """)
        self.btn_webcam.setStyleSheet("""
            QPushButton { background-color: #E3F2FD; color: #0D47A1; border: 1px solid #90CAF9; border-radius: 6px; font-weight: bold; font-size: 14px;}
            QPushButton:hover { background-color: #BBDEFB; }
        """)
        self.btn_select_img.setStyleSheet("""
            QPushButton { background-color: #FFF3E0; color: #E65100; border: 1px solid #FFCC80; border-radius: 6px; font-weight: bold; font-size: 14px;}
            QPushButton:hover { background-color: #FFE0B2; }
        """)

    def log_message(self, message, is_success=True):
        timestamp = time.strftime("[%H:%M:%S]")
        full_msg = f"{timestamp} {message}"
        self.list_log.addItem(full_msg)
        self.list_log.scrollToBottom()
        item = self.list_log.item(self.list_log.count() - 1)
        if is_success:
            item.setForeground(Qt.GlobalColor.darkGreen)
        else:
            item.setForeground(Qt.GlobalColor.red)

    def convert_cv_qt(self, cv_img):
        try:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            rgb_image = np.ascontiguousarray(rgb_image)

            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w

            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            p = convert_to_Qt_format.scaled(self.lbl_display.width(), self.lbl_display.height(),
                                            Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation)
            return QPixmap.fromImage(p)
        except Exception as e:
            print(f"Lỗi hiển thị ảnh: {e}")
            return QPixmap()

    def semantic_correction(self, lp_text):
        if len(lp_text) < 3: return lp_text
        lp_list = list(lp_text)
        dict_char_to_num = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'J': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6'}
        dict_num_to_char = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '8': 'B', '6': 'G'}
        is_army = lp_list[0].isalpha() and lp_list[1].isalpha()
        text_check = "".join(lp_list)
        is_diplomatic = "NG" in text_check or "NN" in text_check or "QT" in text_check
        if is_army:
            for i in range(2):
                if lp_list[i] in dict_num_to_char: lp_list[i] = dict_num_to_char[lp_list[i]]
            for i in range(2, len(lp_list)):
                if lp_list[i] in dict_char_to_num: lp_list[i] = dict_char_to_num[lp_list[i]]

        elif is_diplomatic:
            for i in range(2):
                if lp_list[i] in dict_char_to_num: lp_list[i] = dict_char_to_num[lp_list[i]]

        else:
            for i in range(2):
                if i < len(lp_list) and lp_list[i] in dict_char_to_num:
                    lp_list[i] = dict_char_to_num[lp_list[i]]
            if len(lp_list) > 2:
                if lp_list[2] in dict_num_to_char:
                    lp_list[2] = dict_num_to_char[lp_list[2]]
            for i in range(4, len(lp_list)):
                if lp_list[i] in dict_char_to_num:
                    lp_list[i] = dict_char_to_num[lp_list[i]]

        return "".join(lp_list)

    def draw_plate_info(self, img, plate, lp_text, x1, y1):
        conf_val = plate[4]
        display_text = f"{lp_text} ({conf_val:.2f})"
        (text_w, text_h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img, (x1, y1 - 30), (x1 + text_w, y1), (0, 0, 0), -1)
        cv2.putText(img, display_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def process_image(self):
        if self.is_camera_on: self.toggle_webcam()

        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh xe", "", "Image files (*.jpg *.jpeg *.png)")
        if not file_path: return

        img = cv2.imread(file_path)
        if img is None: return

        plates = self.yolo_LP_detect(img, size=640)
        list_plates = plates.pandas().xyxy[0].values.tolist()

        if len(list_plates) == 0:
            self.log_message("⚠️ Không tìm thấy khung biển. Đang thử đọc toàn ảnh...", False)
            h, w, _ = img.shape
            list_plates = [[0, 0, w, h, 1.0, 0]]
        else:
            self.log_message(f"🔍 Tìm thấy {len(list_plates)} vùng biển số.", True)

        found_plate = False
        list_plates.sort(key=lambda x: x[0])

        for i, plate in enumerate(list_plates):
            flag = 0
            x1, y1, x2, y2 = int(plate[0]), int(plate[1]), int(plate[2]), int(plate[3])
            conf = plate[4]
            crop_img = img[y1:y2, x1:x2]

            if conf < 1.0:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            for cc in range(2):
                for ct in range(2):
                    lp = helper.read_plate(self.yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                    if lp != "unknown":
                        lp = self.semantic_correction(lp)
                        self.lbl_result.setText(lp)
                        self.log_message(f"Phát hiện: {lp} (Conf: {conf:.2f})")
                        self.draw_plate_info(img, plate, lp, x1, y1)
                        found_plate = True
                        flag = 1
                        break
                if flag: break

        self.lbl_display.setPixmap(self.convert_cv_qt(img))

    def toggle_webcam(self):
        if self.is_camera_on:
            self.timer.stop()
            if self.cap: self.cap.release()
            self.is_camera_on = False
            self.btn_webcam.setText("📹 BẬT WEBCAM")
            self.btn_webcam.setStyleSheet(
                "QPushButton { background-color: #E3F2FD; color: #0D47A1; border: 1px solid #90CAF9; border-radius: 6px; font-weight: bold; font-size: 14px;} QPushButton:hover { background-color: #BBDEFB; }")
            self.lbl_display.setText("Đã tắt Camera")
            self.lbl_display.setPixmap(QPixmap())
        else:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.log_message("❌ Không thể mở Camera!", False)
                return
            self.is_camera_on = True
            self.btn_webcam.setText("🛑 TẮT WEBCAM")
            self.btn_webcam.setStyleSheet(
                "background-color: #FFCDD2; color: #C62828; border: 1px solid #E57373; border-radius: 6px; font-weight: bold; font-size: 14px;")
            self.timer.start(30)

    def update_webcam(self):
        ret, frame = self.cap.read()
        if ret:
            plates = self.yolo_LP_detect(frame, size=640)
            list_plates = plates.pandas().xyxy[0].values.tolist()
            current_frame_plates = []

            for i, plate in enumerate(list_plates):
                x1, y1, x2, y2 = int(plate[0]), int(plate[1]), int(plate[2]), int(plate[3])
                crop_img = frame[y1:y2, x1:x2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                found_in_box = False
                for cc in range(2):
                    for ct in range(2):
                        lp = helper.read_plate(self.yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                        if lp != "unknown":
                            lp = self.semantic_correction(lp)
                            current_frame_plates.append(lp)
                            self.draw_plate_info(frame, plate, lp, x1, y1)
                            found_in_box = True
                            break
                    if found_in_box: break

            self.temp_plates.extend(current_frame_plates)
            if len(self.temp_plates) >= self.MAX_SCANS:
                most_common_plate, count = Counter(self.temp_plates).most_common(1)[0]
                now = time.time()
                last_time = self.last_sent_time.get(most_common_plate, 0)

                if now - last_time >= self.RESEND_DELAY:
                    self.last_sent_time[most_common_plate] = now
                    self.lbl_result.setText(most_common_plate)
                    self.log_message(f"Phát hiện: {most_common_plate}")

                self.temp_plates.clear()

            self.lbl_display.setPixmap(self.convert_cv_qt(frame))
        else:
            self.toggle_webcam()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())