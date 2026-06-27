import pathlib
from pathlib import Path
import sys
import os
import cv2
import torch
import time
from collections import Counter
import warnings

# --- 1. SỬA LỖI ĐƯỜNG DẪN WINDOWS (BẮT BUỘC) ---
pathlib.PosixPath = pathlib.WindowsPath

# --- 2. CẤU HÌNH ĐƯỜNG DẪN ---
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
YOLO_DIR = os.path.join(str(ROOT), 'yolov5')  # Trỏ vào thư mục yolov5

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if YOLO_DIR not in sys.path:
    sys.path.append(YOLO_DIR)

# Import các hàm phụ trợ
try:
    import function.utils_rotate as utils_rotate
    import function.helper as helper
except ImportError:
    print("⚠️ Cảnh báo: Không tìm thấy thư mục 'function'.")

warnings.filterwarnings("ignore")

# ============================
# LOAD MODELS (OFFLINE)
# ============================
print(f"⏳ Đang tải mô hình từ: {YOLO_DIR} ...")
try:
    # 1. Detect Model
    yolo_LP_detect = torch.hub.load(YOLO_DIR, 'custom', path='model/Plate_YOLOv5n.pt', source='local')
    yolo_license_plate = torch.hub.load(YOLO_DIR, 'custom', path='model/OCR_YOLOv5n_FixName.pt', source='local')
    yolo_license_plate.conf = 0.60

    print("✅ Tải thành công!")
except Exception as e:
    sys.exit(f"❌ Lỗi tải model: {e}")

# ============================
# CẤU HÌNH LOGIC (FULL TÍNH NĂNG)
# ============================
MAX_SCANS = 30  # Gom đủ 30 kết quả mới chốt
RESEND_DELAY = 10  # Chặn trùng lặp trong 10 giây
DISPLAY_SIZE = (1280, 720)  # Kích thước hiển thị

# Biến trạng thái
temp_plates = []  # Kho chứa tạm các biển số đọc được
last_sent_time = {}  # Lưu thời gian gửi gần nhất để chặn spam
paused = False  # Trạng thái tạm dừng
last_confirmed_plate = "Waiting..."  # Dòng chữ hiện trên màn hình
message_timer = 0  # Thời gian hiển thị thông báo
prev_frame_time = 0

# Tạo/Xóa file output lúc bắt đầu
with open("output.txt", "w") as f:
    pass

# Mở Camera
vid = cv2.VideoCapture(0)
print(f"🚀 Bắt đầu... (Màn hình {DISPLAY_SIZE})")
print("⌨️  Phím 'p': Tạm dừng | Phím 'q': Thoát")

while True:
    # --- XỬ LÝ PHÍM BẤM ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused
        status = "⏸ DA TAM DUNG" if paused else "▶ TIEP TUC"
        print(status)
        last_confirmed_plate = status
        message_timer = time.time()

    # Nếu không Pause thì mới đọc và xử lý
    if not paused:
        ret, frame = vid.read()
        if not ret:
            print("❌ Lỗi Camera")
            break

        # --- 1. NHẬN DIỆN & ĐỌC (YOLO) ---
        try:
            plates = yolo_LP_detect(frame, size=640)
            list_plates = plates.pandas().xyxy[0].values.tolist()

            # Danh sách biển số trong frame hiện tại
            current_frame_results = []

            for plate in list_plates:
                x1, y1, x2, y2 = int(plate[0]), int(plate[1]), int(plate[2]), int(plate[3])
                crop_img = frame[y1:y2, x1:x2]

                # Vẽ khung đỏ
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Logic xoay & đọc OCR
                found_text = False
                for cc in range(2):
                    for ct in range(2):
                        lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                        if lp != "unknown":
                            current_frame_results.append(lp)
                            # Hiện chữ nhỏ ngay trên biển số
                            cv2.putText(frame, lp, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            found_text = True
                            break
                    if found_text: break

            # Gom kết quả vào kho tạm
            temp_plates.extend(current_frame_results)

            # --- 2. LOGIC GOM 30 ẢNH & GHI FILE ---
            if len(temp_plates) >= MAX_SCANS:
                # Tìm biển số xuất hiện nhiều nhất trong 30 mẫu
                most_common_plate, count = Counter(temp_plates).most_common(1)[0]

                now = time.time()
                last_time = last_sent_time.get(most_common_plate, 0)

                # Kiểm tra Delay (chống spam)
                if now - last_time >= RESEND_DELAY:
                    last_sent_time[most_common_plate] = now

                    # Ghi file
                    print(f"✅ CHỐT KẾT QUẢ: {most_common_plate}")
                    with open("output.txt", "a") as f:
                        f.write(most_common_plate + "\n")

                    # Cập nhật thông báo
                    last_confirmed_plate = f"Da luu: {most_common_plate}"
                    message_timer = time.time()
                else:
                    # Bị chặn do trùng lặp
                    remain = int(RESEND_DELAY - (now - last_time))
                    print(f"⏳ Bỏ qua {most_common_plate} (Chờ {remain}s)")
                    last_confirmed_plate = f"Da ton tai: {most_common_plate}"
                    message_timer = time.time()

                # Reset kho tạm sau khi xử lý xong
                temp_plates.clear()

        except Exception:
            pass  # Bỏ qua lỗi nhỏ khi không detect được

        # --- 3. VẼ GIAO DIỆN (UI) ---
        # Tính FPS
        new_frame_time = time.time()
        fps = int(1 / (new_frame_time - prev_frame_time)) if prev_frame_time != 0 else 0
        prev_frame_time = new_frame_time

        # Resize màn hình hiển thị cho đẹp
        display_frame = cv2.resize(frame, DISPLAY_SIZE)

        # Hiện FPS góc trái
        cv2.putText(display_frame, f'FPS: {fps}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hiện thông báo nền đen góc phải (trong 3 giây)
        if time.time() - message_timer < 3:
            # Vẽ nền đen
            cv2.rectangle(display_frame, (DISPLAY_SIZE[0] - 550, 0), (DISPLAY_SIZE[0], 80), (0, 0, 0), -1)
            # Viết chữ thông báo
            cv2.putText(display_frame, last_confirmed_plate, (DISPLAY_SIZE[0] - 530, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        cv2.imshow('License Plate Recognition Pro', display_frame)

vid.release()
cv2.destroyAllWindows()