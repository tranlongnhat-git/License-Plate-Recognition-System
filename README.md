# License Plate Recognition System 🚗

Dự án Hệ thống nhận diện biển số xe ứng dụng mô hình YOLOv5. Hệ thống bao gồm hai giai đoạn chính: phát hiện vị trí biển số (License Plate Detection) và nhận dạng các ký tự trên biển số (OCR - Optical Character Recognition).

## 📂 Cấu trúc dự án
* `function/`: Chứa các hàm hỗ trợ xử lý ảnh, xoay ảnh và các utility khác.
* `model/`: Chứa các mô hình pre-trained (file `.pt`).
* `result/`: Thư mục lưu trữ kết quả nhận diện đầu ra.
* `yolov5_old/`: Chứa mã nguồn phiên bản YOLOv5 cũ được tuỳ chỉnh/sử dụng riêng cho dự án này.
* `main.py`: Script chạy nhận diện trên ảnh tĩnh.
* `webcam1.py` : Script chạy nhận diện trực tiếp qua luồng Webcam.
* `LP_Detect_nano_MAIN.ipynb`: Source code huấn luyện mô hình phát hiện biển số (YOLOv5 Nano).
* `OCR_MAIN.ipynb`: Source code huấn luyện mô hình nhận diện ký tự (OCR).

## 🗄️ Dữ liệu huấn luyện & Mã nguồn liên quan
Dự án sử dụng các bộ dữ liệu và mã nguồn sau:

1. **License Plate Detection Dataset** (Dữ liệu huấn luyện vị trí biển số): 
   [Tải xuống tại đây](https://drive.google.com/file/d/1xchPXf7a1r466ngow_W_9bittRqQEf_T/view?usp=sharing)
2. **Character Detection Dataset** (Dữ liệu huấn luyện nhận dạng ký tự): 
   [Tải xuống tại đây](https://drive.google.com/file/d/1bPux9J0e1mz-_Jssx4XX1-wPGamaS8mI/view?usp=sharing)
3. **Mã nguồn YOLOv5 (Phiên bản cũ)** (Được sử dụng trong dự án):
   [Tải xuống tại đây](https://drive.google.com/file/d/1g1u7M4NmWDsMGOppHocgBKjbwtDA-uIu/view?usp=sharing)

## 🚀 Hướng dẫn sử dụng
1. Cài đặt các thư viện cần thiết:
   ```bash

   pip install -r requirements.txt
