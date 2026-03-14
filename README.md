# 🔥 FireTracking - Hệ thống Giám sát & Phát hiện Cháy sớm

Dự án này tập trung vào việc xây dựng hệ thống phát hiện cháy sử dụng Deep Learning, từ khâu thu thập dữ liệu, gán nhãn tự động đến huấn luyện và triển khai ứng dụng giám sát.

## 🚀 Tiến trình Tuần này (Cập nhật 15/03/2026)

Tuần này, đã đạt được những bước tiến quan trọng trong việc nâng cấp hệ thống lên phiên bản **v2**, bao gồm:

### 1. Xây dựng Pipeline Huấn luyện Toàn diện (`v2/`)
- **Đa kiến trúc mô hình:** Hỗ trợ huấn luyện nhiều loại mô hình khác nhau để so sánh hiệu năng:
    - Custom CNN (Kiến trúc tự định nghĩa)
    - ResNet-50
    - MobileNet-V2 (Tối ưu cho thiết bị di động)
    - VGG-16
    - EfficientNet-B0
- **Tự động hóa huấn luyện:** Script `v2/train.py` cho phép dễ dàng cấu hình model và số lượng epochs.
- **Tiền xử lý dữ liệu:** Module `v2/data_preparation.py` và `v2/dataset.py` giúp chuẩn hóa dữ liệu đầu vào.

### 2. Phát triển Ứng dụng Giám sát (Web App)
- Triển khai giao diện người dùng bằng **Streamlit** (`v2/app.py`).
- Cho phép người dùng tải ảnh lên và nhận kết quả dự đoán (Cháy hoặc An toàn) với độ tin cậy cụ thể.
- Hệ thống tự động tải trọng số mô hình tốt nhất từ thư mục `checkpoints`.

### 3. Cải tiến Công cụ Gán nhãn (Labeling)
- Nâng cấp script gán nhãn `fire_labeler_v2.py` để xử lý tập dữ liệu mới hiệu quả hơn.
- Sử dụng **YOLO-World** để tăng cường độ chính xác khi phát hiện vùng cháy trong ảnh.

---

## 📁 Cấu trúc Thư mục Chính

- `v2/app.py`: Ứng dụng Streamlit hiển thị kết quả.
- `v2/train.py`: Script huấn luyện mô hình.
- `v2/models.py`: Định nghĩa các kiến trúc Deep Learning.
- `v2/dataset/`: Thư mục lưu trữ dữ liệu đã phân loại (Fire/Non-Fire).
- `checkpoints/`: Lưu trữ các file trọng số (.pth) tốt nhất sau khi huấn luyện.

## 🛠 Hướng dẫn Chạy nhanh

### Huấn luyện mô hình:
```bash
python v2/train.py --model mobilenet_v2 --epochs 20
```

### Chạy ứng dụng Web:
```bash
streamlit run v2/app.py
```

---
*Ghi chú: Đã cấu hình `.gitignore` để loại bỏ các thư mục dữ liệu nặng (`fire_dataset/`, `v2/dataset/`) và các file trọng số lớn (`*.pt`) khi đẩy lên hệ thống.*
