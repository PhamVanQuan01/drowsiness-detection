# Drowsiness Detection using Eye State Classification + EAR

## 1. Mô tả
Ứng dụng phát hiện dấu hiệu buồn ngủ của tài xế bằng cách kết hợp:
- Mô hình phân loại mắt mở / mắt nhắm
- Chỉ số EAR (Eye Aspect Ratio)
- Webcam laptop để chạy thời gian thực
- Cảnh báo âm thanh khi mắt nhắm quá lâu

## 2. Cấu trúc thư mục
```text
22T1020347_PYTHON_EYE/
└── drowsiness-detection/
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── train_colab.ipynb
    ├── models/
    │   ├── eye_state_model_final.keras
    │   └── label_map.json
    ├── data/
    │   ├── raw/
    │   │   ├── train/
    │   │   ├── val/
    │   │   └── test/
    │   └── processed/
    ├── assets/
    │   └── alarm.wav
    └── src/
        ├── preprocessing.py
        ├── inference.py
        ├── ear.py
        ├── face_mesh_utils.py
        └── alarm.py
```

## 3. Cài đặt
```bash
pip install -r requirements.txt
```

## 4. Chạy chương trình
```bash
python app.py
```

## 5. Phím điều khiển
- `q`: thoát chương trình

## 6. Logic cảnh báo
Hệ thống kết hợp 2 tín hiệu:
- Model dự đoán ảnh mắt là `Open_Eyes` hoặc `Closed_Eyes`
- EAR thấp hơn ngưỡng

Nếu trạng thái nhắm mắt kéo dài liên tục nhiều frame, hệ thống sẽ:
- Hiển thị `DROWSY ALERT`
- Phát `alarm.wav`

## 7. Ghi chú
- Nếu chỉ demo webcam thì không cần dataset local.
- Chỉ cần giữ đúng:
  - `models/eye_state_model_final.keras`
  - `models/label_map.json`
  - `assets/alarm.wav`
- Nếu webcam không mở được, thử đổi camera index trong `app.py` từ `0` sang `1`.
