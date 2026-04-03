# Ứng dụng phát hiện tài xế ngủ gật với Streamlit

## 1. Giới thiệu đề tài

**Tên đề tài:** Phát hiện trạng thái buồn ngủ của tài xế bằng mô hình MobileNetV2 kết hợp chỉ số khía cạnh mắt nhằm cảnh báo kịp thời
**Sinh viên:** Phạm Văn Quân 
**MSV:** 22T1020347  

Đây là ứng dụng Machine Learning hỗ trợ phát hiện dấu hiệu buồn ngủ của tài xế thông qua video. Hệ thống sử dụng hai tín hiệu chính:

- **EAR (Eye Aspect Ratio)** để đo mức độ đóng mở mắt từ landmark khuôn mặt.
- **Mô hình MobileNetV2** để phân loại trạng thái mắt từ ảnh crop vùng mắt.

Sau đó, hệ thống kết hợp hai tín hiệu này để đưa ra cảnh báo khi phát hiện tài xế có dấu hiệu ngủ gật.

### Giá trị thực tiễn

Ứng dụng có ý nghĩa trong các bài toán:
- hỗ trợ an toàn giao thông,
- giám sát trạng thái tỉnh táo của tài xế,
- làm nền tảng cho các hệ thống cảnh báo trên xe thông minh.

---

## 2. Chức năng chính của ứng dụng

Ứng dụng được xây dựng bằng **Streamlit** và gồm **3 trang chính** theo yêu cầu của bài tập:

### Trang 1: Giới thiệu & Khám phá dữ liệu (EDA)
Trang này giúp người dùng:
- hiểu bài toán phát hiện ngủ gật,
- xem một phần dữ liệu đầu vào,
- quan sát các biểu đồ phân tích dữ liệu,
- đọc nhận xét tổng quan về tập dữ liệu.

### Trang 2: Triển khai mô hình
Trang này cho phép người dùng:
- tải video lên hệ thống,
- cấu hình tham số xử lý,
- chạy mô hình phát hiện ngủ gật,
- xem kết quả dự đoán theo từng frame,
- nhận cảnh báo trực quan và âm thanh,
- tải video đã xử lý.

### Trang 3: Đánh giá & Hiệu năng
Trang này trình bày:
- các chỉ số đánh giá mô hình,
- ma trận nhầm lẫn,
- đường cong huấn luyện,
- phân tích lỗi và hướng cải thiện.

---

## 3. Kiến trúc hệ thống

Quy trình xử lý chính của ứng dụng:

1. Người dùng tải video lên.
2. Hệ thống đọc video theo từng frame.
3. MediaPipe Face Mesh phát hiện khuôn mặt và landmark mắt.
4. Tính **EAR** từ các điểm landmark mắt.
5. Crop vùng mắt trái và mắt phải.
6. Dùng mô hình **MobileNetV2** để dự đoán mắt mở / nhắm.
7. Kết hợp **EAR + xác suất từ mô hình** để tạo điểm cảnh báo.
8. Nếu dấu hiệu buồn ngủ kéo dài đủ lâu, hệ thống bật cảnh báo.
9. Kết quả được hiển thị trực tiếp trên giao diện và có thể xuất thành video kết quả.

---

## 4. Cấu trúc thư mục project

```bash
project/
│
├── app.py
├── README.md
├── requirements.txt
├── packages.txt
│
├── assets/
│   └── alarm.wav
│
├── models/
│   ├── eye_state_model_final.keras
│   └── label_map.json
│
├── data/
│   ├── sample_metadata.csv
│   ├── evaluation_metrics.json
│   ├── confusion_matrix.csv
│   └── training_history.csv
│
├── outputs/
│
└── src/
    ├── alarm.py
    ├── ear.py
    ├── face_mesh_utils.py
    ├── inference.py
    └── preprocessing.py
```

---

## 5. Các tệp dữ liệu cần chuẩn bị

### Trong thư mục `models/`
Bạn cần tự thêm mô hình đã huấn luyện:

- `eye_state_model_final.keras`
- `label_map.json`

Ví dụ `label_map.json`:

```json
{
  "0": "Closed_Eyes",
  "1": "Open_Eyes"
}
```

### Trong thư mục `data/`
Để hiển thị đầy đủ Trang 1 và Trang 3, bạn nên chuẩn bị:

#### `sample_metadata.csv`
Ví dụ các cột:
- `filename`
- `label`
- `split`
- `width`
- `height`

#### `evaluation_metrics.json`
Ví dụ:

```json
{
  "accuracy": 0.94,
  "precision": 0.93,
  "recall": 0.95,
  "f1_score": 0.94
}
```

#### `confusion_matrix.csv`
Ví dụ:

```csv
,Pred_Open,Pred_Closed
True_Open,120,8
True_Closed,6,110
```

#### `training_history.csv`
Ví dụ:

```csv
epoch,train_loss,val_loss,train_accuracy,val_accuracy
1,0.52,0.48,0.78,0.80
2,0.39,0.35,0.85,0.87
3,0.31,0.29,0.89,0.90
```

---

## 6. Cài đặt môi trường

### Bước 1: Tạo môi trường ảo

```bash
python -m venv venv
```

### Bước 2: Kích hoạt môi trường ảo

**Windows:**

```bash
venv\Scripts\activate
```

**MacOS / Linux:**

```bash
source venv/bin/activate
```

### Bước 3: Cài thư viện

```bash
pip install -r requirements.txt
```

---

## 7. Chạy ứng dụng

Dùng lệnh sau để chạy app:

```bash
streamlit run app.py
```

Sau đó mở trình duyệt tại địa chỉ:

```bash
http://localhost:8501
```

---

## 8. Hướng dẫn sử dụng

### Trang 1: Giới thiệu & EDA
- xem thông tin đề tài,
- xem dữ liệu mẫu,
- xem biểu đồ phân tích,
- đọc nhận xét về tập dữ liệu.

### Trang 2: Triển khai mô hình
1. Tải video lên bằng `Upload video`.
2. Điều chỉnh các tham số ở sidebar nếu cần.
3. Nhấn **Bắt đầu xử lý**.
4. Quan sát:
   - video đang được xử lý,
   - trạng thái `AWAKE / SUSPICIOUS / DROWSY ALERT`,
   - EAR, CNN score, fused score,
   - nhật ký sự kiện,
   - video kết quả để tải xuống.

### Trang 3: Đánh giá & Hiệu năng
- xem Accuracy, Precision, Recall, F1-score,
- xem confusion matrix,
- xem biểu đồ loss / accuracy,
- đọc phần phân tích sai số và hướng cải thiện.

---

## 9. Một số tham số quan trọng trong ứng dụng

- **display_width**: kích thước khung hình xử lý.
- **frame_skip**: bỏ qua bớt frame để tăng tốc.
- **alpha**: trọng số của CNN trong phép fusion với EAR.
- **ear_open / ear_closed**: ngưỡng quy đổi EAR thành điểm nhắm mắt.
- **model_closed_threshold**: ngưỡng xác suất mắt nhắm từ mô hình.
- **smoothing_window**: số frame dùng để làm mượt điểm cảnh báo.
- **drowsy_seconds_threshold**: số giây nghi ngủ gật để bật cảnh báo.
- **recovery_seconds_threshold**: số giây hồi phục để tắt cảnh báo.

---

## 10. Đánh giá ưu điểm và hạn chế

### Ưu điểm
- Kết hợp cả **đặc trưng hình học (EAR)** và **mô hình học sâu (MobileNetV2)**.
- Có giao diện web thân thiện, dễ sử dụng.
- Có thể chạy trực tiếp với video tải lên.
- Có khả năng xuất video kết quả.
- Có giao diện đánh giá mô hình tương đối đầy đủ.

### Hạn chế
- Tốc độ xử lý có thể chậm với video dài hoặc độ phân giải cao.
- Kết quả có thể bị ảnh hưởng bởi:
  - ánh sáng yếu,
  - mặt quay lệch,
  - kính che mắt,
  - chất lượng video thấp,
  - trường hợp landmark không ổn định.
- Cảnh báo âm thanh trên trình duyệt có thể bị ảnh hưởng bởi chính sách autoplay của browser.

---

## 11. Hướng cải thiện

Trong tương lai có thể cải tiến theo các hướng:
- bổ sung phát hiện ngáp,
- thêm head pose estimation,
- dùng mô hình temporal như LSTM/TCN,
- tối ưu tốc độ xử lý,
- tăng dữ liệu huấn luyện đa dạng hơn,
- tinh chỉnh ngưỡng EAR theo từng người dùng.

---

## 12. Yêu cầu triển khai

Ứng dụng có thể triển khai trên:
- **Streamlit Community Cloud**
- **Hugging Face Spaces**

Khi deploy, cần bảo đảm:
- có `requirements.txt`,
- có thư mục `models/`,
- có file `assets/alarm.wav`,
- có các file dữ liệu trong `data/` nếu muốn hiển thị đầy đủ EDA và Evaluation.

---

## 13. Ghi chú

- Nếu chưa thêm mô hình vào thư mục `models/`, Trang 2 sẽ không thể suy luận.
- Nếu chưa thêm dữ liệu trong thư mục `data/`, Trang 1 và Trang 3 vẫn mở được nhưng sẽ thiếu biểu đồ và chỉ số.
- Để ứng dụng chạy mượt hơn, có thể giảm `display_width`, tăng `frame_skip`, hoặc tắt bớt các tùy chọn vẽ landmark.

---

## 14. Thông tin liên hệ

**Sinh viên thực hiện:** Điền họ tên của bạn  
**MSSV:** Điền MSSV của bạn  
**Lớp:** Điền lớp của bạn  

