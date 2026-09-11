# TASK: So sánh Linear SVM vs RBF SVM trên dataset make_moons

> File này dùng để GitHub Copilot (Chat / Edit / Agent mode trong VS Code) đọc và
> generate code cho toàn bộ dự án. Copilot: hãy tạo đầy đủ các file được liệt kê
> ở mục "Cấu trúc dự án" theo đúng nội dung mô tả bên dưới.

## 1. Mục tiêu

Xây dựng một project Python nhỏ, chạy trong Docker, huấn luyện và so sánh
**2 model**:

1. **Linear SVM** (`kernel='linear'`)
2. **RBF SVM** (`kernel='rbf', gamma='scale'`)

trên dataset `make_moons` (từ `sklearn.datasets`), nhằm minh hoạ:

- **Margin** (lề) và **support vectors**
- **Kernel trick**: vì sao Linear SVM thất bại với dữ liệu phi tuyến (hình lưỡi
  liềm) trong khi RBF SVM tách được

## 2. Cấu trúc dự án

```
svm-moons-demo/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── src/
│   └── train_compare_svm.py
├── outputs/              # nơi lưu hình ảnh kết quả (mount volume)
└── README.md
```

## 3. requirements.txt

```
numpy
matplotlib
scikit-learn
```

## 4. Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# matplotlib cần vài lib hệ thống để render ảnh không cần GUI
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["python", "src/train_compare_svm.py"]
```

## 5. docker-compose.yml

```yaml
services:
  svm-demo:
    build: .
    volumes:
      - ./outputs:/app/outputs
    environment:
      - MPLBACKEND=Agg   # bắt buộc: chạy trong container không có màn hình
```

## 6. src/train_compare_svm.py — yêu cầu chi tiết

Copilot generate file này với các phần sau:

### 6.1. Import & tạo dữ liệu
- Dùng `make_moons(n_samples=300, noise=0.25, random_state=42)`
- Chia `train_test_split(test_size=0.3, random_state=42, stratify=y)`
- Chuẩn hoá bằng `StandardScaler` (fit trên train, transform cả train/test)

### 6.2. Định nghĩa đúng 2 model
```python
models = {
    "Linear SVM": SVC(kernel="linear", C=1.0),
    "RBF SVM":    SVC(kernel="rbf", C=1.0, gamma="scale"),
}
```

### 6.3. Hàm vẽ decision boundary
Viết hàm `plot_svm_decision_boundary(model, X, y, ax, title)`:
- Vẽ scatter điểm dữ liệu, tô màu theo nhãn
- Dùng `model.decision_function` để vẽ 3 đường contour ở level `[-1, 0, 1]`
  (2 đường margin nét đứt, 1 đường decision boundary nét liền)
- Khoanh tròn các `model.support_vectors_` bằng viền màu nổi bật (vd `lime`)

### 6.4. Train, đánh giá, và vẽ song song 2 model
- Dùng `plt.subplots(1, 2, figsize=(12, 5))`
- Với mỗi model: `fit` trên tập train đã scale, `predict` trên test, tính
  `accuracy_score`, in ra số lượng support vectors
- Title mỗi subplot ghi rõ: tên model, accuracy, số support vectors
- Lưu hình ra `outputs/svm_linear_vs_rbf.png` (đường dẫn tương đối, để hoạt
  động đúng khi chạy trong container)

### 6.5. In báo cáo ra console
- `classification_report` và `confusion_matrix` cho **cả 2 model**
- Một đoạn in kết luận ngắn dạng text, ví dụ:
  ```
  Linear SVM: accuracy=0.62, SV=95/210 -> không tách được vì moons phi tuyến
  RBF SVM:    accuracy=0.91, SV=60/210 -> kernel trick tách tốt hơn
  ```

### 6.6. Không dùng `plt.show()`
- Vì chạy trong Docker không có GUI, chỉ dùng `plt.savefig(...)`, không gọi
  `plt.show()`. Đảm bảo thư mục `outputs/` được tạo nếu chưa tồn tại
  (`os.makedirs("outputs", exist_ok=True)`).

## 7. README.md — nội dung cần có

Hướng dẫn chạy bằng Docker:

```bash
docker compose up --build
```

Sau khi chạy xong, kết quả nằm ở `outputs/svm_linear_vs_rbf.png` (đã mount
volume ra máy host) và log console hiển thị accuracy + confusion matrix của
2 model.

## 8. Ghi chú cho Copilot khi generate

- Code Python 3.11+, có type hint cơ bản cho các hàm chính
- Comment bằng tiếng Việt, ngắn gọn, giải thích ý nghĩa chứ không diễn giải
  lại code
- Không thêm thư viện ngoài danh sách ở `requirements.txt`
- Không cần argparse / CLI phức tạp — script chạy thẳng, tham số cố định
  trong code là đủ (đây là bài học minh hoạ, không phải sản phẩm production)