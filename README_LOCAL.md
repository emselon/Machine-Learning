# So sánh Linear SVM và RBF SVM trên make_moons

Project Python nhỏ minh họa margin, support vectors và kernel trick bằng cách
so sánh hai mô hình SVM trên dữ liệu hình lưỡi liềm (`make_moons`):

- Linear SVM với `kernel="linear"`
- RBF SVM với `kernel="rbf"` và `gamma="scale"`

## Chạy bằng Docker

Từ thư mục gốc project, chạy:

```bash
docker compose up --build
```

Sau khi chạy xong, hình so sánh nằm tại:

```text
outputs/svm_linear_vs_rbf.png
```

Thư mục `outputs/` được mount vào container, vì vậy file ảnh được lưu trực
tiếp trên máy host. Log console hiển thị accuracy, classification report và
confusion matrix của cả hai model.