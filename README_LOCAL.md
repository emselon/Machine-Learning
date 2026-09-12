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

## API dự đoán

Sau khi chạy `docker compose up --build`, API chạy tại
`http://localhost:8000`. Mở `http://localhost:8000/docs` để thử API.

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post `
	-ContentType "application/json" `
	-Body '{"feature_1": 0.5, "feature_2": -0.2}'
```

Muốn chia sẻ cho máy khác, mở terminal mới và chạy:

```bash
ngrok http 8000
```

Sau đó dùng URL HTTPS ngrok cấp, thêm `/predict` vào cuối URL. Model hiện tại
dùng hai đặc trưng của `make_moons`, không dùng các trường `area`, `rooms`,
`distance` của ví dụ bất động sản.