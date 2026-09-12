# SVM Moons Demo — Linear SVM vs RBF SVM

Dự án nhỏ minh họa sự khác biệt giữa **Linear SVM** và **RBF SVM** trên dataset
phi tuyến `make_moons`, giúp hiểu trực quan 3 khái niệm cốt lõi của SVM:
**margin**, **support vectors**, và **kernel trick**.

## Cấu trúc dự án

```
svm-moons-demo/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── src/
│   ├── train_compare_svm.py
│   └── api.py
├── outputs/                  # ảnh kết quả được sinh ra ở đây
└── README.md
```

## Yêu cầu

- Docker và Docker Compose đã cài đặt và đang chạy (Docker Desktop mở sẵn)

Không cần cài Python hay bất kỳ thư viện nào trên máy host — mọi thứ chạy
trong container.

## Cách chạy

```bash
docker compose up --build
```

Lệnh này sẽ:
1. Build image (cài Python 3.11 + các thư viện trong `requirements.txt`)
2. Khởi động API tại `http://localhost:8000`

Mở `http://localhost:8000/docs` để xem và thử endpoint trực tiếp trên Swagger UI.

Dừng bằng `Ctrl+C`, dọn dẹp container bằng:

```bash
docker compose down
```

Muốn build lại từ đầu (ví dụ sau khi sửa code) mà không dùng cache cũ:

```bash
docker compose build --no-cache
docker compose up
```

## Kết quả sau khi chạy

- **Console**: in ra `accuracy`, `confusion matrix`, `classification report`
  cho cả 2 model, và một dòng kết luận ngắn so sánh Linear SVM vs RBF SVM.
- **Ảnh**: file `outputs/svm_linear_vs_rbf.png` tự động xuất hiện trên máy host
  (nhờ volume mount trong `docker-compose.yml`), gồm 2 biểu đồ đặt cạnh nhau:
  - Bên trái: **Linear SVM** — đường biên thẳng, thường tách sai nhiều điểm
    vì dữ liệu moons có hình cong.
  - Bên phải: **RBF SVM** — đường biên uốn cong theo hình dạng dữ liệu nhờ
    kernel trick, accuracy cao hơn.
  - Trên mỗi biểu đồ: đường liền là decision boundary, 2 đường đứt nét là
    margin, các điểm khoanh viền xanh lá là support vectors.

## Gọi API dự đoán

API dùng RBF SVM và nhận 2 đặc trưng của dataset `make_moons`:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post `
  -ContentType "application/json" `
  -Body '{"feature_1": 0.5, "feature_2": -0.2}'
```

API trả về `prediction`, `label`, `decision_score` và tên model.

## Chia sẻ API qua ngrok

Giữ Docker đang chạy, mở terminal khác và chạy:

```bash
ngrok http 8000
```

Ngrok sẽ cấp một URL HTTPS, ví dụ
`https://7216-34-9-84-250.ngrok-free.app`. Máy khác gọi endpoint bằng cách
thêm `/predict` vào URL:

```powershell
Invoke-RestMethod -Uri "https://7216-34-9-84-250.ngrok-free.app/predict" `
  -Method Post -ContentType "application/json" `
  -Body '{"feature_1": 0.5, "feature_2": -0.2}'
```

Link miễn phí thường thay đổi sau mỗi lần khởi động lại. Máy chạy Docker và
ngrok phải luôn bật trong lúc máy khác gọi API.

## Dữ liệu dùng để huấn luyện

`make_moons(n_samples=300, noise=0.25, random_state=42)` — dataset 2 lớp hình
lưỡi liềm, cố định `random_state` để kết quả tái lập được giữa các lần chạy.
Dữ liệu được chuẩn hóa bằng `StandardScaler` trước khi đưa vào SVM.

## Model

| Model      | Tham số                          |
|------------|-----------------------------------|
| Linear SVM | `kernel="linear", C=1.0`          |
| RBF SVM    | `kernel="rbf", C=1.0, gamma="scale"` |

## Thử nghiệm thêm

Muốn hiểu sâu hơn về ảnh hưởng của tham số, có thể sửa trực tiếp trong
`src/train_compare_svm.py` rồi chạy lại `docker compose up --build`:

- Đổi `noise` trong `make_moons` (0.1 → dễ tách hơn, 0.4 → khó tách hơn)
- Đổi `gamma` của RBF SVM sang một số cụ thể (vd `gamma=0.1` hoặc `gamma=10`)
  để thấy hiện tượng underfitting / overfitting
- Đổi `C` (nhỏ → margin rộng hơn, chấp nhận nhiều lỗi hơn; lớn → margin hẹp,
  cố phân loại đúng hết)

## Ghi chú kỹ thuật

- `MPLBACKEND=Agg` được set trong `docker-compose.yml` vì container không có
  màn hình — bắt buộc để `matplotlib` chạy được mà không lỗi.
- Script không gọi `plt.show()`, chỉ `fig.savefig(...)`, phù hợp với môi
  trường không có GUI như container.
