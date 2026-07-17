# Hướng dẫn triển khai Furigana Aid Reader

Tài liệu này hướng dẫn chi tiết cách đóng gói, tải mô hình lên Hugging Face Hub (Private) và triển khai dịch vụ backend của ứng dụng Furigana Aid Reader.

## 1. Yêu cầu hệ thống
- Docker và Docker Compose (nếu chạy local container)
- Tài khoản Hugging Face và một Hugging Face Token (có quyền write)
- Python 3.12+ (để chạy script upload)

---

## 2. Tải model artifacts lên Hugging Face Hub (Private Repo)

Để backend có thể tự động tải model về khi chạy trên đám mây mà vẫn giữ bảo mật cho mô hình tự train của bạn, hãy đẩy các tệp mô hình lên một repository ở chế độ **Private**:

1. **Chuẩn bị Token:**
   Lấy Hugging Face Write Token từ trang cài đặt tài khoản của bạn (Settings -> Access Tokens).
   Thiết lập biến môi trường cục bộ:
   ```bash
   # Windows (cmd)
   set HF_TOKEN=your_token_here
   # Linux/macOS
   export HF_TOKEN=your_token_here
   ```

2. **Chạy script upload:**
   Sử dụng script `scripts/hf_upload.py` để tự động tạo repository Private và đẩy các tệp trong thư mục `model/` lên:
   ```bash
   python scripts/hf_upload.py --repo_id "username/furigana-aid-model" --model_dir "/path/to/your/model"
   ```
   *Lưu ý: Thay thế `username/furigana-aid-model` bằng ID tài khoản và tên repo mong muốn của bạn.*

---

## 3. Chạy thử nghiệm local bằng Docker

Bạn có thể chạy thử nghiệm backend dưới dạng Docker container tại máy cá nhân trước khi deploy thật:

1. **Build Docker image:**
   Chạy lệnh sau tại thư mục gốc của dự án (`tsubuyaki/`):
   ```bash
   docker build -t furigana-aid-backend .
   ```

2. **Chạy Container:**
   ```bash
   docker run -d -p 8000:7860 \
     -e FURIGANA_MODEL_LOCAL_DIR="" \
     -e FURIGANA_HF_MODEL_REPO="username/furigana-aid-model" \
     -e FURIGANA_HF_MODEL_REVISION="main" \
     -e HF_TOKEN="your_token_here" \
     furigana-aid-backend
   ```
   *Lưu ý: Truy cập `http://localhost:8000/api/health` để kiểm tra trạng thái.*

---

## 4. Triển khai lên Hugging Face Spaces (Khuyên dùng - Free CPU Tier)

Hugging Face Spaces hỗ trợ chạy Docker containers hoàn toàn miễn phí trên CPU tier.

1. **Tạo Space mới:**
   - Truy cập Hugging Face Spaces và click **Create a new Space**.
   - Nhập tên Space và chọn Space SDK là **Docker** (chọn template Blank).
   - Chọn chế độ hiển thị (Public hoặc Private tùy ý bạn).

2. **Cấu hình Secret Variables:**
   Trong trang Space mới tạo, truy cập **Settings -> Variables and secrets** và thêm secret sau:
   - Name: `HF_TOKEN`
   - Value: *Hugging Face Access Token của bạn (cần có quyền read để tải model private)*

3. **Cấu hình Environment Variables:**
   Cũng trong mục Settings, thêm các biến môi trường sau:
   - `FURIGANA_HF_MODEL_REPO` = `username/furigana-aid-model` (ID Repo chứa model của bạn)
   - `FURIGANA_HF_MODEL_REVISION` = `main`
   - `PORT` = `7860` (Cổng mặc định của HF Spaces)

4. **Đẩy mã nguồn lên Space:**
   - Clone Git repository của Space mới tạo về máy.
   - Sao chép toàn bộ code dự án của dự án `tsubuyaki` (bao gồm `Dockerfile`, `backend/`, `frontend/`) vào thư mục Space đó.
   - Commit và push mã nguồn lên branch `main` của Hugging Face Space.
   - Space sẽ tự động build Docker image và khởi chạy.

---

## 5. Triển khai Frontend

Giao diện Frontend của ứng dụng là một trang web tĩnh (client-only static page) nằm trong thư mục `frontend/`.

- **Chạy local:**
  Bạn chỉ cần mở trực tiếp file `frontend/index.html` trên trình duyệt hoặc chạy một HTTP server siêu nhẹ:
  ```bash
  cd frontend
  python -m http.server 8080
  ```
- **Deploy:**
  Vì frontend chỉ gồm các file tĩnh (`index.html`, `manifest.json`, `service-worker.js`), bạn có thể deploy miễn phí lên GitHub Pages, Vercel, Netlify, Cloudflare Pages hoặc lưu trữ trực tiếp cùng repo.
- **Cấu hình kết nối:**
  Khi mở ứng dụng, truy cập **Settings (⚙)** trên giao diện và nhập URL backend vừa deploy (ví dụ: `https://username-space-name.hf.space`) vào ô **API Server Endpoint**. Nhấn **Tạo Furigana** để bắt đầu sử dụng!
