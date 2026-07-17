# Hướng dẫn triển khai tsubuyaki

Tài liệu này hướng dẫn chi tiết cách đóng gói, tải mô hình lên Hugging Face Hub (Private) và triển khai ứng dụng tsubuyaki.

## 1. Yêu cầu hệ thống
- Docker và Docker Compose (nếu chạy local container)
- Tài khoản Hugging Face; dùng token write cục bộ để upload và token read-only
  riêng cho Space
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
   python scripts/hf_upload.py --repo-id "username/furigana-aid-model" --model-dir "/path/to/your/model"
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
     -e FURIGANA_HF_MODEL_REVISION="<model-commit-sha>" \
     -e HF_TOKEN="your_token_here" \
     furigana-aid-backend
   ```
   *Lưu ý: Truy cập `http://localhost:8000/api/health` để kiểm tra trạng thái.*

---

## 4. Triển khai frontend và backend lên Hugging Face Spaces

Docker image của dự án phục vụ cả giao diện web và API FastAPI trên cùng cổng
`7860`. Đây là cách triển khai được khuyên dùng vì người dùng chỉ cần truy cập
một URL và trình duyệt không gặp lỗi CORS giữa frontend và backend.

1. **Tạo Space mới:**
   - Truy cập Hugging Face Spaces và click **Create a new Space**.
   - Nhập tên Space và chọn Space SDK là **Docker** (chọn template Blank).
   - Chọn **Public** để mọi người truy cập được cả ứng dụng và source code.
   - Nếu có Hugging Face PRO, có thể chọn **Protected** để ứng dụng vẫn public
     nhưng source code chỉ chủ sở hữu/cộng tác viên xem được.

2. **Cấu hình Secret Variables:**
   Trong trang Space mới tạo, truy cập **Settings -> Variables and secrets** và thêm secret sau:
   - Name: `HF_TOKEN`
   - Value: *Token read-only riêng, có quyền tải model private.*

   Không dùng token write đang dùng trên máy cá nhân làm secret lâu dài và không
   đặt token trong `README.md`, source code hoặc Docker image.

3. **Cấu hình Environment Variables:**
   Cũng trong mục Settings, thêm các biến môi trường sau:
   - `FURIGANA_HF_MODEL_REPO` = `username/furigana-aid-model` (ID Repo chứa model của bạn)
   - `FURIGANA_HF_MODEL_REVISION` = commit SHA bất biến do script upload in ra
   - `FURIGANA_DEVICE` = `cpu`
   - `FURIGANA_INFERENCE_BATCH_SIZE` = `8`
   - `PORT` = `7860` (cổng mặc định của HF Spaces)

   Không dùng branch `main` làm revision của model vì nội dung branch có thể
   thay đổi giữa hai lần khởi động.

4. **Đẩy mã nguồn lên Space:**
   - Clone Git repository của Space mới tạo về máy.
   - Sao chép toàn bộ code dự án của dự án `tsubuyaki` (bao gồm `Dockerfile`, `backend/`, `frontend/`) vào thư mục Space đó.
   - Commit và push mã nguồn lên branch `main` của Hugging Face Space.
   - Space sẽ tự động build Docker image và khởi chạy.

5. **Kiểm tra sau khi deploy:**
   - `https://<space-subdomain>.hf.space/api/health` phải trả về `{"status":"ok"}`.
   - `https://<space-subdomain>.hf.space/api/ready` phải trả về `"ready": true`.
   - Mở `https://<space-subdomain>.hf.space/` để sử dụng giao diện.

---

## 5. Khi nào cần triển khai frontend riêng?

Không cần triển khai frontend riêng trong cấu hình mặc định: FastAPI đã mount
thư mục `frontend/` tại `/`, còn API nằm tại `/api/` trên cùng domain.

- **Chạy frontend riêng để phát triển:**
  ```bash
  cd frontend
  python -m http.server 8080
  ```
- Nếu triển khai frontend lên GitHub Pages, Vercel, Netlify hoặc Cloudflare
  Pages, nhập URL backend tại **Settings -> API Server Endpoint** và thêm domain
  frontend vào `FURIGANA_CORS_ORIGINS` của backend.
- Với Docker Space mặc định, để trống **API Server Endpoint** để ứng dụng tự gọi
  `/api` trên cùng origin, sau đó chọn **Generate furigana**.
