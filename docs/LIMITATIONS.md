# Technical Limitations & Constraints

Tài liệu này tổng hợp các giới hạn kỹ thuật của hệ thống **tsubuyaki** nhằm hướng dẫn vận hành và định hướng tối ưu trong tương lai.

## 1. Giới hạn tài nguyên và Hiệu năng (Resource & Performance)

- **CPU Inference:** 
  Hệ thống được thiết kế chạy trên CPU-default (khoảng ~14.5ms mỗi cue). Mặc dù tốc độ xử lý nhanh, việc chạy đa luồng đồng thời hoặc phục vụ quá nhiều người dùng cùng lúc trên cấu hình CPU 1-core (free tier của HF Spaces) có thể gây tăng đột biến latency.
- **Dung lượng RAM:**
  Model BERT (khoảng 350MB) + MLP (khoảng vài chục MB) tiêu tốn khoảng ~500-800MB RAM khi chạy thực tế. Ở các môi trường bộ nhớ giới hạn (< 512MB RAM), hệ thống có khả năng bị OOM (Out Of Memory). Nên chạy trên môi trường có ít nhất 1GB RAM.

---

## 2. Giới hạn API và Cấu hình an toàn (API Limits)

Để tránh tình trạng quá tải (Denial of Service) và OOM khi xử lý dữ liệu lớn, backend áp dụng các validator cực kỳ nghiêm ngặt:
- **Số lượng cues tối đa mỗi request (`max_cues_per_request`):** Mặc định là **64 cues** (tối đa cho phép 512). Request vượt quá giới hạn này sẽ trả về lỗi `422 Unprocessable Content`.
- **Độ dài ký tự tối đa mỗi cue (`max_chars_per_cue`):** Mặc định là **2,000 ký tự**. Một câu thoại dài bất thường hoặc bị lỗi lặp ký tự vượt ngưỡng sẽ bị từ chối.
- **Tổng số ký tự tối đa mỗi request (`max_total_chars`):** Mặc định là **32,000 ký tự**.

---

## 3. Cơ chế Fallback sang MeCab (Fallback Mechanism)

Hệ thống sử dụng mô hình học máy BERT kết hợp MLP để dự đoán cách đọc Kanji theo ngữ cảnh. Tuy nhiên:
- Đối với các từ Kanji không có trong danh sách nhãn dự đoán (out-of-vocabulary labels) hoặc khi mô hình trả về kết quả có độ tin cậy thấp, hệ thống sẽ tự động chuyển sang cơ chế **fallback** sử dụng công cụ từ điển phân tích từ vựng **MeCab** (hoặc `fugashi` / `ipadic`).
- Độ chính xác của fallback phụ thuộc hoàn toàn vào chất lượng từ điển ipadic cài đặt kèm theo.

---

## 4. Hạn chế về định dạng tệp (Format Constraints)

- **Đầu vào phụ đề:**
  Ứng dụng client chỉ hỗ trợ phân tích định dạng **SRT** (chuẩn SubRip UTF-8) hoặc tệp PWA-native **`.furigana.json`**. Các định dạng khác như ASS, VTT cần được convert trước khi import vào player.
- **Bảo mật XSS ở Client:**
  Để đảm bảo an toàn, mọi thẻ HTML lạ nằm ngoài whitelist `<ruby>` và `<rt>` đều bị vô hiệu hóa và biến thành text thô. Do đó, các định dạng trang trí phụ đề nâng cao (như inline styles, thẻ font màu mè) sẽ bị loại bỏ hoàn toàn trong quá trình parse.
