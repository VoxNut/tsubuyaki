[English](README.md) | **Tiếng Việt** | [日本語](README.ja.md)

# tsubuyaki

Trình đọc media tiếng Nhật có Furigana theo ngữ cảnh. Ứng dụng tạo cách đọc
có thể chỉnh sửa cho phụ đề, đồng thời giữ nguyên luồng học nghe quen thuộc
của kikiyomi.

[Mở ứng dụng công khai](https://voxnuts947-tsubuyaki.hf.space) ·
[Hugging Face Space](https://huggingface.co/spaces/voxnuts947/tsubuyaki) ·
[Notebook Kaggle](https://www.kaggle.com/code/voxnuts465/furigana-aid-generation)

## Hình ảnh

### Thư viện media

![Thư viện media của tsubuyaki](https://raw.githubusercontent.com/VoxNut/tsubuyaki/main/demo%20images/home%20page.png)

### Trình đọc với Furigana theo ngữ cảnh

![Trình đọc tsubuyaki](https://raw.githubusercontent.com/VoxNut/tsubuyaki/main/demo%20images/main%20page.png)

## Những thay đổi chính

Trải nghiệm học nghe ban đầu của kikiyomi đã được mở rộng thành một trình đọc
hỗ trợ Furigana hoàn chỉnh:

- Phân giải cách đọc theo ngữ cảnh bằng Japanese BERT đã fine-tune và MLP nhận
  biết vị trí từ mục tiêu.
- Candidate mask và prior riêng cho từng surface giúp loại bỏ các cách đọc
  không hợp lệ.
- Ensemble BERT–MLP được hiệu chỉnh trên validation, kèm Most-Frequent và
  MeCab fallback.
- Furigana chỉ nằm trên phần Kanji; okurigana hiển thị bên ngoài thẻ ruby.
- Hai chế độ `Always` và `On hover`, điều chỉnh kích thước reading và sửa cách
  đọc thủ công.
- JSON riêng của ứng dụng là định dạng round-trip chính; SRT có thẻ ruby là
  định dạng phụ để chia sẻ và nhập lại.
- Toàn bộ giao diện dùng tiếng Anh, bảng màu Rosé Pine và hệ icon SVG thống
  nhất, không dùng emoji làm điều khiển.
- Một Docker image duy nhất phục vụ cả frontend tĩnh và FastAPI.

## Tính năng

### Luồng sử dụng trình đọc

- Kéo thả video, audio, SRT hoặc JSON riêng của ứng dụng.
- Player tích hợp, tua theo phụ đề, phát lại, Focus mode, Chapters, Bookmarks,
  lịch sử nghe và theo dõi tiến độ.
- Hỗ trợ bàn phím, gamepad, desktop và mobile.
- Nhận các container MP4, WebM, MKV, MOV, M4V, OGV, AVI, MPEG/MPG, MP3,
  M4A/M4B, AAC, OGG/OGA, OPUS, WAV và FLAC.
- Khả năng phát thực tế còn phụ thuộc codec của trình duyệt. MP4 H.264/AAC và
  WebM VP9/Opus là hai lựa chọn an toàn nhất.

### Luồng tạo Furigana

- Tạo Furigana theo lô cho toàn bộ cue, có tiến trình và hủy tác vụ.
- Căn reading phía trên văn bản Nhật, tách đúng Kanji và okurigana.
- Lưu confidence và prediction source cho từng reading.
- Nhấp vào ruby để sửa hoặc xóa cách đọc.
- Xuất JSON hoặc `.furigana.srt` có thẻ ruby.

### JSON round-trip

JSON bảo toàn:

- timestamp và plain text của từng cue;
- ruby segment và reading;
- confidence và prediction source;
- chỉnh sửa thủ công;
- phiên bản model và artifact.

HTML chỉ là định dạng hiển thị, không phải cấu trúc dữ liệu chính.

## Pipeline suy luận

1. MeCab tách token cho từng cue phụ đề tiếng Nhật.
2. Surface đã biết được ánh xạ tới tập cách đọc hợp lệ.
3. Surface đa âm được đánh dấu trong câu bằng `[TGT]` và `[/TGT]`.
4. Japanese BERT tạo biểu diễn ngữ cảnh và classifier logits.
5. MLP target-aware dùng đặc trưng CLS, target marker và target span.
6. Điểm BERT và MLP sau candidate mask được kết hợp với surface log prior.
7. Dự đoán confidence thấp quay về cách đọc mode trong tập train.
8. Surface không cần ngữ cảnh hoặc chưa từng thấy dùng Most-Frequent hoặc
   MeCab fallback.
9. Reading được căn với phần Kanji rồi trả về dưới dạng JSON có cấu trúc.

Các prediction source gồm `ContextEnsemble`,
`MostFrequentLowConfidence`, `MostFrequent`, `MeCabFallback`, `PlainText` và
`ManualEdit`.

## Dữ liệu và khả năng tái tạo

Notebook train đọc streaming dataset
[`Calvin-Xu/Furigana-Aozora`](https://huggingface.co/datasets/Calvin-Xu/Furigana-Aozora)
và chia nhóm theo file để dữ liệu từ cùng một file nguồn không xuất hiện ở
nhiều tập.

| Hạng mục | Giá trị |
| --- | ---: |
| Random seed | 42 |
| Tổng số target | 259.476 |
| Train / validation / test | 207.635 / 25.914 / 25.927 |
| Surface trong train | 26.210 |
| Surface đa âm hợp lệ | 1.004 |
| Reading label | 1.491 |
| Độ dài chuỗi tối đa | 192 |
| Accuracy validation của target-aware MLP | 95,20% |
| Accuracy validation của ensemble | 95,46% |
| Accuracy validation sau hiệu chỉnh | 95,48% |

Model đang deploy nằm trong repository private và được pin bằng commit bất
biến. API kiểm tra checksum artifact, ánh xạ label, target marker của tokenizer
và metadata trước khi chuyển sang trạng thái ready.

## Kiến trúc

| Thành phần | Trách nhiệm |
| --- | --- |
| `frontend/` | Thư viện media, player, phụ đề, chỉnh Furigana và export |
| FastAPI | Phục vụ frontend và các endpoint `/api` trên cùng origin |
| BERT classifier | Tạo điểm cách đọc theo ngữ cảnh |
| Target-aware MLP | Tạo đặc trưng theo đúng từ mục tiêu và điểm cách đọc |
| Candidate artifacts | Mask, prior, mode, label và calibration |
| MeCab | Tokenization và cách đọc fallback |

Frontend và backend chạy cùng origin để việc deploy đơn giản và không cần một
biên CORS riêng.

## Chạy local bằng Docker

Build image CPU:

```bash
docker build -t tsubuyaki .
```

Chạy với thư mục model local:

```bash
docker run --rm -p 8000:7860 \
  -e FURIGANA_MODEL_LOCAL_DIR=/model \
  -e FURIGANA_DEVICE=cpu \
  -v /absolute/path/to/model:/model:ro \
  tsubuyaki
```

Mở `http://localhost:8000/`.

Để tải model private từ Hugging Face, cấu hình các biến môi trường hoặc secret
của nền tảng deploy:

```text
FURIGANA_HF_MODEL_REPO=owner/private-model
FURIGANA_HF_MODEL_REVISION=<immutable-40-character-commit-sha>
HF_TOKEN=<dedicated-read-only-token>
FURIGANA_DEVICE=cpu
```

Không commit access token hoặc đưa token vào Docker image.

## API

| Method | Endpoint | Mục đích |
| --- | --- | --- |
| `GET` | `/api/health` | Kiểm tra process |
| `GET` | `/api/ready` | Kiểm tra model và artifact đã sẵn sàng |
| `GET` | `/api/version` | Phiên bản API, model và artifact |
| `POST` | `/api/furigana/generate-batch` | Tạo Furigana có cấu trúc cho các cue |

## Deploy

Bản công khai đang chạy trên Hugging Face Docker Space với CPU Basic:

<https://voxnuts947-tsubuyaki.hf.space>

Dockerfile này cũng có thể chuyển sang VPS hoặc máy có GPU.
`FURIGANA_DEVICE=auto` chỉ chọn CUDA khi máy có GPU tương thích và PyTorch được
build với CUDA.

Xem [hướng dẫn deploy tiếng Việt](docs/DEPLOY.md) hoặc
[hướng dẫn deploy tiếng Anh](docs/DEPLOY_EN.md).

## Tài liệu bổ sung

- [Kiến trúc](docs/ARCHITECTURE.md)
- [Ghi chú triển khai Phase 1](docs/PHASE1.md)
- [Bảo mật và quản lý secret](docs/SECURITY.md)
- [Giới hạn của model](docs/MODEL_LIMITATIONS.md)

## Giới hạn

- Reading tự động vẫn có thể sai với tên riêng, từ hiếm, văn bản chuyên ngành
  và surface chưa từng thấy. Có thể dùng trình chỉnh reading để sửa.
- Cold start trên CPU bao gồm thời gian tải và kiểm tra model private.
- Trình duyệt nhận một container không đồng nghĩa với việc giải mã được mọi
  codec bên trong.
- Confidence là tín hiệu của model, không đảm bảo tuyệt đối tính đúng đắn về
  ngôn ngữ.

## Ghi công và giấy phép

Luồng sử dụng player được phát triển dựa trên
[`rtr46/kikiyomi`](https://github.com/rtr46/kikiyomi). Thông tin ghi công được
giữ trong [NOTICE.md](NOTICE.md), và dự án tiếp tục sử dụng giấy phép
[GPL-3.0](LICENSE).

Giao diện dùng bảng màu chính thức của
[Rosé Pine](https://github.com/rose-pine/rose-pine-palette).
