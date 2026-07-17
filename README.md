---
title: tsubuyaki
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
fullWidth: true
license: gpl-3.0
models:
  - voxnuts947/furigana-aid-model
datasets:
  - Calvin-Xu/Furigana-Aozora
short_description: Context-aware Japanese media reader with BERT, MLP, and MeCab furigana.
---

**English** | [Tiếng Việt](README.vi.md) | [日本語](README.ja.md)

# tsubuyaki

A context-aware Japanese media reader that generates editable furigana for
subtitle text while preserving the original media-learning workflow.

[Open the public application](https://voxnuts947-tsubuyaki.hf.space) ·
[Hugging Face Space](https://huggingface.co/spaces/voxnuts947/tsubuyaki) ·
[Kaggle notebook](https://www.kaggle.com/code/voxnuts465/furigana-aid-generation)

## Screenshots

### Media library

![tsubuyaki media library](https://raw.githubusercontent.com/VoxNut/tsubuyaki/main/demo%20images/home%20page.png)

### Reader with context-aware furigana

![tsubuyaki reader](https://raw.githubusercontent.com/VoxNut/tsubuyaki/main/demo%20images/main%20page.png)

## What changed

The original kikiyomi listening experience has been extended into a complete
furigana-assisted reader:

- Context-aware disambiguation using a fine-tuned Japanese BERT classifier and
  a target-aware MLP.
- Surface-specific candidate masks and reading priors prevent impossible
  predictions.
- Validation-tuned BERT/MLP ensemble with Most-Frequent and MeCab fallbacks.
- Furigana is aligned above the Kanji portion only; visible okurigana remains
  outside the ruby annotation.
- `Always` and `On hover` display modes, adjustable reading size, and manual
  reading edits.
- App-native JSON is the round-trip source of truth; ruby-tagged SRT remains
  available for sharing and re-import.
- A redesigned English interface using the Rosé Pine palette and a consistent
  inline SVG icon system without emoji UI controls.
- One portable Docker image serves both the static frontend and FastAPI API.

## Main features

### Reader workflow

- Drag and drop video, audio, SRT, or app-native JSON files.
- Built-in video/audio player, subtitle seeking, replay, focus mode, chapters,
  bookmarks, listening history, and progress tracking.
- Keyboard, gamepad, desktop, and mobile-friendly controls.
- Supported containers include MP4, WebM, MKV, MOV, M4V, OGV, AVI, MPEG/MPG,
  MP3, M4A/M4B, AAC, OGG/OGA, OPUS, WAV, and FLAC.
- Actual playback still depends on browser codec support. MP4 with H.264/AAC
  and WebM with VP9/Opus are the safest formats.

### Furigana workflow

- Batch generation for subtitle cues with progress and cancellation.
- Readings displayed above Japanese text with correct Kanji/okurigana
  alignment.
- Per-reading source and confidence retained in JSON.
- Click a ruby segment to edit or remove its reading.
- Export app-native JSON or `.furigana.srt` with ruby tags.

### Round-trip JSON

The JSON format preserves:

- cue timestamps and plain text;
- ruby segments and readings;
- confidence and prediction source;
- manual edits;
- model and artifact versions.

HTML is a rendering target, not the application data model.

## Inference pipeline

1. MeCab tokenizes each Japanese subtitle cue.
2. Known surfaces are mapped to allowed reading candidates.
3. Ambiguous surfaces are marked inside their sentence with `[TGT]` and
   `[/TGT]` tokens.
4. Japanese BERT produces contextual representations and classifier logits.
5. A target-aware MLP uses CLS, target-marker, and target-span features.
6. Candidate-masked BERT and MLP scores are combined with a surface log prior.
7. Low-confidence predictions fall back to the training-set mode reading.
8. Non-contextual or unseen surfaces use Most-Frequent or MeCab fallback.
9. The selected reading is aligned to the Kanji base and returned as structured
   JSON segments.

Prediction sources reported by the API are `ContextEnsemble`,
`MostFrequentLowConfidence`, `MostFrequent`, `MeCabFallback`, `PlainText`, and
`ManualEdit`.

## Data and reproducibility

The training notebook streams
[`Calvin-Xu/Furigana-Aozora`](https://huggingface.co/datasets/Calvin-Xu/Furigana-Aozora)
and uses a file-grouped split to prevent examples from the same source file
from crossing train, validation, and test partitions.

| Item | Value |
| --- | ---: |
| Random seed | 42 |
| All target examples | 259,476 |
| Train / validation / test | 207,635 / 25,914 / 25,927 |
| Training surfaces | 26,210 |
| Valid ambiguous surfaces | 1,004 |
| Reading labels | 1,491 |
| Maximum sequence length | 192 |
| Target-aware MLP validation accuracy | 95.20% |
| Ensemble validation accuracy | 95.46% |
| Tuned validation accuracy | 95.48% |

The deployed model repository is private and pinned to an immutable commit.
Artifact checksums, label mappings, tokenizer markers, and metadata are
validated before the API becomes ready.

## Architecture

| Component | Responsibility |
| --- | --- |
| `frontend/` | Media library, player, subtitle reader, furigana editing, and export |
| FastAPI | Static frontend hosting and `/api` endpoints on the same origin |
| BERT classifier | Context-sensitive reading scores |
| Target-aware MLP | Target-specific contextual features and reading scores |
| Candidate artifacts | Surface masks, priors, modes, labels, and calibration |
| MeCab | Tokenization and fallback readings |

Serving the frontend and backend from one origin keeps deployment simple and
avoids a separate CORS boundary.

## Run locally with Docker

Build the CPU image:

```bash
docker build -t tsubuyaki .
```

Run with a local model directory:

```bash
docker run --rm -p 8000:7860 \
  -e FURIGANA_MODEL_LOCAL_DIR=/model \
  -e FURIGANA_DEVICE=cpu \
  -v /absolute/path/to/model:/model:ro \
  tsubuyaki
```

Open `http://localhost:8000/`.

To load a private model from Hugging Face instead, configure these runtime
values through environment variables or platform secrets:

```text
FURIGANA_HF_MODEL_REPO=owner/private-model
FURIGANA_HF_MODEL_REVISION=<immutable-40-character-commit-sha>
HF_TOKEN=<dedicated-read-only-token>
FURIGANA_DEVICE=cpu
```

Never commit an access token or bake it into a Docker image.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Process health |
| `GET` | `/api/ready` | Model and artifact readiness |
| `GET` | `/api/version` | API, model, and artifact versions |
| `POST` | `/api/furigana/generate-batch` | Generate structured furigana for subtitle cues |

## Deployment

The public deployment runs on a Hugging Face Docker Space using CPU Basic:

<https://voxnuts947-tsubuyaki.hf.space>

The same Dockerfile is portable to a VPS or GPU host. `FURIGANA_DEVICE=auto`
selects CUDA only when both compatible hardware and a CUDA-enabled PyTorch
build are available.

See [the English deployment guide](docs/DEPLOY_EN.md) or
[the Vietnamese deployment guide](docs/DEPLOY.md) for details.

## Further documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Phase 1 implementation notes](docs/PHASE1.md)
- [Security and secret handling](docs/SECURITY.md)
- [Model limitations](docs/MODEL_LIMITATIONS.md)

## Limitations

- Automatic readings can still be wrong, especially for names, rare terms,
  domain-specific text, and unseen surfaces. Use the reading editor when
  needed.
- CPU cold start includes downloading and validating the private model.
- Browser container support does not guarantee that every internal codec can
  be decoded.
- Confidence is a model signal, not a guarantee of linguistic correctness.

## Attribution and license

The player workflow is adapted from
[`rtr46/kikiyomi`](https://github.com/rtr46/kikiyomi). Attribution is retained
in [NOTICE.md](NOTICE.md), and the project remains licensed under
[GPL-3.0](LICENSE).

The interface uses the official
[Rosé Pine palette](https://github.com/rose-pine/rose-pine-palette).
