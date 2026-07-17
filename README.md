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
short_description: Japanese media reader with BERT, MLP, and MeCab furigana.
---

# tsubuyaki

Context-aware Japanese media reader. The application
combines a fine-tuned Japanese BERT classification head, a target-aware MLP,
surface-specific candidate priors, Most-Frequent fallback, and MeCab. Its
single Docker image serves both the browser UI and the inference API.

The player workflow is adapted from
[kikiyomi](https://github.com/rtr46/kikiyomi). Its attribution and the
GPL-3.0 license are preserved in [NOTICE.md](NOTICE.md) and [LICENSE](LICENSE).
The UI uses the official
[Rosé Pine palette](https://github.com/rose-pine/rose-pine-palette).

## Current status

- CPU is the default runtime.
- `FURIGANA_DEVICE=auto` selects CUDA when the installed PyTorch build and host
  both support it.
- A private Hugging Face model repository is supported through `HF_TOKEN`.
- Model repository visibility is configuration-only; changing private to
  public does not change the inference pipeline.
- The service refuses to become ready when artifact checksums, label mapping,
  tokenizer markers, or model metadata do not match.
- No weights or secrets are committed to this repository.
- The media picker accepts MP4, WebM, MKV, MOV, M4V, OGV, AVI, MPEG/MPG and
  common audio formats. Actual playback still depends on the codecs supported
  by the user's browser; WebM (VP9/Opus) and MP4 (H.264/AAC) are the safest
  exchange formats.

The completed Kaggle weights are available locally, but production
`HybridTuned` inference additionally requires the Phase 0 CSR-like artifacts.
Use `scripts/recover_phase0_artifacts.py` only to rebuild dataset statistics
and candidates. It never fine-tunes BERT or trains the MLP.

## Local setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt
Copy-Item backend\.env.example backend\.env
```

Set either a local model directory:

```text
FURIGANA_MODEL_LOCAL_DIR=D:\path\to\model
```

or a pinned Hugging Face repository:

```text
FURIGANA_HF_MODEL_REPO=owner/private-model
FURIGANA_HF_MODEL_REVISION=<commit-sha>
HF_TOKEN=<read-token-in-environment-or-platform-secret>
```

Run:

```powershell
$env:PYTHONPATH = "backend\src"
.\.venv\Scripts\python.exe -m uvicorn furigana_aid.main:app `
  --host 127.0.0.1 --port 7860
```

Test:

```powershell
$env:PYTHONPATH = "backend\src"
.\.venv\Scripts\python.exe -m pytest backend\tests
```

Open `http://127.0.0.1:7860/` for the tsubuyaki UI. API endpoints:

- `GET /api/health`
- `GET /api/ready`
- `GET /api/version`
- `POST /api/furigana/generate-batch`

The Docker image exposes the UI and API on port `7860`. Model weights are downloaded
at container startup from the configured immutable Hub revision; they are not
baked into the image or committed to this repository.

See [Phase 1 notes](docs/PHASE1.md), [architecture](docs/ARCHITECTURE.md),
[security](docs/SECURITY.md), and [model limitations](docs/MODEL_LIMITATIONS.md).
