# Furigana Aid Reader

Backend Phase 1 for a context-aware Japanese Furigana reader. The service
combines a fine-tuned Japanese BERT classification head, a target-aware MLP,
surface-specific candidate priors, Most-Frequent fallback, and MeCab.

The frontend adaptation of
[kikiyomi](https://github.com/rtr46/kikiyomi) is intentionally deferred to
Phase 2. The original player workflow and GPL-3.0 attribution will be
preserved.

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
.\.venv\Scripts\python.exe -m uvicorn furigana_aid.api.main:app `
  --host 127.0.0.1 --port 7860
```

Test:

```powershell
$env:PYTHONPATH = "backend\src"
.\.venv\Scripts\python.exe -m pytest backend\tests
```

API endpoints:

- `GET /api/health`
- `GET /api/ready`
- `GET /api/version`
- `POST /api/furigana/generate-batch`

See [Phase 1 notes](docs/PHASE1.md), [architecture](docs/ARCHITECTURE.md),
[security](docs/SECURITY.md), and [model limitations](docs/MODEL_LIMITATIONS.md).
