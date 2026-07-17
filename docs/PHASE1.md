# Phase 1: backend inference service

## Decisions applied

- CPU-first and free/cheap-friendly.
- Same Python service can use CUDA when a CUDA PyTorch build is installed.
- Docker packaging is platform-neutral; Hugging Face Spaces Docker is an MVP
  candidate, not a hard dependency.
- Remote model repository is private initially and authenticated with
  `HF_TOKEN`.
- Remote revisions are immutable configuration; the service does not use
  unpinned `latest`.
- No cache, Redis, Celery, quantization, or ONNX optimization is enabled before
  benchmark evidence.

## Recovered inference artifacts

The local Kaggle outputs contained model weights and report files, but no
candidate/prior artifacts. The recovery script re-ran only:

- pinned Hugging Face dataset streaming;
- notebook normalization/extraction;
- the saved `split_assignments.csv`;
- train-surface statistics;
- candidate/prior construction;
- artifact export.

It did not fine-tune BERT or train the MLP.

Pinned dataset:

```text
Calvin-Xu/Furigana-Aozora
be9abdc9313a16146df74358c482015f2576ef03
```

Validated counts:

| Metric | Value |
|---|---:|
| All targets | 259,476 |
| Train | 207,635 |
| Validation | 25,914 |
| Test | 25,927 |
| Train surfaces | 26,210 |
| Contextual surfaces | 1,004 |
| Reading labels | 1,491 |

Recovery environment:

| Package | Version |
|---|---:|
| Python | 3.12.13 |
| datasets | 4.8.5 |
| NumPy | 2.5.1 |
| pandas | 2.3.3 |
| PyArrow | 25.0.0 |
| Transformers | 4.57.6 |

## Commands

Convert the trusted local MLP:

```powershell
.\.venv\Scripts\python.exe scripts\convert_mlp_to_safetensors.py
```

Recover non-neural artifacts:

```powershell
.\.venv\Scripts\python.exe scripts\recover_phase0_artifacts.py
```

Run backend tests:

```powershell
$env:PYTHONPATH = "backend\src"
.\.venv\Scripts\python.exe -m pytest backend\tests
```

Run the benchmark only after the real model/artifact directory is configured:

```powershell
$env:FURIGANA_MODEL_LOCAL_DIR = "D:\path\to\model"
$env:PYTHONPATH = "backend\src"
.\.venv\Scripts\python.exe scripts\benchmark_inference.py
```

Benchmark output is written as measured JSON. CPU suitability is not declared
until the 10/100/1,000-span measurements complete on the target hardware.
