# Backend architecture

## Scope

Phase 1 contains only the inference service. The kikiyomi-derived frontend,
native JSON round-trip, SRT import/export, and browser persistence are Phase 2
work.

## Runtime flow

```text
FastAPI lifespan
  -> resolve local directory or pinned Hugging Face snapshot
  -> verify checksums, manifest, label map, tokenizer markers and MLP metadata
  -> resolve auto/cpu/cuda device
  -> load BERT + tokenizer + MLP once in eval mode
  -> create fugashi/MeCab tokenizer and inference engine

POST /api/furigana/generate-batch
  -> validate cue count, timestamps and character limits
  -> MeCab tokenization with source offsets
  -> greedy longest known-surface merge (at most four tokens)
  -> MostFrequent for known non-contextual surfaces
  -> MeCab or PlainText fallback for unknown surfaces
  -> target-centered batching for contextual surfaces
  -> one BERT forward pass per batch
  -> BERT head logits + four-position final-layer feature pooling
  -> target-aware MLP logits
  -> candidate-only HybridTuned ensemble
  -> restore the original cue/span order
  -> return structured text/ruby segments
```

## Artifact contract

`inference_artifacts.npz` is loaded with `allow_pickle=False` and contains:

- `candidate_indptr` (`int64`);
- `candidate_label_ids` (`int32`);
- `candidate_log_priors` (`float32`);
- `surface_mode_label_ids` (`int32`).

`surfaces.json` defines the stable order of 1,004 contextual surfaces.
`surface_mode_readings.json` contains all 26,210 train-surface modes used by
greedy merge and Most-Frequent fallback. A contextual mode label ID of `-1` is
valid when the raw mode reading lies outside the 1,491-label classifier space.

The production MLP files are:

- `target_aware_mlp.safetensors`;
- `target_aware_mlp.metadata.json`.

The original `.pt` file is only accepted as a trusted conversion source and is
not required by the production manifest.

## Device and container portability

The application-level device selector supports:

- `auto`: CUDA when both the host and installed PyTorch build support it,
  otherwise CPU;
- `cpu`: force CPU;
- `cuda`: require CUDA and fail clearly when unavailable.

The CPU and CUDA images use the same application code. They install different
PyTorch wheels; a CPU-only wheel cannot become CUDA-capable merely because the
host has a GPU.

## Private Hugging Face repository

For a remote model, both repository ID and an immutable revision are
configuration. `HF_TOKEN` is read only from the environment/hosting secret.
Changing a model repository from private to public only changes whether a
token is supplied; artifact loading and inference do not change.
