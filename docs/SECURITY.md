# Security notes

## Secrets

- `HF_TOKEN` is accepted only through process environment or platform secret.
- The token is never included in API responses, readiness output, log
  messages, Docker build arguments, Docker image layers, or committed files.
- `.env`, model weights, downloaded snapshots, caches, subtitle reports, and
  generated artifacts are ignored by Git and excluded from Docker build
  context.

## Artifact loading

- NumPy artifacts use `np.load(..., allow_pickle=False)`.
- Production MLP weights use Safetensors.
- The conversion utility accepts only the project owner's trusted local `.pt`
  and calls `torch.load(..., weights_only=True)`. There is no unsafe fallback.
- Every declared artifact/model file is checked against SHA-256 before the
  runtime becomes ready.
- Schema, dtype, shape, candidate ranges, label-map hash, surface order,
  tokenizer marker IDs, MLP metadata, and tuning values are validated.
- A mismatch prevents readiness; the service must not silently switch to raw
  global softmax or a different model.

## Requests and logs

- Requests contain parsed cues, never arbitrary server file paths.
- Cue count, per-cue length, total length, and timestamps are validated.
- Log records should contain request ID, counts, latency, status, and model
  version; subtitle text is not logged by default.
- CORS uses an environment allowlist. Credentialed production deployments
  must not use wildcard origins.

Rate limiting, reverse-proxy request limits, public-domain CORS settings, and
malicious subtitle sanitization are completed during Phase 3 because they
depend on the chosen frontend and hosting boundary.
