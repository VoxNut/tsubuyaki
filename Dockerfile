# syntax=docker/dockerfile:1.7
FROM python:3.12.13-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend/src \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PORT=7860 \
    FURIGANA_DEVICE=auto \
    HF_HOME=/home/app/.cache/huggingface \
    FURIGANA_HF_CACHE_DIR=/home/app/.cache/huggingface/hub

WORKDIR /app

RUN groupadd --system app \
    && useradd --system --gid app --create-home app

COPY backend/requirements-core.txt /tmp/requirements-core.txt
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --no-compile \
      --index-url "${TORCH_INDEX_URL}" torch==2.13.0 \
    && python -m pip install --no-compile \
      -r /tmp/requirements-core.txt

COPY backend/src /app/backend/src
COPY frontend /app/frontend
COPY LICENSE NOTICE.md README.md /app/

RUN mkdir -p /home/app/.cache/huggingface/hub \
    && chown -R app:app /app /home/app

USER app
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=3)"

CMD ["sh", "-c", "exec python -m uvicorn furigana_aid.main:app --host 0.0.0.0 --port \"${PORT:-7860}\""]
