from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from furigana_aid.config import Settings
from furigana_aid.runtime import build_runtime
from furigana_aid.api.endpoints import router

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.settings = settings
    app.state.runtime_error = None

    logging.basicConfig(level=settings.log_level)
    logger = logging.getLogger("furigana_aid")
    logger.info(f"Khởi động backend service với cấu hình: {settings.redacted_summary()}")

    try:
        runtime = build_runtime(settings)
        app.state.runtime = runtime
        logger.info(f"Model đã load thành công trên thiết bị: {runtime.device}")
    except Exception as exc:
        logger.critical(f"Lỗi nghiêm trọng khi khởi động/tải model: {exc}", exc_info=True)
        app.state.runtime = None
        app.state.runtime_error = str(exc)
        raise exc

    yield

    logger.info("Đang dừng backend service...")


app = FastAPI(
    title="tsubuyaki API",
    description="API phân giải cách đọc Kanji tiếng Nhật bằng BERT + MLP",
    version="1.0.0",
    lifespan=lifespan,
)

cors_kwargs = {
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}
if "*" in settings.cors_origins:
    cors_kwargs["allow_origins"] = ["*"]
    cors_kwargs["allow_credentials"] = False
else:
    cors_kwargs["allow_origins"] = list(settings.cors_origins)
    cors_kwargs["allow_credentials"] = True

app.add_middleware(
    CORSMiddleware,
    **cors_kwargs
)

app.include_router(router)

# Mount the SPA last so /api routes keep precedence. The same path works both
# from the source tree and from /app in the Docker image.
frontend_dir = Path(__file__).resolve().parents[3] / "frontend"
if frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
