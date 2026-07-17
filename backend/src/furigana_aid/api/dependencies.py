"""FastAPI dependencies without mutable module-level model state."""

from __future__ import annotations

from typing import Any

from fastapi import Request


def get_runtime(Request: Request) -> Any:
    return Request.app.state.runtime
