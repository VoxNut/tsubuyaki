from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from furigana_aid.main import app
from furigana_aid.api.schemas import PredictionSource
from furigana_aid.inference.engine import CueResult, TextSegment, RubySegment


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_endpoint_not_initialized(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(app.state._state, "runtime", None)
    monkeypatch.setitem(app.state._state, "runtime_error", "Mocked runtime error")

    response = client.get("/api/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is False
    assert data["error"] == "Mocked runtime error"


def test_ready_endpoint_ok(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    class MockRuntime:
        device = "cpu"
        model_id = "test-model-id"
        model_revision = "test-revision"
        artifact_schema_version = 1
        tokenizer_special_tokens_valid = True

    monkeypatch.setitem(app.state._state, "runtime", MockRuntime())

    response = client.get("/api/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["device"] == "cpu"
    assert data["model_id"] == "test-model-id"
    assert data["model_revision"] == "test-revision"
    assert data["artifact_schema_version"] == 1
    assert data["tokenizer_special_tokens_valid"] is True


def test_generate_batch_not_ready(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(app.state._state, "runtime", None)

    payload = {
        "request_id": "test-req-1",
        "cues": [
            {"cue_id": "1", "start_ms": 0, "end_ms": 1000, "text": "test"}
        ]
    }
    response = client.post("/api/furigana/generate-batch", json=payload)
    assert response.status_code == 503
    assert "not ready" in response.json()["detail"]


def test_generate_batch_limits(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    class MockSettings:
        max_cues_per_request = 2
        max_chars_per_cue = 10
        max_total_chars = 15

    class MockRuntime:
        pass

    monkeypatch.setitem(app.state._state, "settings", MockSettings())
    monkeypatch.setitem(app.state._state, "runtime", MockRuntime())

    # Test max_cues_per_request
    payload = {
        "request_id": "test-req-1",
        "cues": [
            {"cue_id": "1", "start_ms": 0, "end_ms": 1000, "text": "a"},
            {"cue_id": "2", "start_ms": 1000, "end_ms": 2000, "text": "b"},
            {"cue_id": "3", "start_ms": 2000, "end_ms": 3000, "text": "c"},
        ]
    }
    response = client.post("/api/furigana/generate-batch", json=payload)
    assert response.status_code == 422
    assert "Số lượng cues" in response.json()["detail"]

    # Test max_chars_per_cue
    payload = {
        "request_id": "test-req-1",
        "cues": [
            {"cue_id": "1", "start_ms": 0, "end_ms": 1000, "text": "abcdefghijkl"},
        ]
    }
    response = client.post("/api/furigana/generate-batch", json=payload)
    assert response.status_code == 422
    assert "vượt quá độ dài ký tự tối đa" in response.json()["detail"]

    # Test max_total_chars
    payload = {
        "request_id": "test-req-1",
        "cues": [
            {"cue_id": "1", "start_ms": 0, "end_ms": 1000, "text": "abcdefgh"},
            {"cue_id": "2", "start_ms": 1000, "end_ms": 2000, "text": "abcdefgh"},
        ]
    }
    response = client.post("/api/furigana/generate-batch", json=payload)
    assert response.status_code == 422
    assert "Tổng số ký tự" in response.json()["detail"]


def test_generate_batch_ok(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_results = [
        CueResult(
            cue_id="1",
            start_ms=0,
            end_ms=1000,
            plain_text="今日は晴れです。",
            segments=(
                RubySegment(
                    type="ruby",
                    base="今日",
                    reading="きょう",
                    source=PredictionSource.CONTEXT_ENSEMBLE.value,
                    confidence=0.95,
                    edited=False,
                ),
                TextSegment(type="text", text="は晴れです。"),
            )
        )
    ]

    class MockRuntime:
        device = "cpu"
        model_id = "test-model-id"
        model_revision = "test-revision"
        artifact_schema_version = 1
        tokenizer_special_tokens_valid = True

        def generate_batch(self, cues):
            return mock_results

    class MockSettings:
        max_cues_per_request = 64
        max_chars_per_cue = 2000
        max_total_chars = 32000

    monkeypatch.setitem(app.state._state, "settings", MockSettings())
    monkeypatch.setitem(app.state._state, "runtime", MockRuntime())

    payload = {
        "request_id": "test-req-1",
        "cues": [
            {"cue_id": "1", "start_ms": 0, "end_ms": 1000, "text": "今日は晴れです。"}
        ]
    }
    response = client.post("/api/furigana/generate-batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["request_id"] == "test-req-1"
    assert data["model_id"] == "test-model-id"
    assert data["model_revision"] == "test-revision"
    assert data["artifact_schema_version"] == 1

    cues = data["cues"]
    assert len(cues) == 1
    c = cues[0]
    assert c["cue_id"] == "1"
    assert c["plain_text"] == "今日は晴れです。"
    segments = c["segments"]
    assert len(segments) == 2
    assert segments[0]["type"] == "ruby"
    assert segments[0]["base"] == "今日"
    assert segments[0]["reading"] == "きょう"
    assert segments[0]["source"] == "ContextEnsemble"
    assert segments[0]["confidence"] == 0.95
    assert segments[0]["edited"] is False

    assert segments[1]["type"] == "text"
    assert segments[1]["text"] == "は晴れです。"
