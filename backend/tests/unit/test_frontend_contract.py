from __future__ import annotations

from pathlib import Path


FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"


def test_rosé_pine_main_palette_and_branding() -> None:
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

    for color in (
        "#191724",  # base
        "#1f1d2e",  # surface
        "#26233a",  # overlay
        "#e0def4",  # text
        "#eb6f92",  # love
        "#c4a7e7",  # iris
    ):
        assert color in html
    assert "<title>tsubuyaki</title>" in html
    assert ">tsubuyaki</h1>" in html


def test_media_picker_and_player_cover_requested_formats() -> None:
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

    for extension in (
        "mp4", "m4v", "webm", "mov", "mkv", "ogv", "avi", "mpeg", "mpg",
        "mp3", "m4b", "aac", "m4a", "ogg", "oga", "opus", "wav", "flac",
    ):
        assert f"'{extension}'" in html
        assert f".{extension}" in html
    assert '<video id="media-el"' in html
    assert "internal codecs must also be browser-compatible" in html
    assert "this.elSubList.scrollTo" in html


def test_ruby_readings_render_above_base_text() -> None:
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

    assert "ruby-position: over" in html
    assert "display: ruby;" in html
    assert "display: ruby-text;" in html
    assert 'id="ruby-size-value"' in html


def test_user_interface_is_english_only() -> None:
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

    for vietnamese_text in (
        "Tạo Furigana",
        "Chưa nạp phụ đề",
        "Sinh Furigana",
        "Không có dữ liệu",
        "Vui lòng nhập",
        "Trình duyệt không",
        "Chỉ hỗ trợ file",
    ):
        assert vietnamese_text not in html

    for english_text in (
        "Generate furigana",
        "Backend offline",
        "Reading size",
        "Remove reading",
        "Keyboard shortcuts",
    ):
        assert english_text in html


def test_pwa_brand_and_cache_are_updated() -> None:
    manifest = (FRONTEND_DIR / "manifest.json").read_text(encoding="utf-8")
    service_worker = (FRONTEND_DIR / "service-worker.js").read_text(encoding="utf-8")

    assert '"name": "tsubuyaki"' in manifest
    assert '"theme_color": "#191724"' in manifest
    assert "tsubuyaki-v2" in service_worker
    assert "fetch(e.request)" in service_worker
