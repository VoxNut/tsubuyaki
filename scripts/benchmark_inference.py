from __future__ import annotations

import argparse
import json
import math
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil
import torch

from furigana_aid.config import Settings
from furigana_aid.inference.engine import Cue
from furigana_aid.runtime import build_runtime


SAMPLE_SENTENCES = (
    "今日は日曜日です。",
    "明日は学校で日本語を勉強します。",
    "この本を読んで意味を確認してください。",
    "友達と映画を見に行きました。",
    "東京駅で新しい電車に乗ります。",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the production Furigana inference runtime."
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=(10, 100, 1_000),
        help="Cue counts to benchmark (default: 10 100 1000).",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/latest.json"),
    )
    args = parser.parse_args()
    if any(size < 1 for size in args.sizes):
        parser.error("Every benchmark size must be at least 1.")
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be >= 0 and --repeats must be >= 1.")
    return args


def make_cues(size: int) -> list[Cue]:
    return [
        Cue(
            cue_id=str(index),
            start_ms=index * 2_000,
            end_ms=(index + 1) * 2_000,
            text=SAMPLE_SENTENCES[index % len(SAMPLE_SENTENCES)],
        )
        for index in range(size)
    ]


def percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def main() -> None:
    args = parse_args()
    runtime = build_runtime(Settings())
    process = psutil.Process()

    for _ in range(args.warmup):
        runtime.generate_batch(make_cues(min(args.sizes)))

    measurements: list[dict[str, object]] = []
    for size in args.sizes:
        cues = make_cues(size)
        durations: list[float] = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            results = runtime.generate_batch(cues)
            elapsed = time.perf_counter() - started
            if len(results) != size:
                raise RuntimeError(
                    f"Expected {size} results, received {len(results)}."
                )
            durations.append(elapsed)

        mean_seconds = sum(durations) / len(durations)
        measurements.append(
            {
                "cues": size,
                "repeats": args.repeats,
                "mean_seconds": mean_seconds,
                "p95_seconds": percentile_95(durations),
                "mean_ms_per_cue": mean_seconds * 1_000 / size,
                "mean_cues_per_second": size / mean_seconds,
            }
        )

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": runtime.device,
        "model_id": runtime.model_id,
        "model_revision": runtime.model_revision,
        "rss_mb": process.memory_info().rss / (1024 * 1024),
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved benchmark report to {args.output}")


if __name__ == "__main__":
    main()
