"""Rebuild non-neural inference statistics without retraining any model.

This script intentionally stops before exporting if the pinned dataset,
preprocessing result, saved split assignment, classifier labels, or expected
counts differ from the completed Kaggle run.
"""

from __future__ import annotations

import argparse
import html
import importlib.metadata
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer


SEED = 42
DATASET_ID = "Calvin-Xu/Furigana-Aozora"
DATASET_REVISION = "be9abdc9313a16146df74358c482015f2576ef03"
MAX_TARGET_EXAMPLES = 260_000
MAX_TARGETS_PER_ROW = 12
MAX_LENGTH = 192
MAX_READING_CLASSES = 3_072
MIN_READING_COUNT = 3
MIN_VARIANT_COUNT = 2
SHUFFLE_BUFFER_SIZE = 50_000
TARGET_FEATURE_DIM = 768 * 4

EXPECTED_COUNTS = {
    "all_targets": 259_476,
    "train": 207_635,
    "validation": 25_914,
    "test": 25_927,
    "train_surfaces": 26_210,
    "contextual_surfaces": 1_004,
    "reading_labels": 1_491,
}

KANJI_RE = re.compile(r"[一-龠々〆ヶ]")
KANA_RE = re.compile(r"^[ぁ-ゖァ-ヺー]+$")
RUBY_RE = re.compile(
    r"<ruby>(.*?)<rt>(.*?)</rt></ruby>",
    re.IGNORECASE | re.DOTALL,
)
TAG_RE = re.compile(r"<[^>]+>")


def clean_html_text(value: str | None) -> str:
    value = html.unescape(TAG_RE.sub("", value or ""))
    return unicodedata.normalize("NFKC", value).strip()


def katakana_to_hiragana(text: str | None) -> str:
    return "".join(
        chr(ord(char) - 0x60) if "ァ" <= char <= "ヶ" else char
        for char in (text or "")
    )


def normalize_reading(text: str | None) -> str:
    normalized = re.sub(r"\s+", "", clean_html_text(text))
    return katakana_to_hiragana(normalized)


def extract_targets(
    row: dict[str, Any],
    max_targets: int = MAX_TARGETS_PER_ROW,
) -> list[dict[str, Any]]:
    sentence = unicodedata.normalize(
        "NFKC",
        str(row.get("input") or ""),
    ).strip()
    ruby_html = str(row.get("output") or "")
    file_path = unicodedata.normalize(
        "NFKC",
        str(row.get("file_path") or "unknown"),
    )
    if not sentence or not ruby_html or len(sentence) > 450:
        return []

    results: list[dict[str, Any]] = []
    search_from = 0
    for match in RUBY_RE.finditer(ruby_html):
        surface = clean_html_text(match.group(1))
        reading = normalize_reading(match.group(2))
        if (
            not surface
            or not KANJI_RE.search(surface)
            or not KANA_RE.fullmatch(reading)
        ):
            continue
        if len(surface) > 12 or len(reading) > 24:
            continue

        start = sentence.find(surface, search_from)
        if start < 0:
            start = sentence.find(surface)
        if start < 0:
            continue

        end = start + len(surface)
        results.append(
            {
                "sentence": sentence,
                "surface": surface,
                "reading": reading,
                "start": start,
                "end": end,
                "file_path": file_path,
            }
        )
        search_from = end
        if len(results) >= max_targets:
            break
    return results


def generate_furigana_html(
    sentence: str,
    max_merge_tokens: int = 4,
) -> str:
    """Signature-only parity hook used by the Phase 0 exporter."""

    del max_merge_tokens
    return sentence


def package_versions() -> dict[str, str]:
    names = ("datasets", "numpy", "pandas", "pyarrow", "transformers")
    return {
        name: importlib.metadata.version(name)
        for name in names
    }


def require_counts(actual: dict[str, int]) -> None:
    mismatches = {
        name: {
            "expected": EXPECTED_COUNTS[name],
            "actual": value,
        }
        for name, value in actual.items()
        if EXPECTED_COUNTS[name] != value
    }
    if mismatches:
        raise RuntimeError(
            "Dừng trước export vì số liệu tái tạo không khớp lần Kaggle gốc:\n"
            + json.dumps(mismatches, ensure_ascii=False, indent=2)
        )


def rebuild(
    *,
    workspace_root: Path,
    model_dir: Path,
    split_assignments_path: Path,
) -> dict[str, Any]:
    print(
        f"Loading {DATASET_ID}@{DATASET_REVISION} "
        f"(streaming, shuffle seed={SEED}, buffer={SHUFFLE_BUFFER_SIZE})"
    )
    stream = load_dataset(
        DATASET_ID,
        split="train",
        streaming=True,
        revision=DATASET_REVISION,
    )
    stream = stream.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER_SIZE)

    targets: list[dict[str, Any]] = []
    rows_seen = 0
    for row in stream:
        rows_seen += 1
        targets.extend(extract_targets(row))
        if rows_seen % 10_000 == 0:
            print(
                f"  source rows={rows_seen:,}, raw targets={len(targets):,}"
            )
        if len(targets) >= MAX_TARGET_EXAMPLES:
            break

    all_df = (
        pd.DataFrame(targets[:MAX_TARGET_EXAMPLES])
        .drop_duplicates(
            subset=["sentence", "surface", "reading", "start"],
        )
        .reset_index(drop=True)
    )
    all_df["example_id"] = np.arange(len(all_df))

    assignments = pd.read_csv(
        split_assignments_path,
        encoding="utf-8-sig",
    )
    if assignments[["file_path", "split"]].isna().any().any():
        raise RuntimeError("split_assignments.csv chứa giá trị rỗng.")
    if assignments.file_path.duplicated().any():
        raise RuntimeError("split_assignments.csv chứa file_path trùng.")
    valid_split_names = {"Train", "Validation", "Test"}
    if set(assignments.split) != valid_split_names:
        raise RuntimeError(
            "Tên split không hợp lệ: "
            f"{sorted(set(assignments.split))!r}."
        )

    split_by_file = dict(zip(assignments.file_path, assignments.split))
    missing_files = sorted(set(all_df.file_path) - set(split_by_file))
    extra_files = sorted(set(split_by_file) - set(all_df.file_path))
    if missing_files or extra_files:
        raise RuntimeError(
            "split_assignments.csv không khớp file_path của dataset pinned. "
            f"missing={missing_files[:5]!r}, extra={extra_files[:5]!r}"
        )

    assigned = all_df.file_path.map(split_by_file)
    train_all = all_df[assigned == "Train"].copy().reset_index(drop=True)
    val_all = all_df[assigned == "Validation"].copy().reset_index(drop=True)
    test_all = all_df[assigned == "Test"].copy().reset_index(drop=True)

    surface_reading_counts: defaultdict[str, Counter[str]] = defaultdict(
        Counter
    )
    for row in train_all.itertuples(index=False):
        surface_reading_counts[row.surface][row.reading] += 1

    surface_mode = {
        surface: counts.most_common(1)[0][0]
        for surface, counts in surface_reading_counts.items()
    }
    ambiguous_surfaces = {
        surface
        for surface, counts in surface_reading_counts.items()
        if sum(
            count >= MIN_VARIANT_COUNT
            for count in counts.values()
        )
        >= 2
    }
    ambiguous_train = train_all[
        train_all.surface.isin(ambiguous_surfaces)
    ].copy()

    reading_counts = Counter(ambiguous_train.reading)
    reading_vocab = [
        reading
        for reading, count in reading_counts.most_common(
            MAX_READING_CLASSES
        )
        if count >= MIN_READING_COUNT
    ]
    label2id = {
        reading: index
        for index, reading in enumerate(reading_vocab)
    }
    id2label = {
        index: reading
        for reading, index in label2id.items()
    }

    config = json.loads(
        (model_dir / "config.json").read_text(encoding="utf-8")
    )
    saved_label2id = {
        str(label): int(index)
        for label, index in config["label2id"].items()
    }
    if label2id != saved_label2id:
        first_difference = next(
            (
                {
                    "reading": reading,
                    "rebuilt": label2id.get(reading),
                    "saved": saved_label2id.get(reading),
                }
                for reading in sorted(set(label2id) | set(saved_label2id))
                if label2id.get(reading) != saved_label2id.get(reading)
            ),
            None,
        )
        raise RuntimeError(
            "Dừng trước export vì label2id tái tạo không khớp "
            f"model/config.json. first_difference={first_difference!r}"
        )

    def valid_candidates(surface: str) -> list[str]:
        counts = surface_reading_counts.get(surface, {})
        return [
            reading
            for reading, count in counts.items()
            if count >= MIN_VARIANT_COUNT and reading in label2id
        ]

    surface_candidates = {
        surface: valid_candidates(surface)
        for surface in ambiguous_surfaces
    }
    surface_candidates = {
        surface: readings
        for surface, readings in surface_candidates.items()
        if len(readings) >= 2
    }
    surface_list = sorted(surface_candidates)
    surface2idx = {
        surface: index
        for index, surface in enumerate(surface_list)
    }
    surface_candidate_ids = {
        surface: [label2id[reading] for reading in readings]
        for surface, readings in surface_candidates.items()
    }

    candidate_mask_matrix = np.zeros(
        (len(surface_list), len(label2id)),
        dtype=np.bool_,
    )
    surface_log_prior_matrix = np.full(
        (len(surface_list), len(label2id)),
        -1e4,
        dtype=np.float32,
    )
    for surface, surface_index in surface2idx.items():
        candidate_ids = surface_candidate_ids[surface]
        candidate_mask_matrix[surface_index, candidate_ids] = True
        counts = surface_reading_counts[surface]
        smoothed = np.asarray(
            [
                counts[id2label[label_id]] + 0.5
                for label_id in candidate_ids
            ],
            dtype=np.float64,
        )
        smoothed /= smoothed.sum()
        surface_log_prior_matrix[
            surface_index,
            candidate_ids,
        ] = np.log(smoothed)

    actual_counts = {
        "all_targets": len(all_df),
        "train": len(train_all),
        "validation": len(val_all),
        "test": len(test_all),
        "train_surfaces": len(surface_reading_counts),
        "contextual_surfaces": len(surface_candidates),
        "reading_labels": len(label2id),
    }
    print(json.dumps(actual_counts, ensure_ascii=False, indent=2))
    require_counts(actual_counts)

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    tuning_summary = json.loads(
        (workspace_root / "tuning_summary.json").read_text(
            encoding="utf-8"
        )
    )

    sys.path.insert(0, str(workspace_root))
    from scripts.export_inference_artifacts import export_from_namespace

    namespace = {
        "MODEL_DIR": model_dir,
        "MODEL_NAME": "tohoku-nlp/bert-base-japanese-v3",
        "DATASET_NAME": DATASET_ID,
        "DATASET_REVISION": DATASET_REVISION,
        "MAX_LENGTH": MAX_LENGTH,
        "TARGET_FEATURE_DIM": TARGET_FEATURE_DIM,
        "surface_list": surface_list,
        "surface2idx": surface2idx,
        "surface_candidate_ids": surface_candidate_ids,
        "candidate_mask_matrix": candidate_mask_matrix,
        "surface_log_prior_matrix": surface_log_prior_matrix,
        "surface_mode": surface_mode,
        "label2id": label2id,
        "id2label": id2label,
        "tokenizer": tokenizer,
        "tuning_summary": tuning_summary,
        "clean_html_text": clean_html_text,
        "katakana_to_hiragana": katakana_to_hiragana,
        "normalize_reading": normalize_reading,
        "generate_furigana_html": generate_furigana_html,
    }
    export_summary = export_from_namespace(
        namespace,
        model_dir,
        model_id="local-furigana-aid-bert-mlp",
        model_revision=None,
        max_merge_tokens=4,
    )

    recovery_manifest = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "streaming": True,
        "shuffle_seed": SEED,
        "shuffle_buffer_size": SHUFFLE_BUFFER_SIZE,
        "source_rows_seen": rows_seen,
        "split_assignments": str(split_assignments_path.resolve()),
        "counts": actual_counts,
        "library_versions": package_versions(),
        "export_summary": export_summary,
    }
    recovery_path = model_dir / "artifact_recovery_manifest.json"
    recovery_path.write_text(
        json.dumps(
            recovery_manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return recovery_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parents[2]
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=workspace_root,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=workspace_root / "model",
    )
    parser.add_argument(
        "--split-assignments",
        type=Path,
        default=workspace_root / "split_assignments.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = rebuild(
        workspace_root=args.workspace_root.resolve(),
        model_dir=args.model_dir.resolve(),
        split_assignments_path=args.split_assignments.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
