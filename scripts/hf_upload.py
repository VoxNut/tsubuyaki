import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, get_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload model artifacts to a Hugging Face model repository."
    )
    parser.add_argument(
        "--model-dir",
        "--model_dir",
        dest="model_dir",
        default=os.getenv("FURIGANA_MODEL_LOCAL_DIR"),
        help="Local model directory (or set FURIGANA_MODEL_LOCAL_DIR).",
    )
    parser.add_argument(
        "--repo-id",
        "--repo_id",
        dest="repo_id",
        required=True,
        help="Hub repository ID, for example username/furigana-aid-model.",
    )
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create the repository as private (default: true).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.getenv("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit(
            "Missing Hugging Face credentials. Run `hf auth login` or set "
            "HF_TOKEN in the environment."
        )
    if not args.model_dir:
        raise SystemExit(
            "Missing model directory. Pass --model-dir or set "
            "FURIGANA_MODEL_LOCAL_DIR."
        )

    model_path = Path(args.model_dir).expanduser().resolve()
    if not model_path.is_dir():
        raise SystemExit(f"Model directory does not exist: {model_path}")

    api = HfApi(token=token)
    print(f"Creating or checking model repository {args.repo_id!r}...")
    api.create_repo(
        repo_id=args.repo_id,
        private=args.private,
        repo_type="model",
        exist_ok=True,
    )

    print(f"Uploading model artifacts from {model_path}...")
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload Furigana Aid inference artifacts",
    )
    revision = api.model_info(args.repo_id).sha
    if not revision:
        raise SystemExit("Upload completed but the Hub did not return a commit SHA.")

    print("Upload completed successfully.")
    print("Configure the backend with:")
    print(f'  FURIGANA_HF_MODEL_REPO="{args.repo_id}"')
    print(f'  FURIGANA_HF_MODEL_REVISION="{revision}"')
    print("Use a separate read-only token as the deployed Space HF_TOKEN secret.")


if __name__ == "__main__":
    main()
