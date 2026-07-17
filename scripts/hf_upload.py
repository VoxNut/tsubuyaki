import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def main():
    parser = argparse.ArgumentParser(description="Upload model artifacts to Hugging Face Hub (Private Repo)")
    parser.add_argument("--model_dir", type=str, default="D:\\My WorkSpace\\Machine Learning\\Furigana Aid Generation\\model", help="Path to local model folder")
    parser.add_argument("--repo_id", type=str, required=True, help="HF Repo ID (e.g. username/repo-name)")
    parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN"), help="Hugging Face API token")
    parser.add_argument("--private", type=str, default="true", help="Whether to make the repo private ('true' or 'false', default: 'true')")
    
    args = parser.parse_args()
    
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("Error: Missing Hugging Face token. Please set HF_TOKEN environment variable or pass --token.")
        return
        
    model_path = Path(args.model_dir)
    if not model_path.exists():
        print(f"Error: Model directory '{args.model_dir}' does not exist.")
        return
        
    is_private = args.private.lower() == "true"
    api = HfApi()
    
    print(f"Creating/Checking repository '{args.repo_id}' on Hugging Face Hub...")
    try:
        create_repo(
            repo_id=args.repo_id,
            token=token,
            private=is_private,
            repo_type="model",
            exist_ok=True
        )
        print("Repository is ready.")
    except Exception as e:
        print(f"Warning when creating repository: {e}. Proceeding with upload...")
        
    print(f"Uploading all files from '{args.model_dir}' to repository '{args.repo_id}'...")
    try:
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=args.repo_id,
            repo_type="model",
            token=token
        )
        print("\nUpload completed successfully!")
        print(f"You can configure your backend application with:")
        print(f"  - FURIGANA_HF_MODEL_REPO=\"{args.repo_id}\"")
        print(f"  - FURIGANA_HF_MODEL_REVISION=\"main\"")
    except Exception as e:
        print(f"Error uploading folder: {e}")

if __name__ == "__main__":
    main()
