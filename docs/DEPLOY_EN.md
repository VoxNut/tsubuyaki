# tsubuyaki Deployment Guide

This document provides detailed instructions on how to package the backend service, upload model artifacts to a private Hugging Face Hub repository, and deploy the application.

## 1. Prerequisites
- Docker and Docker Compose (if running local containers)
- A Hugging Face account; use a local write token for uploads and a separate
  read-only token for the Space
- Python 3.12+ (for running the upload script)

---

## 2. Upload Model Artifacts to Hugging Face Hub (Private Repo)

To allow the backend to automatically retrieve model files at runtime while keeping your self-trained model secure, push the model folder to a **Private** Hugging Face repository:

1. **Prepare Access Token:**
   Get a Hugging Face Write Token from your settings (Settings -> Access Tokens).
   Set the local environment variable:
   ```bash
   # Windows (cmd)
   set HF_TOKEN=your_token_here
   # Linux/macOS
   export HF_TOKEN=your_token_here
   ```

2. **Run Upload Script:**
   Execute `scripts/hf_upload.py` to create a private repository and push the local `model/` folder contents:
   ```bash
   python scripts/hf_upload.py --repo-id "username/furigana-aid-model" --model-dir "/path/to/your/model"
   ```
   *Note: Replace `username/furigana-aid-model` with your Hugging Face username and desired repository name.*

---

## 3. Local Execution with Docker

You can test the backend container locally before deploying to the cloud:

1. **Build Docker Image:**
   Run the build command from the repository root (`tsubuyaki/`):
   ```bash
   docker build -t furigana-aid-backend .
   ```

2. **Run Container:**
   ```bash
   docker run -d -p 8000:7860 \
     -e FURIGANA_MODEL_LOCAL_DIR="" \
     -e FURIGANA_HF_MODEL_REPO="username/furigana-aid-model" \
     -e FURIGANA_HF_MODEL_REVISION="<model-commit-sha>" \
     -e HF_TOKEN="your_token_here" \
     furigana-aid-backend
   ```
   *Note: Access `http://localhost:8000/api/health` to verify the server status.*

---

## 4. Deploy the frontend and backend to Hugging Face Spaces

The project Docker image serves both the web UI and FastAPI API on port `7860`.
This is the recommended deployment because users visit one URL and the browser
does not cross an origin boundary between the frontend and backend.

1. **Create Space:**
   - Go to Hugging Face Spaces and click **Create a new Space**.
   - Pick a name, select **Docker** as the SDK, and choose the **Blank** template.
   - Choose **Public** to make both the app and source code publicly accessible.
   - Hugging Face PRO users can choose **Protected** to keep source code private
     while the running application remains public.

2. **Configure Secrets:**
   Under **Settings -> Variables and secrets** in your Space, add a new secret:
   - Name: `HF_TOKEN`
   - Value: *A dedicated read-only token allowed to download the private model.*

   Do not use a personal write token as a long-lived runtime secret, and never
   place a token in the README, source code, or Docker image.

3. **Configure Environment Variables:**
   Under Settings, add the following variables:
   - `FURIGANA_HF_MODEL_REPO` = `username/furigana-aid-model`
   - `FURIGANA_HF_MODEL_REVISION` = the immutable commit SHA printed by the upload script
   - `FURIGANA_DEVICE` = `cpu`
   - `FURIGANA_INFERENCE_BATCH_SIZE` = `8`
   - `PORT` = `7860`

   Do not use the model's `main` branch as a production revision because it can
   change between container starts.

4. **Push Code:**
   - Clone the Space's Git repository.
   - Copy all source files from `tsubuyaki` (including `Dockerfile`, `backend/`, `frontend/`) into the Space directory.
   - Commit and push to the Space's `main` branch. The Space will build and boot up automatically.

5. **Verify the deployment:**
   - `https://<space-subdomain>.hf.space/api/health` must return `{"status":"ok"}`.
   - `https://<space-subdomain>.hf.space/api/ready` must return `"ready": true`.
   - Open `https://<space-subdomain>.hf.space/` to use the web application.

---

## 5. When should the frontend be deployed separately?

The frontend does not need a separate deployment by default. FastAPI mounts
`frontend/` at `/`, while the API is available at `/api/` on the same domain.

- **Separate local frontend development:**
  ```bash
  cd frontend
  python -m http.server 8080
  ```
- When hosting the frontend on GitHub Pages, Vercel, Netlify, or Cloudflare
  Pages, enter the backend URL under **Settings -> API Server Endpoint** and add
  the frontend domain to the backend's `FURIGANA_CORS_ORIGINS` variable.
- With the default Docker Space, leave **API Server Endpoint** blank so the app
  calls `/api` on the same origin, then select **Generate furigana**.
