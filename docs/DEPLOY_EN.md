# Furigana Aid Reader Deployment Guide

This document provides detailed instructions on how to package the backend service, upload model artifacts to a private Hugging Face Hub repository, and deploy the application.

## 1. Prerequisites
- Docker and Docker Compose (if running local containers)
- A Hugging Face account and a Write Access Token
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
   python scripts/hf_upload.py --repo_id "username/furigana-aid-model" --model_dir "/path/to/your/model"
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
     -e FURIGANA_HF_MODEL_REVISION="main" \
     -e HF_TOKEN="your_token_here" \
     furigana-aid-backend
   ```
   *Note: Access `http://localhost:8000/api/health` to verify the server status.*

---

## 4. Deploy to Hugging Face Spaces (Recommended - Free CPU Tier)

Hugging Face Spaces provides a free CPU tier for running custom Docker containers.

1. **Create Space:**
   - Go to Hugging Face Spaces and click **Create a new Space**.
   - Pick a name, select **Docker** as the SDK, and choose the **Blank** template.
   - Choose public or private visibility depending on your preferences.

2. **Configure Secrets:**
   Under **Settings -> Variables and secrets** in your Space, add a new secret:
   - Name: `HF_TOKEN`
   - Value: *Your Hugging Face Token (needs read permission to download private models)*

3. **Configure Environment Variables:**
   Under Settings, add the following variables:
   - `FURIGANA_HF_MODEL_REPO` = `username/furigana-aid-model`
   - `FURIGANA_HF_MODEL_REVISION` = `main`
   - `PORT` = `7860`

4. **Push Code:**
   - Clone the Space's Git repository.
   - Copy all source files from `tsubuyaki` (including `Dockerfile`, `backend/`, `frontend/`) into the Space directory.
   - Commit and push to the Space's `main` branch. The Space will build and boot up automatically.

---

## 5. Deploy Frontend

The Frontend is a static page located inside the `frontend/` folder.

- **Local Development:**
  Open `frontend/index.html` directly in a browser or spin up a simple web server:
  ```bash
  cd frontend
  python -m http.server 8080
  ```
- **Cloud Hosting:**
  Since the frontend consists only of static files (`index.html`, `manifest.json`, `service-worker.js`), you can host it for free on GitHub Pages, Vercel, Netlify, Cloudflare Pages, etc.
- **Connection Configuration:**
  Once opened, click the **Settings (⚙)** icon on the player UI, enter your backend API URL (e.g., `https://username-space-name.hf.space`) in the **API Server Endpoint** box, and click **Tạo Furigana** to process subtitles!
