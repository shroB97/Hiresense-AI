# HireSense AI

HireSense AI is an evidence-first Streamlit application that compares a resume with a job description. It breaks the job description into individual requirements, retrieves supporting resume evidence, and produces a transparent match report.

## Features

- Upload PDF and DOCX resumes
- Discover relevant job-search links
- Analyze required and preferred job requirements
- Show direct, related, uncertain, and missing matches
- Display supporting resume evidence for every assessment
- Generate resume-improvement recommendations
- Download an improved resume as DOCX or PDF
- Track applications during the current session
- Run with an optional OpenAI API key or use the deterministic matcher without one

## Privacy

Uploaded resumes are processed in the active Streamlit session and are not deliberately written to local storage. In deterministic mode, matching runs locally. When AI mode is enabled, the job description and selected resume excerpts are sent to the configured OpenAI API.

Do not commit API keys, `.env`, or `.streamlit/secrets.toml` files to GitHub.

## Run locally

Use Python 3.12 or another supported Python version.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Optional AI configuration

The application works without an API key using its deterministic evidence matcher. To enable structured AI assessment locally, create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your-api-key"
OPENAI_MODEL = "gpt-4.1-mini"
```

For Streamlit Community Cloud, add the same values through the app's Secrets settings instead of committing the file.

## Deploy on Streamlit Community Cloud

1. Fork or upload this repository to GitHub.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/).
3. Create an app from the repository.
4. Select `main` as the branch and `streamlit_app.py` as the entrypoint.
5. Choose Python 3.12 in Advanced settings.
6. Add optional secrets and deploy.

The deployed application will receive a public `streamlit.app` URL.
