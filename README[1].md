# Free PDF RAG Lab

A beginner-friendly Retrieval-Augmented Generation (RAG) lab. Upload a PDF, retrieve relevant text passages, and ask questions using a free local open-source language model.

## Files

- `app.py` — Streamlit web front end for GitHub + Streamlit Community Cloud.
- `rag_colab.py` — Google Colab notebook-style Python script for the lab.
- `requirements.txt` — Python dependencies.

## No paid API required

This project does not use OpenAI API or another paid inference API. It uses:
- `google/flan-t5-small` for local text generation.
- TF-IDF from scikit-learn for document retrieval.

**Important:** There is no guarantee of unlimited tokens or unlimited compute on free services. Running the model locally avoids per-token API charges, but hosting services have memory, CPU, time, and usage limits. The first run downloads the model and can be slow.

## Run in Google Colab

1. Open https://colab.research.google.com/
2. Create a new notebook.
3. Upload `rag_colab.py` to the notebook environment, or copy its cells into the notebook.
4. Run the script. It installs dependencies, prompts for a PDF upload, indexes it, then asks a question.
5. Ask additional questions by calling `ask_rag("your question")`.

## Run Streamlit locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publish to GitHub

1. Create a new repository on https://github.com/new
2. Upload `app.py`, `requirements.txt`, `README.md`, and optionally `rag_colab.py`.
3. Keep the repository root arranged like this:

```text
free-rag-lab/
├── app.py
├── rag_colab.py
├── requirements.txt
└── README.md
```

Do not upload private PDFs or API keys.

## Deploy on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io/ using GitHub.
2. Select **Create app**.
3. Select your repository, branch (usually `main`), and main file path `app.py`.
4. Click **Deploy** and wait for dependencies and the model to load.
5. Open the generated `*.streamlit.app` URL and upload a PDF.

See the official guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Notes and limitations

- The app indexes the uploaded PDF in memory; it does not fine-tune or train the model.
- Text-based PDFs are supported. Scanned PDFs require OCR first.
- This is a teaching prototype, not a production service. Large PDFs may exceed free hosting memory/time limits.
- Answers can be wrong. Check the retrieved passages shown in the app.
