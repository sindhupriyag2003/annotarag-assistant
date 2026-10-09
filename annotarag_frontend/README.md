# AnnotaRAG — Annotation Guideline Intelligence

A polished Streamlit frontend for a Retrieval-Augmented Generation (RAG) assistant for AI/ML annotation guidelines.

## Features
- Modern dashboard-style frontend
- Ingest public webpage guidelines
- Text extraction and chunking
- Sentence Transformer embeddings
- Persistent ChromaDB semantic search
- Retrieved evidence with original source links
- Optional answer generation through Groq API
- Without an API key, the app still retrieves and displays source evidence

## Run locally
Install Python 3.10–3.12, then:

```bash
python -m venv .venv
# Windows CMD:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local address printed by Streamlit, usually http://localhost:8501.

The embedding model downloads on first launch. Add public webpage URLs in the sidebar. For generated summaries, create a Groq API key at https://console.groq.com/keys and paste it into the optional key field. Do not publish API keys.

## Publish online
1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt` (and optionally this README).
3. Visit https://share.streamlit.io and sign in with GitHub.
4. Create an app using your repository and `app.py`.
5. Deploy; Streamlit will provide a `*.streamlit.app` URL when it finishes.

The app uses Groq's hosted API for generation when a key is provided. A local Ollama process on your laptop is not accessible from a cloud-hosted Streamlit app. Hosted filesystems may reset, so use managed persistent storage for production.

## Security and limitations
This is a learning prototype, not a production-secure web crawler. It blocks non-public IPs and redirects but still needs additional hardening before public production deployment. Only ingest webpages you are authorized to access. Verify answers against the original annotation guidelines.
