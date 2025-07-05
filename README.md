# 🤖 RAG Chatbot V2 - PDF + OCR + Mistral (Ollama)

An open-source local chatbot that answers questions from PDFs (even scanned ones) using:

- 🧠 Mistral via Ollama
- 🔍 SentenceTransformers (`MiniLM`)
- 🧾 OCR support via Tesseract + Poppler
- 🖼️ Streamlit UI for chat + file upload

## 🚀 Features

- Ask questions from uploaded PDFs
- Handles scanned image PDFs with OCR
- ChatGPT-like fallback if no PDF is used
- Local-only: No API key or internet required!

## 📦 Setup

```bash
pip install -r requirements.txt
