# 🤖 RAG Chatbot V2 - PDF + OCR + Mistral (Ollama)

An open-source local chatbot that answers questions from PDFs (even scanned ones) and images (PNG, JPG, JPEG) using:

- 🧠 Mistral, Mixtral via Ollama
- 🔍 SentenceTransformers (`MiniLM`)
- 🧾 OCR support via Tesseract + Poppler
- 🖼️ Streamlit UI for chat + file upload

## 🚀 Features

- Ask questions from uploaded PDFs or images (PNG, JPG, JPEG)
- Handles scanned image PDFs and images with OCR
- ChatGPT-like fallback if no file is used
- Local-only: No API key or internet required!

## 📦 Setup

```bash
pip install -r requirements.txt

### Additional dependencies for image OCR

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (required for both PDF and image OCR)
- [Poppler](https://blog.alivate.com.au/poppler-windows/) (required for PDF OCR)

Make sure Tesseract and Poppler executables are in your system PATH.
