import faiss
import numpy as np
import subprocess
from sentence_transformers import SentenceTransformer

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L12-v2')  # Good semantic quality

def chunk_text(text, chunk_size=300, overlap=50):
    text = text.strip()
    if not text:
        return []

    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i+chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def create_vector_store(chunks):
    if not chunks:
        raise ValueError("❌ No text chunks were found. Please upload a valid PDF with extractable text.")

    embeddings = embed_model.encode(chunks)

    if len(embeddings) == 0 or len(embeddings[0]) == 0:
        raise ValueError("❌ Embeddings could not be generated. Please check the document content.")

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return index, embeddings, chunks

def retrieve_similar_chunks(question, index, chunks, k=3):
    q_embed = embed_model.encode([question])
    D, I = index.search(np.array(q_embed), k)
    return [chunks[i] for i in I[0]]

def generate_answer(question, context, model_name="mistral"):
    if context:
        prompt = f"Answer the question using only the following context:\n\n{context}\n\nQ: {question}\nA:"
    else:
        prompt = f"Q: {question}\nA:"

    try:
        # Use Ollama to call local LLM
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt,
            text=True,
            capture_output=True,
            encoding="utf-8"  # ✅ Fixes UnicodeDecodeError
        )
        return result.stdout.strip()
    except Exception as e:
        return f"❌ Ollama error: {str(e)}"
