import streamlit as st
from utils import load_pdf_text
from rag_engine import chunk_text, create_vector_store, retrieve_similar_chunks, generate_answer

st.set_page_config(page_title="📄 RAG Chatbot + Free Chat", layout="wide")
st.title("📚 RAG Chatbot using Ollama + Mistral")

mode = st.sidebar.radio("Choose mode:", ["Ask from PDF", "Free Chat"])

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"]) if mode == "Ask from PDF" else None

if mode == "Ask from PDF" and uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("🔍 Extracting and chunking text...")
    raw_text = load_pdf_text("temp.pdf")
    chunks = chunk_text(raw_text)

    if not chunks:
        st.error("❌ No extractable text found in PDF. Try a different file.")
        st.stop()

    try:
        index, embeddings, chunk_list = create_vector_store(chunks)
        st.success("✅ Document processed.")
    except ValueError as e:
        st.error(str(e))
        st.stop()

query = st.text_input("💬 Ask your question:")

if query:
    if mode == "Ask from PDF":
        relevant_chunks = retrieve_similar_chunks(query, index, chunk_list)
        context = "\n".join(relevant_chunks)
    else:
        context = ""

    with st.spinner("🤖 Generating answer..."):
        answer = generate_answer(query, context)

    st.markdown("### 💡 Answer")
    st.write(answer)
