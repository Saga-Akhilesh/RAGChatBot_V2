import streamlit as st
from utils import load_pdf_text
from rag_engine import (
    chunk_text,
    create_vector_store,
    retrieve_similar_chunks,
    generate_answer
)

st.set_page_config(page_title="📄 RAG Chatbot", layout="wide")
st.title("🤖 RAG Chatbot with Mistral + OCR Support")

# Sidebar settings
st.sidebar.title("⚙️ Settings")
model_name = st.sidebar.selectbox("LLM Model", ["mistral", "mixtral"])
chunk_size = st.sidebar.slider("Chunk Size", 100, 1000, 300, 50)
top_k = st.sidebar.slider("Top-K Results", 1, 10, 3)
use_ocr = st.sidebar.checkbox("Enable OCR for scanned PDFs", value=True)
mode = st.sidebar.radio("Chat Mode", ["PDF Q&A", "Free Chat"])

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# PDF Q&A mode
if mode == "PDF Q&A":
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        try:
            st.info("Extracting and chunking text...")
            raw_text = load_pdf_text("temp.pdf", use_ocr=use_ocr)
            chunks = chunk_text(raw_text, chunk_size=chunk_size)
            index, embeddings, chunk_list = create_vector_store(chunks)
            st.success("✅ Document processed successfully.")

            query = st.text_input("Ask a question based on the PDF:")
            if query:
                matched_chunks = retrieve_similar_chunks(query, index, chunk_list, k=top_k)
                context = "\n\n".join(matched_chunks)
                answer = generate_answer(query, context, model_name=model_name)
                st.markdown(f"**Answer:** {answer}")
                st.session_state.chat_history.append((query, answer))

        except Exception as e:
            st.error(str(e))

# Free Chat mode
else:
    query = st.text_input("Ask a question:")
    if query:
        answer = generate_answer(query, context="", model_name=model_name)
        st.markdown(f"**Answer:** {answer}")
        st.session_state.chat_history.append((query, answer))

# Display chat history
if st.session_state.chat_history:
    with st.expander("🕓 Chat History"):
        for q, a in reversed(st.session_state.chat_history[-10:]):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")
