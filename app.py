import streamlit as st
from multi_modal_ingest import multi_modal_ingest, normalize_element
from vector_store import build_faiss, load_faiss
from qa_engine import answer_question
import hashlib
import os

st.set_page_config(page_title="📄 Multi-Modal RAG Chatbot", layout="wide")

# ----------------------------
# SESSION STATE STRUCTURE
# ----------------------------
if "faiss_store" not in st.session_state:
    st.session_state.faiss_store = None

if "ready" not in st.session_state:
    st.session_state.ready = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None


@st.cache_resource(show_spinner=False)
def build_cached_store(file_hash, file_bytes):
    index_root = os.path.join(".cache", "faiss")
    index_path = os.path.join(index_root, file_hash)

    if os.path.exists(index_path):
        return load_faiss(index_path)

    elements = multi_modal_ingest(file_bytes)

    chunks = []
    for el in elements:
        ch = normalize_element(el, file_bytes)
        if ch:
            chunks.append(ch)

    if not chunks:
        raise ValueError("No usable content was extracted from the PDF.")

    os.makedirs(index_root, exist_ok=True)
    return build_faiss(chunks, index_path, persist=True)


# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.title("📂 Document Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF", 
    type=["pdf"]
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    st.sidebar.info(f"📄 File uploaded: {uploaded_file.name}")

    if st.session_state.pdf_hash != file_hash:
        # Process PDF only when the uploaded file changes.
        with st.spinner("Extracting and indexing PDF..."):
            try:
                st.session_state.faiss_store = build_cached_store(file_hash, file_bytes)
            except ValueError:
                st.session_state.faiss_store = None
                st.session_state.ready = False
                st.session_state.pdf_name = None
                st.session_state.pdf_hash = None
                st.error("No usable content was extracted from the PDF.")
            else:
                st.session_state.ready = True
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.pdf_hash = file_hash
                st.sidebar.success("✅ PDF indexed and ready to query!")
    elif st.session_state.ready:
        st.sidebar.success("✅ PDF already indexed and ready to query.")


# ----------------------------
# MAIN UI
# ----------------------------

st.title("🤖 Multi-Modal RAG Chatbot")
st.caption("Upload a PDF and ask questions about it")

if st.session_state.ready:
    st.success(f"Current Document: {st.session_state.pdf_name}")

else:
    st.warning("Upload a PDF to get started")


# User question
query = st.chat_input("Ask a question")


# ----------------------------
# Handle Query
# ----------------------------

if query:

    if not st.session_state.ready:
        st.error("Upload a PDF first!")
    else:
        with st.chat_message("user"):
            st.write(query)

        with st.spinner("Searching document..."):
            answer, citations = answer_question(query, store=st.session_state.faiss_store)

        with st.chat_message("assistant"):
            st.write(answer)

            #if citations:
            #    st.write("### 🔖 Citations")
            #    for c in citations:
            #        st.write("- " + c)
