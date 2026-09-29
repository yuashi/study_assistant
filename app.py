import hashlib
import os
import shutil
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = "vector_store"

st.set_page_config(page_title="RAG Study Assistant", page_icon="📚")

# ---------------------------------------------------------------------------
# Cached, stateless resources (created once, reused across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-mpnet-base-v2"
    )


@st.cache_resource
def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b")


PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are a helpful AI assistant.
     Use only the provided context to answer the question.
     If the answer is not present in the context,
     say: "I could not find the answer in the document." and do not make up an answer.
     """),
    ("human",
     """Context:
     {context}

     Question:
     {question}
     """),
])


# ---------------------------------------------------------------------------
# Vector store lifecycle
# ---------------------------------------------------------------------------
def delete_current_db():
    """Drop the Chroma collection and remove its folder from disk."""
    vs = st.session_state.get("vectorstore")
    if vs is not None:
        try:
            vs.delete_collection()
        except Exception:
            pass
    st.session_state.vectorstore = None

    db_dir = st.session_state.get("db_dir")
    if db_dir and os.path.exists(db_dir):
        shutil.rmtree(db_dir, ignore_errors=True)
    st.session_state.db_dir = None


def build_db(pdf_bytes: bytes):
    """Create a brand-new Chroma DB from the uploaded PDF. Returns (vectorstore, dir, n_chunks)."""
    # PyPDFLoader needs a file path, so write the upload to a temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        docs = PyPDFLoader(tmp_path).load()
    finally:
        os.remove(tmp_path)

    if not any(d.page_content.strip() for d in docs):
        raise ValueError(
            "No text could be extracted from this PDF (it may be a scanned document)."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Unique folder per upload avoids Chroma's cached-client / file-lock problems
    # that can happen when deleting and re-creating the same directory.
    db_dir = os.path.join(BASE_DIR, f"chroma_{uuid.uuid4().hex[:8]}")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        persist_directory=db_dir,
    )
    return vectorstore, db_dir, len(chunks)


# ---------------------------------------------------------------------------
# Session state init (+ clean up leftovers from previous runs / crashed sessions)
# ---------------------------------------------------------------------------
if "initialized" not in st.session_state:
    shutil.rmtree(BASE_DIR, ignore_errors=True)
    os.makedirs(BASE_DIR, exist_ok=True)
    st.session_state.initialized = True
    st.session_state.vectorstore = None
    st.session_state.db_dir = None
    st.session_state.file_hash = None
    st.session_state.file_name = None
    st.session_state.n_chunks = 0
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Sidebar: upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Document")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded is not None:
        pdf_bytes = uploaded.getvalue()
        file_hash = hashlib.md5(pdf_bytes).hexdigest()

        # Only rebuild when a *different* PDF is uploaded (not on every rerun)
        if file_hash != st.session_state.file_hash:
            delete_current_db()  # remove the previous DB first
            st.session_state.messages = []  # new doc -> fresh chat

            with st.spinner("Reading and indexing your PDF..."):
                try:
                    vs, db_dir, n_chunks = build_db(pdf_bytes)
                except Exception as e:
                    st.session_state.file_hash = None
                    st.session_state.file_name = None
                    st.error(f"Could not process the PDF: {e}")
                else:
                    st.session_state.vectorstore = vs
                    st.session_state.db_dir = db_dir
                    st.session_state.file_hash = file_hash
                    st.session_state.file_name = uploaded.name
                    st.session_state.n_chunks = n_chunks

    elif st.session_state.file_hash is not None:
        # User clicked the "x" on the uploader -> clean up
        delete_current_db()
        st.session_state.file_hash = None
        st.session_state.file_name = None
        st.session_state.messages = []

    if st.session_state.vectorstore is not None:
        st.success(f"Ready: {st.session_state.file_name}")
        st.caption(f"{st.session_state.n_chunks} chunks indexed")

# ---------------------------------------------------------------------------
# Main: chat
# ---------------------------------------------------------------------------
st.title("📚 RAG Study Assistant")

if st.session_state.vectorstore is None:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question about your document")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retriever = st.session_state.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
            )
            docs = retriever.invoke(query)
            context = "\n\n".join(doc.page_content for doc in docs)

            prompt = PROMPT.invoke({"context": context, "question": query})
            answer = get_llm().invoke(prompt).content

        st.markdown(answer)

        with st.expander("Sources"):
            for i, doc in enumerate(docs, 1):
                page = doc.metadata.get("page", "?")
                st.markdown(f"**Chunk {i}** (page {page + 1 if isinstance(page, int) else page})")
                st.caption(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})