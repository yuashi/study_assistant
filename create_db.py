from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

from document_loaders.page import url_docs
from document_loaders.pdf import pdf_docs

load_dotenv()

docs=pdf_docs

splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks=splitter.split_documents(docs)

embedding_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-mpnet-base-v2"
)

vector_store=Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="vector_store/chroma_db"
)

