from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

splitter=TokenTextSplitter(chunk_size=1000, chunk_overlap=10)

data=PyPDFLoader("cim.pdf")
docs=data.load()

chunks=splitter.split_documents(docs)