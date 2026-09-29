from langchain_community.document_loaders import PyPDFLoader

data=PyPDFLoader("document_loaders/report.pdf")
pdf_docs=data.load()
