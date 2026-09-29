from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

data=PyPDFLoader("document_loaders/report.pdf")
docs=data.load()

splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks=splitter.split_documents(docs)



template=ChatPromptTemplate.from_messages([
    ("system", "You are a helpful study assistant. Summarize the following document and answer any questions about it."),
    ("human", "{data}")
])

model=ChatGroq(model="openai/gpt-oss-120b")

prompt=template.format_messages(data=chunks[0].page_content)

result=model.invoke(prompt)

print(result.content)