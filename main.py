from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

embedding_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-mpnet-base-v2"
)

vectorstore=Chroma(
    persist_directory="vector_store/chroma_db",
    embedding_function=embedding_model
)

retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3,"fetch_k": 10,"lambda_mult": 0.5})

llm=ChatGroq(model="openai/gpt-oss-120b")

template=ChatPromptTemplate.from_messages([
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
    """)
])

print("===========================RAG study assistant=========================")
print("press 0 to exit")

while True:
    query=input("You : ")
    if query == '0':
        break

    docs=retriever.invoke(query)
    context="\n\n".join([doc.page_content for doc in docs])
    prompt=template.invoke({"context": context, "question": query})

    response=llm.invoke(prompt)
    print("\nAI : ", response.content)



