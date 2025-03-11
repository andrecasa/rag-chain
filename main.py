import os
import warnings
import faiss
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from fastapi import FastAPI, Request


# FastAPI
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Chat API"}

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    question = body.get("question")
    if not question:
        return {"error": "Pergunta não foi enviada"}
    output = rag_chain.invoke(question)
    return {"answer": output}


# Ollama RAG
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

warnings.filterwarnings("ignore")

load_dotenv()

loader = PyMuPDFLoader("./rag-dataset/fundepar.pdf")

docs = loader.load()

# pdfs = []
# for root, dirs, files in os.walk('rag-dataset'):
#     for file in files:
#         if file.endswith('.pdf'):
#             pdfs.append(os.path.join(root, file))

# for pdf in pdfs:
#     loader = PyMuPDFLoader(pdf)
#     pages = loader.load()
#     docs.extend(pages)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

chunks = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")

single_vector = embeddings.embed_query("")

index = faiss.IndexFlatL2(len(single_vector))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

ids = vector_store.add_documents(documents=chunks)

vector_store.index_to_docstore_id

db_name = "./faiss_db"

vector_store.save_local(db_name)

new_vector_store = FAISS.load_local(db_name, embeddings=embeddings, allow_dangerous_deserialization=True)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs = {'k': 3, 
                                                                          'fetch_k': 100,
                                                                          'lambda_mult': 1})

model = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434")

prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Answer in bullet points. 
    Make sure your answer is relevant to the question and it is answered from the context only.
    Question: {question} 
    Context: {context} 
    Answer:
"""

prompt = ChatPromptTemplate.from_template(prompt)

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = (
    {"context": retriever|format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)