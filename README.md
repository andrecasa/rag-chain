# rag-chain

python -m venv .venv
Windows
.venv\Scripts\activate
Linux
source .venv/bin/activate

pip install -U langchain-community faiss-cpu pymupdf tiktoken langchain-ollama python-dotenv "fastapi[standard]"

.env
LANGCHAIN_API_KEY=""
LANGCHAIN_PROJECT = ""
LANGCHAIN_ENDPOINT = ""
LANGCHAIN_TRACING_V2=true

python main.py 

fastapi dev main.py
fastapi run main.py

