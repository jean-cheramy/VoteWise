from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings 

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "faiss"

embeddings = OllamaEmbeddings(model="llama3")

def load_vector_store():
    return FAISS.load_local(str(DATA_DIR), embeddings, allow_dangerous_deserialization=True)
