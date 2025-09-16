import faiss
import pickle
from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from embeddings import embed_texts

BASE_DIR: Path = Path(__file__).resolve().parent.parent
INDEX_PATH: Path = BASE_DIR / "data/index.faiss"
DOCS_PATH: Path = BASE_DIR / "data/processed/chunks.pkl"

# Load FAISS index
index: faiss.Index = faiss.read_index(str(INDEX_PATH))

# Load documents
with open(DOCS_PATH, "rb") as f:
    docs: List[Document] = pickle.load(f)

# Initialize vector store
vector_store: FAISS = FAISS(
    embedding_function=lambda x: embed_texts([x])[0],
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={i: i for i in range(len(docs))}
)

def retrieve(query: str, k: int = 5) -> List[Document]:
    """
    Perform similarity search on the vector store for a given query.

    Args:
        query (str): The user's question or search query.
        k (int, optional): Number of top documents to retrieve. Defaults to 5.

    Returns:
        List[Document]: A list of retrieved document objects.
    """
    results: List[Document] = vector_store.similarity_search(query, k=k)
    return results
