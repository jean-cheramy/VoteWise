from pathlib import Path
from typing import List
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from embeddings import embed_texts, embeddings

BASE_DIR: Path = Path(__file__).resolve().parent.parent
RAW_DIR: Path = BASE_DIR / "data" / "raw" / "fr"
FAISS_DIR: Path = BASE_DIR / "data" / "faiss"
INDEX_PATH: Path = FAISS_DIR / "index.faiss"

FAISS_DIR.mkdir(parents=True, exist_ok=True)

def pdf_to_text(pdf_path: Path) -> str:
    """
    Extract text from a PDF file.

    Args:
        pdf_path (Path): Path to the PDF file.

    Returns:
        str: Extracted and cleaned text.
    """
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages).replace("\n", " ").strip()

def build_index() -> None:
    """
    Build a FAISS vector store from PDFs in the raw data directory, 
    split text into chunks, compute embeddings, and save the index locally.
    """
    # Load PDFs
    docs_texts: List[str] = [pdf_to_text(f) for f in RAW_DIR.glob("*.pdf")]
    print(f"{len(docs_texts)} PDFs loaded.")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents(docs_texts)
    print(f"{len(chunks)} chunks created.")

    # Compute embeddings
    embeddings_list = embed_texts([chunk.page_content for chunk in chunks])
    print("Embeddings created.")

    # Build FAISS index
    dim: int = embeddings_list.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_list)
    print("FAISS index created.")

    # Wrap in LangChain FAISS
    vector_store = FAISS(
        embedding_function=embeddings.embed_query,
        index=index,
        docstore=InMemoryDocstore({i: doc for i, doc in enumerate(chunks)}),
        index_to_docstore_id={i: i for i in range(len(chunks))}
    )
    print("Vector store created.")

    # Persist FAISS
    vector_store.save_local(str(FAISS_DIR))
    print(f"Index saved at {FAISS_DIR}.")

if __name__ == "__main__":
    build_index()
