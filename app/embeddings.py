from typing import Iterable
import numpy as np
from tqdm import tqdm
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="llama3")

def embed_texts(texts: Iterable[str]) -> np.ndarray:
    """
    Embed a list of texts into a numpy array for FAISS with a progress bar.
    """
    embs = []
    for text in tqdm(texts, desc="Embedding texts"):
        embs.append(embeddings.embed_query(text))
    return np.array(embs, dtype="float32")
