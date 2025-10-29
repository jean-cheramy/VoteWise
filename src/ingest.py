"""Document ingestion and embedding indexing using Azure AI Foundry models + AI Search."""
import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchField,
    SearchFieldDataType,
    VectorSearch
)
from azure.search.documents import SearchClient
from openai import AzureOpenAI

from chunker import Chunker

load_dotenv()


def create_index_if_not_exists() -> None:
    """Create a vector index in Azure Cognitive Search if it doesn't exist."""
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    key = os.getenv("AZURE_SEARCH_API_KEY")
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")

    index_client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    existing = [idx.name for idx in index_client.list_indexes()]
    if index_name in existing:
        print("Index already exists.")
        return

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="content", type=SearchFieldDataType.String, searchable=True),
        SimpleField(name="party", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="language", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="default",
        ),
    ]

    vector_search = VectorSearch(
        profiles=[{"name": "default", "algorithm": "hnsw"}],
        algorithms=[{"name": "hnsw", "kind": "hnsw"}],
    )

    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    index_client.create_index(index)
    print("Vector index created successfully.")


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate an embedding from Azure AI Foundry's embedding model."""
    client = AzureOpenAI(
        api_version=os.getenv("AZURE_FOUNDRY_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_FOUNDRY_ENDPOINT"),
        api_key=os.getenv("AZURE_FOUNDRY_API_KEY")
    )
    deployment = os.getenv("AZURE_FOUNDRY_EMBED_MODEL")

    response = client.embeddings.create(input=texts, model=deployment)
    embeddings = [e.embedding for e in response.data]
    return embeddings


def index_documents(data_dir: Path) -> None:
    """Embed and index political party documents into Azure AI Search."""
    search_client = SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY")),
    )

    chunker = Chunker(base_dir=data_dir, chunk_size=1000, chunk_overlap=200)
    all_chunks = chunker.process_all_parties()
    print("Documents are chunked.")

    BATCH_SIZE = 10
    batch_chunks = []

    for i, chunk_data in enumerate(all_chunks, start=1):
        batch_chunks.append(chunk_data)

        if i % BATCH_SIZE == 0 or i == len(all_chunks):
            texts = [c["chunk"] for c in batch_chunks]
            embeddings = get_embeddings(texts)

            docs_to_upload = []
            for c, emb in zip(batch_chunks, embeddings):
                docs_to_upload.append({
                    "id": str(uuid.uuid4()),
                    "content": c["chunk"],
                    "party": c["party"],
                    "language": c["language"],
                    "source": c["source"],
                    "embedding": emb
                })

            result = search_client.upload_documents(documents=docs_to_upload)
            print(f"Indexed batch of {len(docs_to_upload)}")

            batch_chunks = []


if __name__ == "__main__":
    create_index_if_not_exists()
    BASE_DIR: Path = Path(__file__).resolve().parent
    RAW_DIR: Path = BASE_DIR / "data" / "fr"
    index_documents(RAW_DIR)
