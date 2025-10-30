"""FastAPI RAG API using Azure AI Foundry models and Azure AI Search."""
import os
import json
from fastapi import FastAPI, Query
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI
from dotenv import load_dotenv 
load_dotenv() 

def search_similar(query: dict) -> list:
    """Retrieve relevant documents from Azure AI Search via vector search."""
    search_client = SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY")),
    )

    client = AzureOpenAI(
        api_version=os.getenv("AZURE_FOUNDRY_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_FOUNDRY_ENDPOINT"),
        api_key=os.getenv("AZURE_FOUNDRY_API_KEY")
    )

    embed_model = os.getenv("AZURE_FOUNDRY_EMBED_MODEL")
    embedding = client.embeddings.create(input=[query["question"]], model=embed_model).data[0].embedding
    
    ALLOWED_PARTIES = {"ps", "ecolo", "engages", "mr", "defi", "ptb"}
    filter_query = f"party eq '{query['party']}'" if query["party"] in ALLOWED_PARTIES else None
    
    results = search_client.search(
        vector_queries=[
            {
                "kind": "vector",
                "fields": "embedding",
                "vector": embedding,
                "k": 5
            }
            ],
            filter=filter_query,
            select=["content", "party", "language", "source"]
    )

    results_list = list(results)
    return [dict(doc) for doc in results_list]


app = FastAPI(
    title="VoteWise RAG API",
    docs_url="/docs")

@app.get("/")
def root():
    return {"message": "VoteWise RAG API is running. Use /rag endpoint to query or /docs to interact and test the RAG."}

@app.get("/rag")
def rag(query_s: str = Query(..., description="User query as JSON string", example='{"question": "Quels sont les points clés du programme de l\'Ecolo pour l\'environnement ?", "party": "ecolo"}')):
    query = json.loads(query_s)
    
    chunks = search_similar(query)
    
    if not chunks:
        return {"answer": "No relevant documents found.", "context": ""}

    llm_context = "\n".join([doc["content"] for doc in chunks])

    display_context = "\n------------------------------------\n".join(
        [f"{doc['content']}\n(Source: {doc['source']})" for doc in chunks]
    )

    client = AzureOpenAI(
        api_version=os.getenv("AZURE_FOUNDRY_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_FOUNDRY_ENDPOINT"),
        api_key=os.getenv("AZURE_FOUNDRY_API_KEY")
    )

    llm_model = os.getenv("AZURE_FOUNDRY_LLM_DEPLOYMENT")
    prompt = (
        "Tu es un expert en politique belge, ton rôle est de fournir des explications précises et "
        "vulgarisées sur les programmes des partis politiques belges. "
        f"Utilise le contexte ci-dessous pour répondre à la question.\n\n"
        f"Contexte:\n{llm_context}\n\nQuestion: {query['question']}"
    )

    response = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "answer": response.choices[0].message.content,
        "context": display_context
    }
