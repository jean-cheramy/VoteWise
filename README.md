# VoteWise — Belgian Political Program Comparator

## Overview

VoteWise is a Retrieval-Augmented Generation (RAG) system designed to help users explore, summarize, and compare political party positions in Belgium. It combines **Azure AI Search** for document indexing and vector search with **Azure AI Foundry** for embeddings and language model reasoning. Users can ask natural-language questions about party programs, and the system retrieves the most relevant information before generating precise, context-aware answers.

## Architecture

### Key Components

#### Document Storage & Indexing

* Documents are preprocessed, split into chunks, and indexed in **Azure Cognitive Search** with vector embeddings.
* **Vector search** allows efficient retrieval of the most relevant document chunks for a given query.

#### Embedding & Language Model

* **Azure AI Foundry embeddings** are used to represent each document chunk in vector space.
* **Azure AI Foundry LLMs** generate context-aware summaries and answers.
* Filtering by party ensures results are relevant to the user’s query.

#### API Layer

* **FastAPI** serves as the REST API endpoint (`/rag`), receiving queries as JSON.
* Returns both the model-generated answer and the retrieved context chunks with source metadata.
* Example query for testing in Swagger UI:

```json
{
  "question": "Quels sont les points clés du programme de l'Ecolo pour l'environnement ?",
  "party": "ecolo"
}
```

#### Deployment on Azure (in progress...)

* Party programs, news articles, and other political documents will be stored in **Azure Blob Storage**.
* **Web App for Containers** will host the FastAPI RAG service and/or a streamlit app.
* **Azure Container Registry (ACR)** will store the container images.
* **Managed Identity** with **AcrPull** role will allow secure container pull.
* CI/CD pipeline will deploy new container versions automatically via **GitHub Actions**.
* Logs and monitoring through **Azure Log Analytics**.


## Usage

### Local Docker Testing

```bash
docker build -t votewise-rag:latest .
docker run -p 8000:8000 --env-file .env votewise-rag:latest
```

* Access `http://localhost:8000/docs` for Swagger UI.
* Use example queries to test RAG retrieval.

## Possible Improvements & Next Steps

* Integrate automated scraping from **RTBF news** to enrich document corpus with up-to-date political articles.
* Add social media streams (e.g., **X/Twitter**) for party mentions.
* Implement comparison between party programs and actual government measures.
* Extend language support for Flemish parties (Dutch).
* Test RAG performance. 
* Improve document ingestion (PDF loading) and chunking.
* Introduce unit testing, CI/CD pipelines, and scheduled updates for automated indexing.

## Notes on Filtering

* Filtering by party ensures that only relevant documents are retrieved.
* Example: querying MR pension proposals will only retrieve MR-related content.

Source chunk used for question on MR pension without filtering activated:
```text
travail des mesures pour les fins de carrière. Le PS propose de : ... (Source: ps-federal-2024.pdf)
```

## Free Tier Limitations

Using Azure free tiers for VoteWise RAG is fine for prototyping, but comes with constraints: limited Cognitive Search index size (50MB) and query throughput, restricted OpenAI/Foundry requests and models, single-instance Web App with no auto-scaling, and small Blob storage capacity. Free tiers are not suitable for large datasets, real-time monitoring, or production workloads.
