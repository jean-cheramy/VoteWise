# VoteWise — Belgian Political Program Comparator

## Overview

VoteWise is a prototype system that helps users explore and summarize political party positions in Belgium. It leverages retrieval-augmented generation (RAG) using a local Ollama language model (llama3) and FAISS embeddings to efficiently summarize party programs from documents.

Users can ask questions in natural language, and VoteWise retrieves relevant chunks from the uploaded documents and generates concise, multilingual summaries.

## Features (First Version)

- Load political program PDFs for each party.
- Split documents into chunks and embed them using Ollama embeddings.
- Build a FAISS vector store for efficient similarity search.
- Summarize retrieved chunks using a local llama3 Ollama model via REST API.
- Streamlit interface for asking questions and displaying answers along with the retrieved contexts.

## Usage

- Place PDFs for each party in the raw data folder.
- Build the FAISS index to prepare the embeddings and vector store.
- Launch the Streamlit app.
- Ask questions in natural language and view the summarized answers along with the retrieved document context.

## Possible Improvements

- Add support for multiple parties and organize documents per party.
- Automatically scrape party websites (airflow + github actions) or social media (e.g., tweets) to update documents.
- Improve Streamlit interface for better UX, such as filtering by party or topic.
- Switch to a faster embedding model or implement GPU support to speed up indexing.
- 

## Notes

- Ensure Ollama is running locally with the llama3 model.
- The system currently uses REST API calls with streaming disabled.
- Only use trusted documents for indexing to avoid security issues with deserialization.