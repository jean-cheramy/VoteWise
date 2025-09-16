from typing import List
import re
import requests

API_URL: str = "http://localhost:11434/api/generate"
MODEL_NAME: str = "llama3"

PROMPT_TEMPLATE: str = """You are an expert in Belgian politics.
Provide a concise summary of political party positions based on the provided documents.
Answer clearly in the language of the question and highlight key differences.

Context:
{context}

Question:
{question}
"""

def generate_answer_rest_api(prompt: str, model: str = MODEL_NAME, max_tokens: int = 500) -> str:
    """
    Generate text using the Ollama REST API in non-streaming mode.

    Args:
        prompt (str): The prompt text to send to the model.
        model (str): The model name.
        max_tokens (int): Maximum tokens to generate.

    Returns:
        str: Generated text from the model.
    """
    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": False
    }

    response = requests.post(API_URL, json=data)
    response.raise_for_status()
    result = response.json()
    return result.get("response", "").strip()


def combine_summaries(summaries: List[str], question: str) -> str:
    """
    Combine multiple chunk summaries into a single final summary.

    Args:
        summaries (List[str]): List of intermediate summaries.
        question (str): Original question.

    Returns:
        str: Final combined summary.
    """
    if len(summaries) > 1:
        combined_context = " ".join(summaries)
        final_prompt = PROMPT_TEMPLATE.format(context=combined_context, question=question)
        return generate_answer_rest_api(final_prompt)
    return summaries[0]


def generate_answer(question: str, contexts: List[str], max_tokens_per_chunk: int = 400) -> str:
    """
    Summarize multiple context documents by splitting them into smaller chunks,
    generating per-chunk summaries, and combining them into a final answer.

    Args:
        question (str): The user's question.
        contexts (List[str]): List of context documents to summarize.
        max_tokens_per_chunk (int): Approximate token limit per chunk.

    Returns:
        str: Final answer summarizing the contexts.
    """
    text = " ".join(contexts)
    sentences = re.split(r'(?<=[.?!])\s+', text)

    chunks: List[str] = []
    current_chunk: str = ""
    token_count: int = 0

    for sent in sentences:
        token_count += len(sent.split())
        if token_count > max_tokens_per_chunk:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sent
            token_count = len(sent.split())
        else:
            current_chunk += " " + sent
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    summaries: List[str] = [
        generate_answer_rest_api(PROMPT_TEMPLATE.format(context=chunk, question=question))
        for chunk in chunks
    ]

    return combine_summaries(summaries, question)
