"""
Thin Ollama wrapper for Graph RAG.
"""

import ollama

from graph_rag.config import OLLAMA_MODEL


def ask(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Send a prompt to Ollama and return the response text."""
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response["response"]
    except Exception as exc:
        raise RuntimeError(f"Ollama error ({model}): {exc}") from exc
