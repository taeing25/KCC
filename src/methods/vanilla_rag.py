"""
Role: Vanilla RAG baseline (Method 1).
Retrieves top-k chunks for the original query in a single round,
concatenates them in retrieval-score order, and answers with one LLM call.
"""

import logging
from typing import Dict, List

from ._utils import call_llm, build_context_text

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a precise question-answering assistant. "
    "Answer the question using only the provided context. "
    "Give a short, direct answer (1–5 words when possible). "
    "If the answer cannot be determined from the context, respond with 'unknown'."
)


def run_vanilla_rag(
    question: str,
    index,           # SampleIndex
    client,          # OpenAI
    config: Dict,
) -> Dict:
    """
    Single-round retrieval with the original query; no special ordering.
    Returns: prediction, used_chunks, decomposition (None).
    """
    chunks: List[Dict] = index.search(question, config["top_k"])
    chunks = chunks[: config["max_chunks"]]

    context_text = build_context_text(chunks, config["max_context_tokens"])
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    prediction = call_llm(client, config["model_name"], _SYSTEM, user_prompt)

    return {
        "prediction": prediction,
        "used_chunks": chunks,
        "decomposition": None,
    }
