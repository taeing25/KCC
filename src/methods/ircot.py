"""
Role: IRCoT baseline (Method 2) — Interleaved Retrieval with Chain-of-Thought.
Alternates between generating a single reasoning step and retrieving new evidence.
Budget: up to max_retrieval_rounds CoT+retrieve steps + 1 final answer call
        = max_llm_calls total (default 3 + 1 = 4).
"""

import re
import logging
from typing import Dict, List

from ._utils import call_llm, build_context_text

logger = logging.getLogger(__name__)

_SYSTEM_COT = (
    "You are a careful reasoning assistant. "
    "Given a question and supporting context, think step by step. "
    "At the end of your response, write exactly one sentence starting with "
    "'Search query:' that names what you still need to look up to answer the question. "
    "Do NOT give the final answer yet."
)

_SYSTEM_ANSWER = (
    "You are a precise question-answering assistant. "
    "Given the question, accumulated context, and step-by-step reasoning, "
    "provide the final answer. "
    "Give a short, direct answer (1–5 words when possible). "
    "If the answer cannot be determined, respond with 'unknown'."
)


def _extract_search_query(thought: str, fallback: str) -> str:
    """Pull the 'Search query: ...' sentence from a CoT thought."""
    m = re.search(r"search query:\s*(.+?)(?:\.|$)", thought, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: last non-empty sentence
    sentences = [s.strip() for s in thought.split(".") if s.strip()]
    return sentences[-1] if sentences else fallback


def run_ircot(
    question: str,
    index,       # SampleIndex
    client,      # OpenAI
    config: Dict,
) -> Dict:
    """
    IRCoT: iterative CoT + retrieval.
    LLM call budget: max_retrieval_rounds reasoning calls + 1 answer call.
    """
    max_rounds: int = config["max_retrieval_rounds"]   # 3
    max_calls: int = config["max_llm_calls"]           # 4
    top_k: int = config["top_k"]
    max_chunks: int = config["max_chunks"]
    max_ctx: int = config["max_context_tokens"]
    model: str = config["model_name"]

    all_chunks: List[Dict] = []
    seen_ids: set = set()
    thoughts: List[str] = []
    llm_calls: int = 0
    next_query: str = question

    for _ in range(min(max_rounds, max_calls - 1)):
        # Retrieve
        for c in index.search(next_query, top_k):
            if c["chunk_id"] not in seen_ids:
                seen_ids.add(c["chunk_id"])
                all_chunks.append(c)

        context_text = build_context_text(all_chunks[:max_chunks], max_ctx)

        # Build reasoning prompt
        prior = "\n".join(f"Step {i+1}: {t}" for i, t in enumerate(thoughts))
        user_prompt = (
            f"Question: {question}\n\n"
            f"Context:\n{context_text}\n\n"
            + (f"Reasoning so far:\n{prior}\n\n" if prior else "")
            + "Continue reasoning (one step, then provide a Search query):"
        )
        thought = call_llm(client, model, _SYSTEM_COT, user_prompt, max_tokens=250)
        thoughts.append(thought)
        llm_calls += 1

        next_query = _extract_search_query(thought, question)
        if llm_calls >= max_calls - 1:
            break

    # Final answer
    context_text = build_context_text(all_chunks[:max_chunks], max_ctx)
    reasoning = "\n".join(f"Step {i+1}: {t}" for i, t in enumerate(thoughts))
    user_prompt = (
        f"Question: {question}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Reasoning:\n{reasoning}\n\n"
        "Final Answer:"
    )
    prediction = call_llm(client, model, _SYSTEM_ANSWER, user_prompt)

    return {
        "prediction": prediction,
        "used_chunks": all_chunks[:max_chunks],
        "decomposition": thoughts,
    }
