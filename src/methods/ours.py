"""
Role: Proposed method and its two ablations (Methods 3–5).

Ours (full)  — decompose → retrieve per sub-q → deduplicate → reasoning-chain ordering → answer
Ablation A   — same as full but chunks randomly shuffled  (isolates ordering effect)
Ablation B   — original query retrieval + position-biased ordering  (isolates decomp effect)

Position-biased ordering in Ablation B reverses retrieval order so the most relevant
chunk appears last, exploiting the recency/end-of-context attention bias
("Lost in the Middle", Liu et al. 2023).
"""

import random
import logging
from typing import Dict, List

from ._utils import call_llm, build_context_text

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a precise question-answering assistant. "
    "Answer the question using only the provided context. "
    "The context is ordered to support step-by-step reasoning — "
    "earlier passages establish intermediate facts, Read the context from beginning to end to build reasoning progressively. "
    "Give a short, direct answer (1–5 words when possible). "
    "If the answer cannot be determined, respond with 'unknown'."
)


# ── Retrieval helpers ──────────────────────────────────────────────────────────

def _retrieve_by_subquestions(
    sub_questions: List[str],
    index,
    top_k: int,
    max_chunks: int,
) -> List[Dict]:
    """
    Retrieve chunks for each sub-question in order, deduplicate by chunk_id,
    and preserve reasoning-chain ordering (q1 chunks first, q2 chunks second).
    """
    seen_ids: set = set()
    ordered: List[Dict] = []

    for sub_q in sub_questions:
        for chunk in index.search(sub_q, top_k):
            if chunk["chunk_id"] not in seen_ids:
                seen_ids.add(chunk["chunk_id"])
                chunk = dict(chunk)
                chunk["sub_question"] = sub_q
                ordered.append(chunk)
            if len(ordered) >= max_chunks:
                return ordered

    return ordered[:max_chunks]


# ── Method implementations ────────────────────────────────────────────────────

def run_ours(
    question: str,
    index,
    client,
    config: Dict,
    decomposition: List[str],
) -> Dict:
    """
    Full proposed method.
    Sub-questions drive retrieval; chunks arrive in reasoning-chain order.
    """
    chunks = _retrieve_by_subquestions(
        decomposition, index, config["top_k"], config["max_chunks"]
    )
    context_text = build_context_text(chunks, config["max_context_tokens"])
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    prediction = call_llm(client, config["model_name"], _SYSTEM, user_prompt)

    return {
        "prediction": prediction,
        "used_chunks": chunks,
        "decomposition": decomposition,
    }


def run_ours_ablation_a(
    question: str,
    index,
    client,
    config: Dict,
    decomposition: List[str],
    run_seed: int,
) -> Dict:
    """
    Ablation A: decomposition preserved, chunk order randomly shuffled.
    Tests whether ordering alone contributes to performance.
    """
    chunks = _retrieve_by_subquestions(
        decomposition, index, config["top_k"], config["max_chunks"]
    )
    rng = random.Random(run_seed)
    shuffled = list(chunks)
    rng.shuffle(shuffled)

    context_text = build_context_text(shuffled, config["max_context_tokens"])
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    prediction = call_llm(client, config["model_name"], _SYSTEM, user_prompt)

    return {
        "prediction": prediction,
        "used_chunks": shuffled,
        "decomposition": decomposition,
    }


def run_ours_ablation_b(
    question: str,
    index,
    client,
    config: Dict,
) -> Dict:
    """
    Ablation B: no decomposition (original query), with position-biased ordering.
    Retrieves top-k chunks, same as Vanilla RAG, but reverses their order.
    Tests whether decomposition (vs ordering alone) matters.
    """
    # Retrieve same number of candidates as Vanilla RAG for fair comparison
    chunks = index.search(question, config["top_k"])
    chunks = chunks[: config["max_chunks"]]

    # Reverse: most relevant chunk ends up last (position-biased ordering without decomposition)
    position_biased = list(reversed(chunks))

    context_text = build_context_text(position_biased, config["max_context_tokens"])
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    prediction = call_llm(client, config["model_name"], _SYSTEM, user_prompt)

    return {
        "prediction": prediction,
        "used_chunks": position_biased,
        "decomposition": None,
    }
