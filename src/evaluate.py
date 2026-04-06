"""
Role: Compute evaluation metrics for multi-hop QA.
Implements HotpotQA official normalization for EM and token-level F1,
plus supporting fact hit rate (coverage of gold supporting passages).
"""

import re
import string
from collections import Counter
from typing import List, Dict


# ── Normalization ──────────────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def get_tokens(s: str) -> List[str]:
    return normalize_answer(s).split()


# ── Core Metrics ───────────────────────────────────────────────────────────────

def exact_match(prediction: str, gold: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = get_tokens(prediction)
    gold_tokens = get_tokens(gold)

    if not pred_tokens or not gold_tokens:
        return int(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


# ── Supporting Fact Hit Rate ───────────────────────────────────────────────────

def supporting_fact_hit_rate(
    chunks: List[Dict],
    supporting_facts: Dict,   # {"title": [...], "sent_id": [...]}
    sample_context: Dict,     # {"title": [...], "sentences": [[...]]}
) -> float:
    """
    Fraction of HotpotQA supporting facts covered by the retrieved chunks.
    A fact is 'hit' if any retrieved chunk shares the same title AND
    contains the fact's sentence text.
    """
    sf_titles = supporting_facts["title"]
    sf_sent_ids = supporting_facts["sent_id"]
    ctx_titles = sample_context["title"]
    ctx_sentences = sample_context["sentences"]

    # Collect (title, sentence_text) for each supporting fact
    sf_set: List[tuple] = []
    for sf_title, sf_sent_id in zip(sf_titles, sf_sent_ids):
        for ctx_title, ctx_sents in zip(ctx_titles, ctx_sentences):
            if ctx_title == sf_title and sf_sent_id < len(ctx_sents):
                sf_set.append((sf_title, ctx_sents[sf_sent_id].strip()))

    if not sf_set:
        return 0.0

    chunk_pairs = [(c["title"], c["text"]) for c in chunks]

    hits = 0
    for sf_title, sf_sent in sf_set:
        for chunk_title, chunk_text in chunk_pairs:
            if chunk_title == sf_title and sf_sent.lower() in chunk_text.lower():
                hits += 1
                break

    return hits / len(sf_set)


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate_metrics(results: List[Dict]) -> Dict:
    """Compute mean EM, F1, and SF hit rate over a list of sample result dicts."""
    if not results:
        return {"em": 0.0, "f1": 0.0, "sf_hit_rate": 0.0, "n": 0}

    em_scores = [r["em"] for r in results if "em" in r]
    f1_scores = [r["f1"] for r in results if "f1" in r]
    sf_scores = [r["sf_hit_rate"] for r in results if "sf_hit_rate" in r]

    return {
        "em": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "sf_hit_rate": sum(sf_scores) / len(sf_scores) if sf_scores else 0.0,
        "n": len(results),
    }
