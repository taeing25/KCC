"""
Role: Load HotpotQA distractor validation split, sample N examples with a fixed seed,
and split each document's context into token-bounded overlapping chunks.
"""

import random
import tiktoken
from typing import List, Dict
from datasets import load_dataset


def load_hotpotqa_samples(sample_size: int, data_seed: int) -> List[Dict]:
    """
    Load HotpotQA distractor validation split and sample `sample_size` examples.
    Uses `data_seed` for reproducibility of the sample selection.
    """
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    total = len(ds)
    rng = random.Random(data_seed)
    indices = sorted(rng.sample(range(total), min(sample_size, total)))

    samples = []
    for i in indices:
        item = ds[i]
        samples.append({
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "type": item["type"],           # "bridge" or "comparison"
            "context": {
                "title": list(item["context"]["title"]),
                "sentences": [list(s) for s in item["context"]["sentences"]],
            },
            "supporting_facts": {
                "title": list(item["supporting_facts"]["title"]),
                "sent_id": list(item["supporting_facts"]["sent_id"]),
            },
        })
    return samples


def chunk_context(
    context: Dict,      # {"title": [str, ...], "sentences": [[str, ...], ...]}
    chunk_size: int,    # max tokens per chunk
    chunk_overlap: int, # token overlap between consecutive chunks
) -> List[Dict]:
    """
    Split each document in `context` into token-bounded chunks.

    Returns a list of chunk dicts:
        chunk_id    : int  – unique id within this sample
        doc_idx     : int  – index of source document
        title       : str  – document title
        text        : str  – chunk text
        token_start : int  – start token offset within the document
        token_end   : int  – end token offset
    """
    enc = tiktoken.get_encoding("cl100k_base")
    chunks: List[Dict] = []
    chunk_id = 0

    for doc_idx, (title, sentences) in enumerate(
        zip(context["title"], context["sentences"])
    ):
        full_text = " ".join(sentences)
        tokens = enc.encode(full_text)

        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_text = enc.decode(tokens[start:end])
            chunks.append({
                "chunk_id": chunk_id,
                "doc_idx": doc_idx,
                "title": title,
                "text": chunk_text,
                "token_start": start,
                "token_end": end,
            })
            chunk_id += 1
            if end == len(tokens):
                break
            start += chunk_size - chunk_overlap

    return chunks
