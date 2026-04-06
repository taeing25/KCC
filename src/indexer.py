"""
Role: Build per-sample FAISS indices from context chunks using OpenAI embeddings.
Caches embeddings to disk (pickle) to avoid redundant API calls across runs.
Uses cosine similarity via L2-normalised inner-product search.
"""

import os
import hashlib
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import faiss
from openai import OpenAI

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CACHE = _PROJECT_ROOT / "data" / "embed_cache.pkl"


class EmbeddingCache:
    """Persistent dict-based cache keyed by MD5 of the input text."""

    def __init__(self, cache_path: Path = _DEFAULT_CACHE):
        self.path = Path(cache_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, List[float]] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path, "rb") as f:
                self._data = pickle.load(f)
            logger.info("Loaded %d cached embeddings from %s", len(self._data), self.path)

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self._data, f)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        return self._data.get(self._key(text))

    def set(self, text: str, embedding: List[float]):
        self._data[self._key(text)] = embedding

    def has(self, text: str) -> bool:
        return self._key(text) in self._data


class FaissIndexer:
    """Embeds texts with OpenAI and builds/queries FAISS indices."""

    _BATCH_SIZE = 512  # max texts per OpenAI embeddings request

    def __init__(
        self,
        client: OpenAI,
        embedding_model: str,
        cache: EmbeddingCache,
    ):
        self.client = client
        self.model = embedding_model
        self.cache = cache

    # ── Embedding ──────────────────────────────────────────────────────────────

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for `texts`, using cache and batched API calls."""
        results: List[Optional[List[float]]] = [None] * len(texts)
        missing_idx: List[int] = []
        missing_texts: List[str] = []

        for i, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                results[i] = cached
            else:
                missing_idx.append(i)
                missing_texts.append(text)

        if missing_texts:
            all_new: List[List[float]] = []
            for start in range(0, len(missing_texts), self._BATCH_SIZE):
                batch = missing_texts[start : start + self._BATCH_SIZE]
                resp = self.client.embeddings.create(model=self.model, input=batch)
                all_new.extend(d.embedding for d in resp.data)

            for orig_idx, text, emb in zip(missing_idx, missing_texts, all_new):
                self.cache.set(text, emb)
                results[orig_idx] = emb
            self.cache.save()

        return results  # type: ignore[return-value]

    def embed_query(self, query: str) -> np.ndarray:
        emb = np.array(self._embed_batch([query]), dtype=np.float32)
        faiss.normalize_L2(emb)
        return emb

    # ── Index Building ─────────────────────────────────────────────────────────

    def build_index(self, chunks: List[Dict]) -> "SampleIndex":
        """Build a FAISS flat inner-product index for one sample's chunks."""
        texts = [c["text"] for c in chunks]
        embeddings = self._embed_batch(texts)
        arr = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(arr)

        index = faiss.IndexFlatIP(arr.shape[1])
        index.add(arr)
        return SampleIndex(index=index, chunks=chunks, indexer=self)


class SampleIndex:
    """FAISS index for a single HotpotQA sample."""

    def __init__(
        self,
        index: faiss.IndexFlatIP,
        chunks: List[Dict],
        indexer: FaissIndexer,
    ):
        self.index = index
        self.chunks = chunks
        self.indexer = indexer

    def search(self, query: str, top_k: int) -> List[Dict]:
        """Return top-k chunks ranked by cosine similarity to `query`."""
        q_emb = self.indexer.embed_query(query)
        k = min(top_k, len(self.chunks))
        scores, idxs = self.index.search(q_emb, k)

        results: List[Dict] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx >= 0:
                chunk = dict(self.chunks[idx])
                chunk["score"] = float(score)
                results.append(chunk)
        return results
