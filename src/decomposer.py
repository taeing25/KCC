"""
Role: LLM-based query decomposition with disk cache.
Uses few-shot prompting. Handles HotpotQA 'bridge' and 'comparison' question types.
Tracks decomposition success rate for reporting in the final paper.

Cache is stored at data/decomp_cache.json — no API call for already-seen questions.
"""

import json
import logging
import re
from pathlib import Path
from typing import List

from openai import OpenAI

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CACHE = _PROJECT_ROOT / "data" / "decomp_cache.json"

# ── Prompts ────────────────────────────────────────────────────────────────────

_BRIDGE_SYSTEM = (
    "Decompose the multi-hop question into ordered sub-questions where each answer "
    "feeds into the next. Return a JSON array of strings only. No explanation."
)

_BRIDGE_EXAMPLES = [
    (
        "Who is the father of the director of Interstellar?",
        '["Who directed Interstellar?", "Who is the father of that director?"]',
    ),
    (
        "What is the nationality of the director of the film that won Best Picture in 2020?",
        '["Which film won Best Picture in 2020?", "Who directed that film?", "What is the nationality of that director?"]',
    ),
    (
        "What year was the birthplace of the author of Hamlet founded?",
        '["Who wrote Hamlet?", "Where was that author born?", "What year was that city founded?"]',
    ),
]

_COMPARISON_SYSTEM = (
    "Decompose the comparison question into parallel sub-questions, one per entity. "
    "Return a JSON array of strings only. No explanation."
)

_COMPARISON_EXAMPLES = [
    (
        "Were Scott Derrickson and Ed Wood both American directors?",
        '["What is Scott Derrickson\'s nationality and profession?", "What is Ed Wood\'s nationality and profession?"]',
    ),
    (
        "Which film was released first, Inception or The Dark Knight?",
        '["When was Inception released?", "When was The Dark Knight released?"]',
    ),
    (
        "Are both the Eiffel Tower and the Colosseum located in Europe?",
        '["Where is the Eiffel Tower located?", "Where is the Colosseum located?"]',
    ),
]


def _build_prompt(question: str, examples: list) -> str:
    parts = [f'Q: "{q}"\nA: {a}' for q, a in examples]
    parts.append(f'Q: "{question}"\nA:')
    return "\n\n".join(parts)


# ── Decomposer ─────────────────────────────────────────────────────────────────

class QueryDecomposer:
    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        cache_path: Path = _DEFAULT_CACHE,
    ):
        self.client = client
        self.model = model_name
        self._cache_path = Path(cache_path)
        self._cache: dict = self._load_cache()

        self.total_count: int = 0
        self.success_count: int = 0

    # ── Cache ──────────────────────────────────────────────────────────────────

    def _load_cache(self) -> dict:
        if self._cache_path.exists():
            with open(self._cache_path, encoding="utf-8") as f:
                data = json.load(f)
            logger.info(
                "Loaded %d cached decompositions from %s", len(data), self._cache_path
            )
            return data
        return {}

    def _save_cache(self):
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    # ── Public API ─────────────────────────────────────────────────────────────

    def decompose(self, question: str, q_type: str) -> List[str]:
        """
        Decompose `question` into sub-questions based on `q_type`.
        Returns [q1, q2, ...] on success or [question] on failure.
        Results are cached to disk; no LLM call for repeated questions.
        """
        self.total_count += 1
        question = question.strip()

        cache_key = f"{q_type}:{question}"
        if cache_key in self._cache:
            sub_qs = self._cache[cache_key]
            if self._is_valid(sub_qs, question):
                self.success_count += 1
            return sub_qs

        sub_qs = self._llm_decompose(question, q_type)
        self._cache[cache_key] = sub_qs
        self._save_cache()

        if self._is_valid(sub_qs, question):
            self.success_count += 1
            logger.debug("Decomposed [%s]: %r → %s", q_type, question, sub_qs)
        else:
            logger.debug("Decomposition failed [%s]: %r", q_type, question)

        return sub_qs

    def get_success_rate(self) -> float:
        return self.success_count / self.total_count if self.total_count > 0 else 0.0

    # ── LLM call ───────────────────────────────────────────────────────────────

    def _llm_decompose(self, question: str, q_type: str) -> List[str]:
        if q_type == "bridge":
            system, examples = _BRIDGE_SYSTEM, _BRIDGE_EXAMPLES
        elif q_type == "comparison":
            system, examples = _COMPARISON_SYSTEM, _COMPARISON_EXAMPLES
        else:
            return [question]

        user_prompt = _build_prompt(question, examples)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=256,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            return self._parse(raw, question)
        except Exception as exc:
            logger.warning("Decomposition LLM call failed: %s", exc)
            return [question]

    # ── Parsing & validation ───────────────────────────────────────────────────

    def _parse(self, raw: str, fallback: str) -> List[str]:
        """Parse JSON array from LLM output. Returns [fallback] on failure."""
        try:
            result = json.loads(raw)
            if isinstance(result, list) and all(isinstance(x, str) for x in result):
                return [q.strip() for q in result if q.strip()]
        except json.JSONDecodeError:
            pass

        # Try extracting a JSON array embedded in surrounding text
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group())
                if isinstance(result, list) and all(isinstance(x, str) for x in result):
                    return [q.strip() for q in result if q.strip()]
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse decomposition output: %r", raw)
        return [fallback]

    def _is_valid(self, sub_qs: List[str], original: str) -> bool:
        """2+ non-empty, non-duplicate sub-questions, not all identical to original."""
        if len(sub_qs) < 2:
            return False
        cleaned = [q.strip().lower() for q in sub_qs]
        if any(not q for q in cleaned):
            return False
        if len(set(cleaned)) < len(cleaned):
            return False
        if all(q == original.strip().lower() for q in cleaned):
            return False
        return True
