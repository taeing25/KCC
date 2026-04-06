"""
Role: Rule-based query decomposition using spaCy NER and regex patterns.
No LLM is used. Handles HotpotQA 'bridge' and 'comparison' question types.
Tracks decomposition success rate for reporting in the final paper.

Requires: pip install spacy && python -m spacy download en_core_web_sm
"""

import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class QueryDecomposer:
    def __init__(self):
        self._nlp = None
        self._spacy_ok = False
        self._load_spacy()

        self.total_count: int = 0
        self.success_count: int = 0

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _load_spacy(self):
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            self._spacy_ok = True
            logger.info("spaCy en_core_web_sm loaded.")
        except Exception as e:
            logger.warning(
                f"spaCy not available ({e}). Using regex-only fallback. "
                "Run: pip install spacy && python -m spacy download en_core_web_sm"
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def decompose(self, question: str, q_type: str) -> List[str]:
        """
        Decompose `question` into sub-questions based on `q_type`.

        Returns [q1, q2] on success or [question] on failure.
        Success criteria: 2+ non-empty, non-duplicate sub-questions.
        """
        self.total_count += 1
        question = question.strip()

        if q_type == "bridge":
            sub_qs = self._decompose_bridge(question)
        elif q_type == "comparison":
            sub_qs = self._decompose_comparison(question)
        else:
            sub_qs = [question]

        if self._is_valid(sub_qs, question):
            self.success_count += 1
            logger.debug("Decomposed [%s]: %r → %s", q_type, question, sub_qs)
            return sub_qs

        logger.debug("Decomposition failed [%s]: %r", q_type, question)
        return [question]

    def get_success_rate(self) -> float:
        return self.success_count / self.total_count if self.total_count > 0 else 0.0

    # ── Validation ─────────────────────────────────────────────────────────────

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

    # ── Bridge Decomposition ───────────────────────────────────────────────────

    def _decompose_bridge(self, question: str) -> List[str]:
        """
        Split on the relative clause introduced by that/which/who/whom.

        Example:
            "Who directed the film that won Best Picture in 2020?"
            → q1: "Which film won Best Picture in 2020?"
            → q2: "Who directed the film?"
        """
        q = question.rstrip("?").strip()

        # Match: [pre-clause] [that|which|who|whom] [embedded-clause]
        m = re.match(
            r"^(.+?)\s+\b(that|which|who|whom)\b\s+(.+)$",
            q,
            flags=re.IGNORECASE,
        )
        if not m:
            return [question]

        pre = m.group(1).strip()   # e.g. "Who directed the film"
        post = m.group(3).strip()  # e.g. "won Best Picture in 2020"

        head_noun = self._head_noun(pre)
        if not head_noun:
            return [question]

        # q1: "Which/Who [head_noun] [post_clause]?"
        wh = self._wh_for(head_noun)
        q1 = f"{wh} {head_noun} {post}?"

        # q2: main question without the relative clause
        q2 = f"{pre}?"

        return [q1, q2]

    def _head_noun(self, text: str) -> str:
        """Return the last noun phrase from `text` (bridge entity anchor)."""
        if self._spacy_ok:
            doc = self._nlp(text)
            chunks = list(doc.noun_chunks)
            if chunks:
                raw = chunks[-1].text
                # Strip leading article
                return re.sub(r"^(the|a|an)\s+", "", raw, flags=re.IGNORECASE).strip()

        # Regex fallback: last content word
        stop = {"who", "what", "where", "when", "why", "how",
                "did", "does", "is", "was", "were", "the", "a", "an",
                "of", "in", "on", "at", "to", "by"}
        for word in reversed(text.split()):
            w = re.sub(r"[^\w]", "", word).lower()
            if w and w not in stop:
                return word
        return ""

    def _wh_for(self, noun: str) -> str:
        """Choose 'Who', 'Which', or 'What' based on noun type."""
        person_words = {
            "person", "people", "man", "woman", "director", "actor",
            "actress", "president", "author", "founder", "singer",
            "player", "writer", "composer", "artist", "politician",
        }
        noun_lower = noun.lower()

        if self._spacy_ok:
            doc = self._nlp(noun)
            if any(ent.label_ == "PERSON" for ent in doc.ents):
                return "Who"

        if any(w in noun_lower for w in person_words):
            return "Who"

        return "Which"

    # ── Comparison Decomposition ───────────────────────────────────────────────

    def _decompose_comparison(self, question: str) -> List[str]:
        """
        Split on two named entities for comparison questions.

        Example:
            "Are both The Dark Knight and Inception Christopher Nolan films?"
            → q1: "Is The Dark Knight a Christopher Nolan film?"
            → q2: "Is Inception a Christopher Nolan film?"
        """
        entities = self._two_entities(question)
        if not entities:
            return [question]

        e1, e2 = entities
        q1 = self._isolate(question, keep=e1, remove=e2)
        q2 = self._isolate(question, keep=e2, remove=e1)

        if q1 and q2:
            return [q1, q2]

        # Last-resort fallback
        return [f"What is {e1}?", f"What is {e2}?"]

    def _two_entities(self, question: str) -> Optional[List[str]]:
        """Extract up to two named entities for a comparison question."""
        # spaCy NER first
        if self._spacy_ok:
            doc = self._nlp(question)
            target = {
                "PERSON", "ORG", "WORK_OF_ART", "PRODUCT",
                "FAC", "GPE", "LOC", "EVENT", "NORP",
            }
            ents = [ent.text for ent in doc.ents if ent.label_ in target]
            if len(ents) >= 2:
                return ents[:2]

        # Regex fallback: "X and Y" with initial capitals
        m = re.search(
            r"\b(?:both\s+)?([A-Z][^\?,\.]+?)\s+and\s+([A-Z][^\?,\.]+?)(?=\s+\w|\?|$)",
            question,
        )
        if m:
            e1, e2 = m.group(1).strip(), m.group(2).strip()
            if e1 and e2 and e1.lower() != e2.lower():
                return [e1, e2]

        return None

    def _isolate(self, question: str, keep: str, remove: str) -> str:
        """Generate a sub-question focused on `keep` by removing `remove`."""
        q = question

        # Remove "both " prefix
        q = re.sub(r"\bboth\s+", "", q, flags=re.IGNORECASE)

        # Remove " and [remove]" or "[remove] and "
        q = re.sub(r"\s+and\s+" + re.escape(remove), "", q, flags=re.IGNORECASE)
        q = re.sub(re.escape(remove) + r"\s+and\s+", "", q, flags=re.IGNORECASE)

        # Fix verb agreement for now-singular subject
        q = re.sub(r"^Are\b", "Is", q)
        q = re.sub(r"^Do\b", "Does", q)
        q = re.sub(r"^Were\b", "Was", q)
        q = re.sub(r"^Have\b", "Has", q)

        q = re.sub(r"\s+", " ", q).strip()
        if q and not q.endswith("?"):
            q += "?"
        return q
