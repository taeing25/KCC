"""
Shared utilities for all QA methods:
- Token counting and context truncation (tiktoken)
- Context string construction from chunk list
- OpenAI chat completion with exponential-backoff retry
"""

import time
import logging
from typing import List, Dict

import tiktoken
from openai import OpenAI

logger = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _enc.decode(tokens[:max_tokens])


def build_context_text(chunks: List[Dict], max_tokens: int) -> str:
    """
    Concatenate chunks into a single context string.
    Stops adding chunks once `max_tokens` would be exceeded;
    partially includes the last chunk if space remains.
    """
    parts: List[str] = []
    used = 0
    for chunk in chunks:
        block = f"[{chunk['title']}]\n{chunk['text']}\n"
        block_tokens = count_tokens(block)
        if used + block_tokens > max_tokens:
            remaining = max_tokens - used
            if remaining > 50:
                parts.append(truncate_to_tokens(block, remaining))
            break
        parts.append(block)
        used += block_tokens
    return "\n".join(parts)


def call_llm(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 256,
    max_retries: int = 3,
) -> str:
    """
    Call OpenAI chat completion with exponential-backoff retry on failure.
    Raises the last exception if all retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, max_retries, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.error("LLM call failed after %d attempts: %s", max_retries, exc)
                raise
