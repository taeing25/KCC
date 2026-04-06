"""
Role: Orchestrates the full multi-hop QA experiment.

Pipeline per sample:
    1. Build per-sample FAISS index from HotpotQA distractor contexts
    2. Run all 5 methods (with result caching to outputs/)
    3. Compute EM, F1, supporting-fact hit rate
    4. Print comparison table + decomposition success rate

Run from project root:
    python src/run_experiment.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Allow `python src/run_experiment.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_hotpotqa_samples, chunk_context
from indexer import FaissIndexer, EmbeddingCache
from decomposer import QueryDecomposer
from evaluate import exact_match, token_f1, supporting_fact_hit_rate, aggregate_metrics
from methods import (
    run_vanilla_rag,
    run_ircot,
    run_ours,
    run_ours_ablation_a,
    run_ours_ablation_b,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _ROOT / "outputs"

# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(path: Optional[Path] = None) -> Dict:
    cfg_path = path or (_ROOT / "configs" / "config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)

# ── Cache helpers ──────────────────────────────────────────────────────────────

def _result_path(method: str, config: Dict) -> Path:
    model = config["model_name"].replace("/", "_")
    return _OUTPUT_DIR / f"{method}_{model}.jsonl"


def _load_cache(path: Path) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    results[item["sample_id"]] = item
    return results


def _append(path: Path, record: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ── Per-sample method runner ───────────────────────────────────────────────────

def _run_one(
    method: str,
    sample: Dict,
    index,
    client: OpenAI,
    config: Dict,
    decomposition: List[str],
) -> Dict:
    """Execute `method` on `sample`. Returns a result dict (never raises)."""
    question = sample["question"]
    gold = sample["answer"]

    try:
        if method == "vanilla_rag":
            out = run_vanilla_rag(question, index, client, config)

        elif method == "ircot":
            out = run_ircot(question, index, client, config)

        elif method == "ours_full":
            out = run_ours(question, index, client, config, decomposition)

        elif method == "ours_ablation_a":
            out = run_ours_ablation_a(
                question, index, client, config,
                decomposition, config["run_seed"],
            )

        elif method == "ours_ablation_b":
            out = run_ours_ablation_b(question, index, client, config)

        else:
            raise ValueError(f"Unknown method: {method!r}")

        prediction: str = out["prediction"]
        used_chunks: List[Dict] = out["used_chunks"]

        em = exact_match(prediction, gold)
        f1 = token_f1(prediction, gold)
        sf = supporting_fact_hit_rate(
            used_chunks,
            sample["supporting_facts"],
            sample["context"],
        )

        # Strip 'score' before serialising (numpy float → not JSON-serialisable)
        chunks_serial = [{k: v for k, v in c.items() if k != "score"} for c in used_chunks]

        return {
            "sample_id": sample["id"],
            "method": method,
            "question": question,
            "prediction": prediction,
            "gold": gold,
            "em": em,
            "f1": f1,
            "sf_hit_rate": sf,
            "used_chunks": chunks_serial,
            "decomposition": out.get("decomposition"),
            "error_reason": None,
        }

    except Exception as exc:
        logger.error("[%s] sample %s failed: %s", method, sample["id"], exc)
        return {
            "sample_id": sample["id"],
            "method": method,
            "question": question,
            "prediction": "",
            "gold": gold,
            "em": 0,
            "f1": 0.0,
            "sf_hit_rate": 0.0,
            "used_chunks": [],
            "decomposition": None,
            "error_reason": str(exc),
        }

# ── Reporting ──────────────────────────────────────────────────────────────────

_METHOD_ORDER = [
    "vanilla_rag",
    "ircot",
    "ours_ablation_a",
    "ours_ablation_b",
    "ours_full",
]

_METHOD_LABEL = {
    "vanilla_rag":     "Vanilla RAG       ",
    "ircot":           "IRCoT             ",
    "ours_ablation_a": "Ours (ablation A) ",
    "ours_ablation_b": "Ours (ablation B) ",
    "ours_full":       "Ours (full)       ",
}


def print_table(method_results: Dict[str, List[Dict]]):
    col = "방법               | EM     | F1     | SF Hit | N    | Fail"
    sep = "-" * len(col)
    print(f"\n{sep}")
    print(col)
    print(sep)
    for key in _METHOD_ORDER:
        rows = method_results.get(key, [])
        valid = [r for r in rows if not r.get("error_reason")]
        failed = len(rows) - len(valid)
        m = aggregate_metrics(valid)
        label = _METHOD_LABEL.get(key, key)
        print(
            f"{label:<19}| {m['em']:.4f} | {m['f1']:.4f} | "
            f"{m['sf_hit_rate']:.4f} | {m['n']:<4} | {failed}"
        )
    print(sep)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    load_dotenv(_ROOT / ".env")
    config = load_config()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found. Check your .env file.")

    client = OpenAI(api_key=api_key)
    cache = EmbeddingCache(_ROOT / "data" / "embed_cache.pkl")
    indexer = FaissIndexer(client, config["embedding_model"], cache)
    decomposer = QueryDecomposer()

    logger.info(
        "Loading %d samples (data_seed=%d)…",
        config["sample_size"], config["data_seed"],
    )
    samples = load_hotpotqa_samples(config["sample_size"], config["data_seed"])
    logger.info("Loaded %d samples.", len(samples))

    # Load existing cached results for all methods
    caches: Dict[str, Dict] = {m: _load_cache(_result_path(m, config)) for m in _METHOD_ORDER}
    for m, c in caches.items():
        logger.info("[%s] %d cached results.", m, len(c))

    method_results: Dict[str, List[Dict]] = {m: [] for m in _METHOD_ORDER}

    for sample in tqdm(samples, desc="Samples", unit="sample"):
        sid = sample["id"]

        # Build per-sample FAISS index (embeddings cached)
        chunks = chunk_context(sample["context"], config["chunk_size"], config["chunk_overlap"])
        sample_index = indexer.build_index(chunks)

        # Decompose once per sample (fast; no API cost)
        decomposition = decomposer.decompose(sample["question"], sample["type"])

        for method in _METHOD_ORDER:
            if sid in caches[method]:
                result = caches[method][sid]
            else:
                result = _run_one(method, sample, sample_index, client, config, decomposition)
                path = _result_path(method, config)
                _append(path, result)
                caches[method][sid] = result

            method_results[method].append(result)

    # ── Final report ───────────────────────────────────────────────────────────
    print_table(method_results)

    total_samples = len(samples)
    print(
        f"\n분해 성공률 : {decomposer.get_success_rate():.4f} "
        f"({decomposer.success_count} / {decomposer.total_count} 샘플)"
    )

    total_fail = sum(
        sum(1 for r in rows if r.get("error_reason"))
        for rows in method_results.values()
    )
    print(f"총 API 실패 : {total_fail} / {total_samples * len(_METHOD_ORDER)} 호출")
    print(f"결과 저장   : {_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
