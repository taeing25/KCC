# Methods package for multi-hop QA experiment
from .vanilla_rag import run_vanilla_rag
from .ircot import run_ircot
from .ours import run_ours, run_ours_ablation_a, run_ours_ablation_b

__all__ = [
    "run_vanilla_rag",
    "run_ircot",
    "run_ours",
    "run_ours_ablation_a",
    "run_ours_ablation_b",
]
