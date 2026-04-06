import os, sys, json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
load_dotenv()

from data_loader import load_hotpotqa_samples
from decomposer import QueryDecomposer

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
decomposer = QueryDecomposer(client, "gpt-4o-mini")

samples = load_hotpotqa_samples(10, 42)
print(f"Loaded {len(samples)} samples\n")

for s in samples:
    result = decomposer.decompose(s["question"], s["type"])
    status = "✅" if len(result) >= 2 else "❌"
    print(f"{status} [{s['type']}] {s['question']}")
    for i, q in enumerate(result, 1):
        print(f"   q{i}: {q}")
    print()

print(f"Success rate: {decomposer.get_success_rate():.1%}")