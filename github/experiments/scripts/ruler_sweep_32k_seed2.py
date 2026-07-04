"""Second-seed variant of ruler_sweep_32k_wrapper.py.

Identical to ruler_sweep_32k_wrapper.py except that each task's 500-example
SaylorTwift split is SHUFFLED with SHUFFLE_SEED (env) before taking the first
100, so this evaluates a genuinely different input draw than the frozen
first-100 run in ruler_32k_qwen15b_sweep.jsonl. Everything else (prompt
assembly, prefill/scoring, eviction, decoding, correctness) is delegated to
ruler_sweep.main() unchanged.
"""
import os
import sys

from datasets import Dataset, concatenate_datasets, load_dataset

sys.path.insert(0, "experiments/scripts")
import ruler_sweep

SRC = "SaylorTwift/RULER-32768-Qwen2.5-3B-tokenizer"
SEED = int(os.environ["SHUFFLE_SEED"])
N = int(os.environ.get("DRAW_N", "100"))


def load_32k(name, config, split="test"):
    assert config == "32768", f"wrapper only serves 32768, got {config}"
    parts = []
    for task in ["vt", "fwe", "niah_multikey_3"]:
        ds = load_dataset(SRC, split=task).shuffle(seed=SEED).select(range(N))
        mapped = Dataset.from_dict({
            "context": ds["input"],
            "question": [""] * len(ds),
            "answer_prefix": [""] * len(ds),
            "answer": ds["outputs"],
            "task": [task] * len(ds),
            "max_new_tokens": [128] * len(ds),
        })
        parts.append(mapped)
    return concatenate_datasets(parts)


ruler_sweep.load_dataset = load_32k

if __name__ == "__main__":
    print(f"[seed2 wrapper] SHUFFLE_SEED={SEED} N={N}")
    ruler_sweep.main()
