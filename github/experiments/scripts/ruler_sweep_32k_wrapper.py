"""Wrapper for ruler_sweep.py at 32K context.

simonjegou/ruler (the dataset used for all 4K/16K cells) only publishes
configs {4096, 8192, 16384}; the '32768' choice in ruler_sweep.py has no
backing data. This wrapper substitutes SaylorTwift/RULER-32768-Qwen2.5-3B-tokenizer
(official RULER generation pipeline, sequences sized to 32768 tokens under the
Qwen2.5 tokenizer, which is shared across Qwen2.5 model sizes) and maps it into
the simonjegou schema, then delegates everything else — prompt assembly,
prefill/scoring, eviction, decoding, output format — to ruler_sweep.main()
unchanged.

Schema mapping (SaylorTwift splits are per-task, fields index/input/outputs/length):
  context        <- input   (the full RULER prompt incl. question + answer prefix)
  question       <- ""      (already inside `input`; build_prompt handles empties)
  answer_prefix  <- ""      (idem)
  answer         <- outputs
  task           <- split name
  max_new_tokens <- 128     (ruler_sweep uses max(max_new_tokens, --max_new=128))

Usage: same CLI as ruler_sweep.py; --config must be 32768.
"""
import sys

from datasets import Dataset, concatenate_datasets, load_dataset

sys.path.insert(0, "experiments/scripts")
import ruler_sweep

SRC = "SaylorTwift/RULER-32768-Qwen2.5-3B-tokenizer"


def load_32k(name, config, split="test"):
    assert config == "32768", f"wrapper only serves 32768, got {config}"
    parts = []
    for task in ["vt", "fwe", "niah_multikey_3"]:
        ds = load_dataset(SRC, split=task)
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
    ruler_sweep.main()
