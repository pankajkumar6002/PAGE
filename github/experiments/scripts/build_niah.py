"""Build a minimal NIAH-MultiKey-style dataset.

Each example: a synthetic long context with K (key, value) pairs hidden in
filler distractor text. The question asks for the value of one specific key.
Correctness is exact match on the integer answer.

Output: JSONL at experiments/data/niah_multikey.jsonl with fields
{id, context, question, answer, keys, values, target_idx, n_filler_tokens}.

Defaults: 200 examples, 4 keys per example, ~16K target context tokens
(measured in approximate chars / 4 since the tokenizer is not loaded here;
the run script re-measures with the actual tokenizer).
"""
import argparse
import json
import os
import random
import string

FILLER_SENTENCES = [
    "The quarterly report indicates a steady growth across all departments.",
    "Researchers continue to investigate the effects of climate on regional ecosystems.",
    "The team prepared for the upcoming presentation with great care.",
    "A long walk in the morning often clears the mind for the day ahead.",
    "Engineers debated the merits of several competing architectures.",
    "The library acquired several rare manuscripts from a private donor.",
    "Most participants agreed that the workshop exceeded their expectations.",
    "Local authorities announced a new initiative to improve public transit.",
    "The volunteers worked through the weekend to finish the renovation.",
    "Historians often disagree about the precise timeline of these events.",
    "Pedestrians crossed the bridge at a steady, measured pace.",
    "The chef adjusted the recipe slightly to account for seasonal ingredients.",
    "Investors expressed cautious optimism following the announcement.",
    "Students gathered in the courtyard to discuss the lecture afterward.",
    "The mountain trail wound through forests of pine and fir.",
    "Curators arranged the new exhibit with attention to chronological detail.",
    "The committee tabled the motion until additional data could be gathered.",
    "Travelers found the alternate route surprisingly scenic and quiet.",
    "Programmers refactored the legacy module over several days of effort.",
    "The bakery introduced a new sourdough that quickly became popular.",
]

CHARS_PER_TOKEN = 4  # rough estimate; the run script remeasures with tokenizer


def rand_key(rng):
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(6))


def rand_value(rng):
    return rng.randint(100_000, 999_999)


def build_example(rng, n_keys, target_tokens):
    keys = [rand_key(rng) for _ in range(n_keys)]
    while len(set(keys)) < n_keys:
        keys = [rand_key(rng) for _ in range(n_keys)]
    values = [rand_value(rng) for _ in range(n_keys)]

    needles = [f"The magic number for {k} is {v}." for k, v in zip(keys, values)]
    target_idx = rng.randint(0, n_keys - 1)
    question = f"What is the magic number for {keys[target_idx]}?"
    answer = str(values[target_idx])

    target_chars = target_tokens * CHARS_PER_TOKEN
    needle_chars = sum(len(n) + 1 for n in needles)
    filler_chars_needed = max(0, target_chars - needle_chars - len(question) - 200)

    filler_lines = []
    current = 0
    while current < filler_chars_needed:
        s = rng.choice(FILLER_SENTENCES)
        filler_lines.append(s)
        current += len(s) + 1

    n_filler = len(filler_lines)
    insert_positions = sorted(rng.sample(range(n_filler), n_keys))
    for pos, needle in zip(insert_positions, needles):
        filler_lines.insert(pos, needle)

    context = "\n".join(filler_lines)
    return {
        "context": context,
        "question": question,
        "answer": answer,
        "keys": keys,
        "values": [str(v) for v in values],
        "target_idx": target_idx,
        "n_filler_tokens_est": len(context) // CHARS_PER_TOKEN,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="experiments/data/niah_multikey.jsonl")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--n_keys", type=int, default=4)
    p.add_argument("--target_tokens", type=int, default=16_000)
    p.add_argument("--seed", type=int, default=20260602)
    args = p.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for i in range(args.n):
            ex = build_example(rng, args.n_keys, args.target_tokens)
            ex["id"] = i
            f.write(json.dumps(ex) + "\n")
    print(f"wrote {args.n} examples to {args.out}")
    print(f"approx tokens per example: ~{args.target_tokens}")


if __name__ == "__main__":
    main()
