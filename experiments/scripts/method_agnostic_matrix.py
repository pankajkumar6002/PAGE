"""Compile the full method-agnostic matrix.

Reads all gated_{policy}_{model}_{context}.jsonl files and outputs a
3-models × 4-baseline matrix showing plain accuracy, gated accuracy, and
delta.
"""
import json
import os
from collections import defaultdict


def headline(path):
    """Return (plain_mean, gated_mean, delta) over eviction budgets."""
    if not os.path.exists(path):
        return None, None, None
    rows = [json.loads(l) for l in open(path)]
    if not rows:
        return None, None, None
    budgets = sorted({r["budget"] for r in rows}, reverse=True)
    eviction_b = [b for b in budgets if b < 1.0]
    if not eviction_b:
        return None, None, None
    plain = []
    gated = []
    for r in rows:
        if r["budget"] in eviction_b:
            plain.append(r["correct_plain"])
            gated.append(r["correct_gated"])
    if not plain:
        return None, None, None
    p_mean = sum(plain) / len(plain)
    g_mean = sum(gated) / len(gated)
    return p_mean, g_mean, g_mean - p_mean


def main():
    base = "experiments/results"

    models = [
        ("Qwen 1.5B", "qwen15b"),
        ("Qwen 3B", "qwen3b"),
        ("Qwen 14B", "qwen14b"),
        ("Mistral 7B", "mistral7b"),
    ]
    policies = ["snapkv", "h2o", "streamingllm", "pyramidkv"]

    # Map model + policy → file path. SnapKV uses the older gated_4k_<model> filename.
    def path_for(model_slug, policy):
        if policy == "snapkv":
            return f"{base}/gated_4k_{model_slug}.jsonl"
        return f"{base}/gated_{policy}_{model_slug}_4k.jsonl"

    print("# Method-agnostic gating matrix at 4K (mixed RULER suite, τ=0.07)\n")
    print("Plain X mean / Gated X mean / Δ over eviction budgets.\n")

    print(f"| Model |", end="")
    for p in policies:
        print(f" {p} plain | {p} gated | Δ |", end="")
    print()
    print(f"|---|", end="")
    for _ in policies:
        print("---:|---:|---:|", end="")
    print()

    summary_deltas = {}
    for model_name, slug in models:
        print(f"| {model_name} |", end="")
        for p in policies:
            file = path_for(slug, p)
            plain, gated, delta = headline(file)
            if plain is None:
                print(" — | — | — |", end="")
            else:
                print(f" {plain:.3f} | {gated:.3f} | **{delta:+.3f}** |", end="")
                summary_deltas[(model_name, p)] = delta
        print()

    # Per-model summary: best plain baseline vs gated-X
    print(f"\n## Best plain vs gated-X (headline reframed metric)\n")
    print(f"| Model | Best plain | Best gated | Δ best-plain → best-gated |")
    print(f"|---|---:|---:|---:|")
    for model_name, slug in models:
        plains = []
        gateds = []
        for p in policies:
            pl, gd, _ = headline(path_for(slug, p))
            if pl is not None:
                plains.append(pl)
            if gd is not None:
                gateds.append(gd)
        if not plains:
            continue
        best_plain = max(plains)
        best_gated = max(gateds)
        print(f"| {model_name} | {best_plain:.3f} | {best_gated:.3f} | **{best_gated - best_plain:+.3f}** |")

    # Average Δ per policy
    print(f"\n## Average Δ per policy across all measured cells\n")
    for p in policies:
        ds = [summary_deltas[(m, p)] for m, _ in models if (m, p) in summary_deltas]
        if ds:
            print(f"- **{p}**: mean Δ = +{sum(ds)/len(ds):.3f} ({len(ds)} cells)")


if __name__ == "__main__":
    main()
