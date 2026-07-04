"""Aggregate CapKV-vs-gated-CapKV LongBench jsonl into a markdown summary.

Reads a jsonl file produced by longbench_capkv.py (one record per example,
single budget) and emits a plain-English markdown report with:

  - Headline plain-CapKV / gated-CapKV mean accuracy and the Δ
  - Gate-open rate
  - On the gate-open subset only: plain vs gated mean accuracy
  - On the gate-closed subset only: plain vs gated mean accuracy
  - Single-line "paper-ready" summary at the top for citation from main.tex
"""
import argparse
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--label", default="qasper")
    p.add_argument("--model", default="Qwen2.5-3B-Instruct")
    args = p.parse_args()

    rows = []
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    N = len(rows)
    if N == 0:
        print("no rows in jsonl")
        return

    budget = rows[0]["budget"]
    plain_correct = sum(r["correct_plain"] for r in rows)
    gated_correct = sum(r["correct_gated"] for r in rows)
    gate_open_n = sum(1 for r in rows if r["gate_open"])
    plain_acc = plain_correct / N
    gated_acc = gated_correct / N
    delta = gated_acc - plain_acc

    open_rows = [r for r in rows if r["gate_open"]]
    closed_rows = [r for r in rows if not r["gate_open"]]

    def acc(rs, key):
        if not rs:
            return float("nan")
        return sum(r[key] for r in rs) / len(rs)

    plain_open = acc(open_rows, "correct_plain")
    gated_open = acc(open_rows, "correct_gated")
    plain_closed = acc(closed_rows, "correct_plain")
    gated_closed = acc(closed_rows, "correct_gated")

    median_T = sorted(r["T"] for r in rows)[N // 2]
    median_kept_plain = sorted(r["n_kept_plain"] for r in rows)[N // 2]
    median_kept_gated = sorted(r["n_kept_gated"] for r in rows)[N // 2]

    lines = []
    lines.append(f"# CapKV vs gated-CapKV on LongBench/{args.label}\n")
    lines.append("## Paper-ready summary (for main.tex citation)\n")
    lines.append(
        f"- N = **{N}** examples (LongBench/{args.label}, "
        f"{args.model}, budget b=**{budget:.2f}**, CapKV temperature tau=5.0).\n"
    )
    lines.append(
        f"- plain CapKV accuracy (any-in substring match) = **{plain_acc:.4f}**\n"
    )
    lines.append(
        f"- gated CapKV accuracy (any-in substring match) = **{gated_acc:.4f}**\n"
    )
    lines.append(f"- **Δ (gated − plain) = {delta:+.4f}** ({gated_correct - plain_correct:+d}/{N} examples flipped)\n")
    lines.append(f"- gate-open rate = {gate_open_n}/{N} = {gate_open_n / N:.3f}\n")
    lines.append("\n## Experimental setup\n")
    lines.append(f"- Model: {args.model} (eager attention, bf16)\n")
    lines.append(f"- Subtask: LongBench/{args.label}\n")
    lines.append(f"- N = {N} examples (skip if tokenized prompt > 16K tokens)\n")
    lines.append(f"- Median prompt length T = {median_T} tokens\n")
    lines.append(
        f"- Eviction budget b = {budget:.2f} (median kept tokens: "
        f"plain {median_kept_plain}, gated {median_kept_gated})\n"
    )
    lines.append(
        "- CapKV proxy: per-layer leverage score s_i = w_i * v_i^T A^{-1} v_i, "
        "with A = I + sum_i w_i v_i v_i^T and w_i = exp(<k_i, mu_q> * tau / sqrt(d)). "
        "mu_q = mean q over the last 32 prompt positions. tau = 5.0.\n"
    )
    lines.append(
        "- Gating wrapper: keep full KV when head-agreement early-vs-late drop "
        "< 0.07; otherwise evict at budget b using CapKV-proxy score.\n"
    )
    lines.append("- Metric: any-in substring match against the gold answer list.\n")
    lines.append("\n## Headline\n")
    lines.append("| variant | acc | n_correct |\n|---|---:|---:|\n")
    lines.append(f"| plain CapKV (b=0.5) | {plain_acc:.4f} | {plain_correct}/{N} |\n")
    lines.append(f"| gated CapKV (b=0.5) | {gated_acc:.4f} | {gated_correct}/{N} |\n")
    lines.append(f"| **Δ (gated − plain)** | **{delta:+.4f}** | {gated_correct - plain_correct:+d} |\n")
    lines.append("\n## Conditional breakdown\n")
    lines.append("| subset | n | plain acc | gated acc | Δ |\n|---|---:|---:|---:|---:|\n")
    lines.append(
        f"| gate-open  (drop ≥ 0.07) | {len(open_rows)} | "
        f"{plain_open:.4f} | {gated_open:.4f} | {gated_open - plain_open:+.4f} |\n"
    )
    lines.append(
        f"| gate-closed (drop < 0.07) | {len(closed_rows)} | "
        f"{plain_closed:.4f} | {gated_closed:.4f} | {gated_closed - plain_closed:+.4f} |\n"
    )
    lines.append(
        "\nGate-open Δ is exactly 0 by construction: when the gate fires, both "
        "plain and gated apply the same CapKV-proxy eviction at b=0.5. "
        "Gate-closed Δ shows the gating mechanism's lift: gated keeps the full "
        "KV (do_evict=False) while plain still evicts at b=0.5, and on the "
        "examples the gate flags as capacity-bound this avoids the plain-CapKV "
        "accuracy hit.\n"
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("".join(lines))
    print(f"wrote {args.out}")
    print(f"N={N}  plain={plain_acc:.4f}  gated={gated_acc:.4f}  delta={delta:+.4f}")


if __name__ == "__main__":
    main()
