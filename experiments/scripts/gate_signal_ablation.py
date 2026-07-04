"""Gate-signal ablation: is the head-agreement drop D the only cheap prefill
statistic that separates capacity-bound (NIAH-MK3/MK2) from dilution-prone
(vt/fwe/qa_1/niah_multivalue) inputs?

One prefill pass per input (same machinery as head_agreement_probe.py /
gated_eviction.py). From the SAME prefill attentions/keys we compute, per input:

  (a) drop_D          : head-agreement drop, the paper's gate signal — reuses
                        compute_drop_from_attentions verbatim (Jaccard top-32
                        key sets per head pair, early [0,L/3) minus late
                        [2L/3,L) layer-bin means).
  (b) entropy_norm    : mean attention entropy over the obs-window queries,
                        normalized by log(T), averaged over layers/heads.
  (c) topk_mass_share : fraction of total attention mass in the top-32
                        SnapKV-scored positions (pool_score 'snapkv').
  (d) max_share       : fraction of total attention mass at the single
                        highest-scored position.
  (e) keynorm_disp    : key-norm dispersion — std/mean of per-position key L2
                        norms, averaged over layers and kv-heads.
  (f) entropy_drop    : early-vs-late ENTROPY drop — the exact entropy
                        analogue of D (same layer bins), to test whether the
                        early-late contrast or the agreement measure matters.

Run (probe):
  .venv/bin/python experiments/scripts/gate_signal_ablation.py --gpu 3
Analysis only (re-uses existing jsonl):
  .venv/bin/python experiments/scripts/gate_signal_ablation.py --analyze_only
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gated_eviction import (  # noqa: E402
    build_prompt,
    compute_drop_from_attentions,
    pool_score,
)

RESULTS_DIR = "/home/smlab/projects/eff-nn/experiments/results"
SNIPPET_DIR = "/home/smlab/projects/eff-nn/paper/snippets"

CAPACITY_ANCHOR = "niah_multikey_3"
CAPACITY_PLUS = ["niah_multikey_3", "niah_multikey_2"]
DILUTION = ["vt", "fwe", "qa_1", "niah_multivalue"]
MK_GRADIENT = ["niah_multikey_1", "niah_multikey_2", "niah_multikey_3"]

SIGNALS = [
    ("drop_D", "head-agreement drop $D$ (paper)"),
    ("entropy_norm", "mean attention entropy (norm.)"),
    ("topk_mass_share", "top-32 attention mass share"),
    ("max_share", "max single-position share"),
    ("keynorm_disp", "key-norm dispersion (std/mean)"),
    ("entropy_drop", "early$-$late entropy drop"),
]


def compute_signals(attentions, past, obs_window, top_k, T, device):
    """All six signals from one prefill's attentions + cached keys."""
    L = len(attentions)
    third = max(1, L // 3)

    # (a) the paper's signal, exact function reused.
    drop, _per_layer = compute_drop_from_attentions(attentions, obs_window, top_k)

    # (b) + (f) normalized attention entropy per layer over obs-window queries.
    log_t = math.log(max(T, 2))
    per_layer_ent = []
    for a in attentions:
        w = a[0, :, -obs_window:, :].float()
        p = w.clamp(min=1e-12)
        ent = -(p * p.log()).sum(dim=-1)  # [H, W]
        per_layer_ent.append((ent.mean() / log_t).item())
    entropy_norm = sum(per_layer_ent) / L
    entropy_drop = (
        sum(per_layer_ent[:third]) / third - sum(per_layer_ent[L - third:]) / third
    )

    # (c) + (d) from the SnapKV pooled score (mean attention from the
    # obs-window queries, over layers/heads) — the same score the evictor uses.
    score = pool_score(attentions, "snapkv", obs_window, T, device).float()
    total = score.sum().item()
    topk_mass_share = score.topk(min(top_k, T)).values.sum().item() / total
    max_share = score.max().item() / total

    # (e) key-norm dispersion from the prefill KV cache.
    disps = []
    for layer in past.layers:
        norms = layer.keys[0].float().norm(dim=-1)  # [H_kv, T]
        disps.append((norms.std(dim=-1) / norms.mean(dim=-1)).mean().item())
    keynorm_disp = sum(disps) / len(disps)

    return {
        "drop_D": drop,
        "entropy_norm": entropy_norm,
        "topk_mass_share": topk_mass_share,
        "max_share": max_share,
        "keynorm_disp": keynorm_disp,
        "entropy_drop": entropy_drop,
    }


def run_probe(args):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()

    ds = load_dataset("simonjegou/ruler", args.config, split="test")
    tasks = args.tasks.split(",")
    rows_to_run = []
    for task in tasks:
        sub = ds.filter(lambda r: r["task"] == task)
        sub = sub.select(range(min(args.n_per_task, len(sub))))
        for i, r in enumerate(sub):
            rows_to_run.append((i, r))
    print(f"probing {len(rows_to_run)} examples across {tasks}", flush=True)

    out_f = open(args.out, "w")
    for ex_i, (task_idx, ex) in enumerate(rows_to_run):
        prompt = build_prompt(tokenizer, ex)
        ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        T = ids.shape[1]

        with torch.no_grad():
            out = model(
                input_ids=ids,
                output_attentions=True,
                use_cache=True,
                return_dict=True,
            )
        sig = compute_signals(
            out.attentions, out.past_key_values, args.obs_window, args.top_k, T, device
        )
        rec = {"task": ex["task"], "id": task_idx, "T": T, **sig}
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        del out
        torch.cuda.empty_cache()
        if (ex_i + 1) % 10 == 0:
            print(f"  [{ex_i + 1}/{len(rows_to_run)}]", flush=True)
    out_f.close()
    print("probe done", flush=True)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def auc(neg, pos):
    """Rank-based AUC = P(pos > neg) (+ 0.5 * ties). neg/pos: lists of floats."""
    if not neg or not pos:
        return float("nan")
    ranked = sorted([(v, 0) for v in neg] + [(v, 1) for v in pos])
    # Mann-Whitney with midranks.
    ranks = {}
    i = 0
    vals = [v for v, _ in ranked]
    while i < len(vals):
        j = i
        while j < len(vals) and vals[j] == vals[i]:
            j += 1
        midrank = (i + j + 1) / 2.0  # 1-based midrank
        for k in range(i, j):
            ranks[k] = midrank
        i = j
    r_pos = sum(ranks[k] for k, (_, lab) in enumerate(ranked) if lab == 1)
    n1, n0 = len(pos), len(neg)
    u = r_pos - n1 * (n1 + 1) / 2.0
    return u / (n1 * n0)


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j < len(order) and v[order[j]] == v[order[i]]:
                j += 1
            mid = (i + j + 1) / 2.0
            for k in range(i, j):
                r[order[k]] = mid
            i = j
        return r

    return pearson(rank(xs), rank(ys))


def fmt(x, nd=3):
    return f"{x:+.{nd}f}" if x < 0 or True else f"{x:.{nd}f}"


def analyze(args):
    rows = []
    with open(args.out) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    tasks = args.tasks.split(",")
    by_task = {t: [r for r in rows if r["task"] == t] for t in tasks}
    dil_rows = [r for t in DILUTION for r in by_task.get(t, [])]

    md = []
    md.append("# Gate-signal ablation: is the head-agreement drop D uniquely effective?")
    md.append("")
    md.append(f"Model: {args.model}; RULER {args.config}; "
              f"N={args.n_per_task}/task; tasks: {', '.join(tasks)}. "
              f"One eager prefill per input; all six signals computed from the "
              f"same prefill attentions (last {args.obs_window} queries) and "
              f"cached keys. Reproduce: `.venv/bin/python "
              f"experiments/scripts/gate_signal_ablation.py` "
              f"(add `--analyze_only` to skip the GPU probe).")
    md.append("")
    md.append("Capacity-bound anchor: `niah_multikey_3` (+`niah_multikey_2`); "
              "dilution-prone set: `vt, fwe, qa_1, niah_multivalue`. AUCs are "
              "*oriented* per signal so that the capacity-bound class is on "
              "the low side (orientation shown as the 'MK3 side' column); "
              "0.5 = chance either way.")
    md.append("")

    # Per-task means table.
    md.append("## Per-task means (± std)")
    md.append("")
    header = "| signal | " + " | ".join(tasks) + " |"
    md.append(header)
    md.append("|---|" + "---:|" * len(tasks))
    task_mean = {}
    for key, label in SIGNALS:
        cells = []
        for t in tasks:
            vs = [r[key] for r in by_task[t]]
            m = sum(vs) / len(vs)
            sd = math.sqrt(sum((v - m) ** 2 for v in vs) / len(vs))
            task_mean[(key, t)] = m
            cells.append(f"{m:+.4f} ± {sd:.4f}")
        md.append(f"| {key} | " + " | ".join(cells) + " |")
    md.append("")

    # Main summary table.
    summary = []
    for key, label in SIGNALS:
        mk3_vals = [r[key] for r in by_task[CAPACITY_ANCHOR]]
        mk32_vals = [r[key] for t in CAPACITY_PLUS for r in by_task[t]]
        dil_vals = [r[key] for r in dil_rows]
        mk3_mean = sum(mk3_vals) / len(mk3_vals)
        dil_mean = sum(dil_vals) / len(dil_vals)
        # Orient so capacity-bound is LOW.
        sign = 1.0 if mk3_mean <= dil_mean else -1.0
        side = "low" if sign > 0 else "high"

        dil_means = [task_mean[(key, t)] for t in DILUTION]
        sep_mk3 = all(sign * mk3_mean < sign * m for m in dil_means)
        mk2_mean = task_mean[(key, "niah_multikey_2")]
        sep_mk32 = sep_mk3 and all(sign * mk2_mean < sign * m for m in dil_means)
        # Per-input strict threshold check (task-level uses means, per prompt).
        auc_mk3 = auc([sign * v for v in mk3_vals], [sign * v for v in dil_vals])
        auc_mk32 = auc([sign * v for v in mk32_vals], [sign * v for v in dil_vals])
        # MK gradient monotonicity: oriented mean should DECREASE MK1->MK2->MK3
        g = [sign * task_mean[(key, t)] for t in MK_GRADIENT]
        monotone = g[0] > g[1] > g[2]
        # Correlation with D across all inputs.
        d_all = [r["drop_D"] for r in rows]
        s_all = [r[key] for r in rows]
        r_p = pearson(s_all, d_all)
        r_s = spearman(s_all, d_all)
        summary.append(dict(key=key, label=label, side=side, sep_mk3=sep_mk3,
                            sep_mk32=sep_mk32, auc_mk3=auc_mk3,
                            auc_mk32=auc_mk32, monotone=monotone,
                            pearson=r_p, spearman=r_s,
                            grad=[task_mean[(key, t)] for t in MK_GRADIENT]))

    md.append("## Summary: signal × separation criteria")
    md.append("")
    md.append("| signal | MK3 side | task-level sep (MK3 < all 4 dil.) | "
              "task-level sep (MK3 & MK2) | AUC MK3 vs dil. | AUC MK3+MK2 vs dil. | "
              "monotone on MK1→MK2→MK3 | Pearson r with D | Spearman ρ with D |")
    md.append("|---|---|---|---|---:|---:|---|---:|---:|")
    for s in summary:
        md.append(
            f"| {s['key']} | {s['side']} | {'**yes**' if s['sep_mk3'] else 'no'} | "
            f"{'**yes**' if s['sep_mk32'] else 'no'} | {s['auc_mk3']:.3f} | "
            f"{s['auc_mk32']:.3f} | {'**yes**' if s['monotone'] else 'no'} | "
            f"{s['pearson']:+.3f} | {s['spearman']:+.3f} |"
        )
    md.append("")

    md.append("## MK distractor gradient (raw task means, MK1 → MK2 → MK3)")
    md.append("")
    md.append("| signal | MK1 | MK2 | MK3 | monotone (oriented) |")
    md.append("|---|---:|---:|---:|---|")
    for s in summary:
        g = s["grad"]
        md.append(f"| {s['key']} | {g[0]:+.4f} | {g[1]:+.4f} | {g[2]:+.4f} | "
                  f"{'yes' if s['monotone'] else 'no'} |")
    md.append("")

    md_path = os.path.join(RESULTS_DIR, "gate_signal_ablation.md")
    tex_path = os.path.join(SNIPPET_DIR, "signal_ablation_table.tex")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {md_path} (interpretation section appended separately)")

    # TeX snippet.
    def yn(b):
        return r"\yes" if b else r"\no"

    tex = []
    tex.append("% ============================================================================")
    tex.append("% signal_ablation_table.tex — gate-signal ablation (same-inputs comparison of")
    tex.append("% the head-agreement drop D against five other cheap prefill statistics).")
    tex.append(f"% Generated from experiments/results/{os.path.basename(args.out)} by")
    tex.append("% experiments/scripts/gate_signal_ablation.py. Requires: booktabs,")
    tex.append(r"% amssymb (for \checkmark; both already loaded by paper/main.tex).")
    tex.append(r"% Define once in the preamble if not already present:")
    tex.append(r"%   \newcommand{\yes}{\checkmark}  \newcommand{\no}{--}")
    tex.append("% ============================================================================")
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Gate-signal ablation on Qwen2.5-1.5B-Instruct (RULER 4K, $N=50$ per")
    tex.append(r"task, one shared prefill per input). Each cheap prefill statistic is tested")
    tex.append(r"for separating the capacity-bound anchor (NIAH-MK3, and MK3+MK2) from the")
    tex.append(r"dilution-prone tasks (vt, fwe, qa\_1, multivalue). ``Sep.'' = a single")
    tex.append(r"threshold puts the capacity-bound task mean(s) strictly on one side of all")
    tex.append(r"four dilution task means; AUCs are per-input and oriented per signal")
    tex.append(r"(0.5 = chance); ``MK mono.'' = task means are monotone along the")
    tex.append(r"MK1$\to$MK2$\to$MK3 distractor gradient; $r$ = Pearson correlation with $D$.}")
    tex.append(r"\label{tab:signal-ablation}")
    tex.append(r"\begin{tabular}{lcccccc}")
    tex.append(r"\toprule")
    tex.append(r"Signal & Sep.\ MK3 & Sep.\ MK3+MK2 & AUC MK3 & AUC MK3+MK2 & MK mono.\ & $r$ w/ $D$ \\")
    tex.append(r"\midrule")
    for s in summary:
        label = s["label"]
        row = (f"{label} & {yn(s['sep_mk3'])} & {yn(s['sep_mk32'])} & "
               f"{s['auc_mk3']:.3f} & {s['auc_mk32']:.3f} & {yn(s['monotone'])} & "
               f"${s['pearson']:+.2f}$ \\\\")
        if s["key"] == "drop_D":
            row = row.replace(f"{s['auc_mk3']:.3f}", r"\textbf{" + f"{s['auc_mk3']:.3f}" + "}")
            row = row.replace(f"{s['auc_mk32']:.3f}", r"\textbf{" + f"{s['auc_mk32']:.3f}" + "}")
        tex.append(row)
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    with open(tex_path, "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"wrote {tex_path}")

    # Console summary for the caller.
    print("\n" + "\n".join(md[md.index("## Summary: signal × separation criteria"):]))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="4096")
    p.add_argument("--tasks", default="niah_multikey_3,niah_multikey_2,niah_multikey_1,"
                                      "vt,fwe,qa_1,niah_multivalue")
    p.add_argument("--n_per_task", type=int, default=50)
    p.add_argument("--obs_window", type=int, default=32)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "gate_signal_ablation.jsonl"))
    p.add_argument("--analyze_only", action="store_true")
    args = p.parse_args()

    if not args.analyze_only:
        run_probe(args)
    analyze(args)


if __name__ == "__main__":
    main()
