#!/usr/bin/env python3
"""PAGE review response: reproducible, read-only checks and analyses A0-A4, A6 and A7.

Reads the released per-input logs, prints markdown tables to stdout, writes
nothing. Written independently of the repo's gatelib.py on purpose, so that it
can catch errors there as well as here.

  A0  Reproduce published values, then re-check the errors found in the paper.
  A1  Does D predict per-input harm? (S-Q1, S-W2)
  A2  Fallback vs selection: PAGE against a random gate at its open rate. (S-W2, R-Q1)
  A3  Matched achieved cache, including harm. (S-W2)
  A4  How -0.69 was selected, and threshold stability. (R-Q2, G-W2)
  A6  Beyond RULER: LongBench and extra RULER tasks, with official metrics. (S-Q2, G-Q1)
  A7  Task-level ceiling. (corrects "no task-level policy can do")

Status of A1: a first pass was looked at before this script existed, so A1 is
exploratory and is not pre-registered.

Usage:
  python3 verify_2026-11-07.py                 # everything
  python3 verify_2026-11-07.py --sections A0,A1
  python3 verify_2026-11-07.py --reps 500      # fewer bootstrap replicates

Exit status is non-zero if any assertion against a published value fails.
"""
import argparse
import collections
import difflib
import json
import os
import re
import string
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
RES = os.environ.get("PAGE_RESULTS", str(REPO / "experiments" / "results"))
TAU = 0.07
THETA_Z = -0.69
MK3 = "niah_multikey_3"
SUITE = [MK3, "vt", "fwe", "qa_1"]
POLICIES = ["snapkv", "h2o", "streamingllm", "pyramidkv"]
POLICY_NAME = {"snapkv": "SnapKV", "h2o": "H2O", "streamingllm": "StreamingLLM", "pyramidkv": "PyramidKV"}
MODELS = [("qwen15b", "Qwen2.5-1.5B"), ("qwen3b", "Qwen2.5-3B"), ("qwen14b", "Qwen2.5-14B"), ("mistral7b", "Mistral-7B")]
SEED = 20261107

FAILS = []


# ---------------------------------------------------------------- helpers
def read_jsonl(name):
    rows, bad = [], 0
    with open(os.path.join(RES, name)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad


def check(name, got, want, tol=0.0011):
    ok = abs(got - want) <= tol
    if not ok:
        FAILS.append(name)
        print(f"FAIL {name}: got {got:.4f}, published {want:.4f}")
    return ok


def check_many(label, pairs, tol=0.0011):
    n_ok = sum(check(f"{label}/{k}", g, w, tol) for k, g, w in pairs)
    print(f"- {label}: {n_ok}/{len(pairs)} reproduce (tolerance {tol})")


def rank_avg(a):
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    first = np.r_[True, s[1:] != s[:-1]]
    dense = np.cumsum(first) - 1
    cnt = np.bincount(dense)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    out = np.empty(len(a))
    out[order] = avg[dense]
    return out


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie). NaN if either side is empty."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rank_avg(np.r_[pos, neg])
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(rank_avg(x), rank_avg(y))[0, 1])


def ci(values, lo=2.5, hi=97.5):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi)))


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def s3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.3f}"


def ci3(c, signed=False):
    fmt = s3 if signed else f3
    return f"[{fmt(c[0])}, {fmt(c[1])}]"


def table(tid, title, headers, rows, note=None):
    print(f"\n### {tid}: {title}\n")
    print("| " + " | ".join(headers) + " |")
    print("|" + "---|" * len(headers))
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    if note:
        print(f"\n{note}")


def bname(b):
    return {0.0625: "1/16", 0.125: "1/8", 0.25: "1/4", 0.375: "3/8", 0.5: "1/2", 0.625: "5/8", 0.75: "3/4", 0.875: "7/8"}.get(b, f"{b:g}")


def bh(pvals, q=0.05):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    passed = np.zeros(m, bool)
    k = 0
    for rank, idx in enumerate(order, 1):
        if p[idx] <= q * rank / m:
            k = rank
    for rank, idx in enumerate(order, 1):
        if rank <= k:
            passed[idx] = True
    return passed


class Cell:
    """One gated log: inputs by budget, with the gate recomputed from `drop`."""

    def __init__(self, fname, label=None, tasks=None):
        rows, self.bad = read_jsonl(fname)
        by = collections.defaultdict(dict)
        for r in rows:
            if tasks is not None and r["task"] not in tasks:
                continue
            by[(r["task"], r["id"])][float(r["budget"])] = r
        self.fname, self.label = fname, label or fname
        self.budgets = sorted({b for v in by.values() for b in v if b < 1.0})
        keys = [k for k in sorted(by) if 1.0 in by[k] and all(b in by[k] for b in self.budgets)]
        self.n_dropped = len(by) - len(keys)
        self.keys = keys
        self.task = np.array([k[0] for k in keys])
        self.id = np.array([k[1] for k in keys])
        self.drop = np.array([by[k][1.0]["drop"] for k in keys], float)
        self.T = np.array([by[k][1.0]["T"] for k in keys], float)
        self.full = np.array([bool(by[k][1.0]["correct_plain"]) for k in keys])
        self.plain = np.array([[bool(by[k][b]["correct_plain"]) for b in self.budgets] for k in keys])
        self.kept = np.array([[by[k][b]["n_kept_plain"] / by[k][1.0]["T"] for b in self.budgets] for k in keys], float)
        self.by = by

    @property
    def n(self):
        return len(self.keys)

    def open(self, tau=TAU):
        return self.drop >= tau

    def gated(self, tau=TAU):
        return np.where(self.open(tau)[:, None], self.plain, self.full[:, None])

    def bi(self, b):
        return self.budgets.index(b)

    def task_groups(self):
        return [np.where(self.task == t)[0] for t in np.unique(self.task)]


_CACHE = {}


def cell(fname, label=None, tasks=None):
    key = (fname, tuple(tasks) if tasks else None)
    if key not in _CACHE:
        _CACHE[key] = Cell(fname, label, tasks)
    return _CACHE[key]


def matrix_file(policy, slug):
    return f"gated_4k_{slug}.jsonl" if policy == "snapkv" else f"gated_{policy}_{slug}_4k.jsonl"


def strat_idx(rng, groups):
    return np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])


# ---------------------------------------------------------------- A0
PUB_T2 = {
    ("snapkv", "qwen15b"): (0.438, 0.559, 0.000), ("snapkv", "qwen3b"): (0.580, 0.839, 0.101),
    ("snapkv", "qwen14b"): (0.709, 0.853, 0.000), ("snapkv", "mistral7b"): (0.623, 0.810, 0.015),
    ("h2o", "qwen15b"): (0.196, 0.357, 0.000), ("h2o", "qwen3b"): (0.339, 0.642, 0.098),
    ("h2o", "qwen14b"): (0.668, 0.868, 0.000), ("h2o", "mistral7b"): (0.400, 0.636, 0.020),
    ("streamingllm", "qwen15b"): (0.033, 0.195, 0.000), ("streamingllm", "qwen3b"): (0.052, 0.482, 0.263),
    ("streamingllm", "qwen14b"): (0.080, 0.330, 0.000), ("streamingllm", "mistral7b"): (0.102, 0.365, 0.053),
    ("pyramidkv", "qwen15b"): (0.417, 0.570, 0.000), ("pyramidkv", "qwen3b"): (0.536, 0.865, 0.144),
    ("pyramidkv", "qwen14b"): (0.632, 0.860, 0.000), ("pyramidkv", "mistral7b"): (0.593, 0.823, 0.020),
}
PUB_T14 = {
    ("snapkv", "qwen15b"): 0.121, ("snapkv", "qwen3b"): 0.183, ("snapkv", "qwen14b"): 0.143, ("snapkv", "mistral7b"): 0.195,
    ("h2o", "qwen15b"): 0.162, ("h2o", "qwen3b"): 0.230, ("h2o", "qwen14b"): 0.200, ("h2o", "mistral7b"): 0.243,
    ("streamingllm", "qwen15b"): 0.163, ("streamingllm", "qwen3b"): 0.232, ("streamingllm", "qwen14b"): 0.250, ("streamingllm", "mistral7b"): 0.247,
    ("pyramidkv", "qwen15b"): 0.152, ("pyramidkv", "qwen3b"): 0.221, ("pyramidkv", "qwen14b"): 0.228, ("pyramidkv", "mistral7b"): 0.240,
}
PUB_T29 = {
    (MK3, 0.0625): (0.026, 0.750), (MK3, 0.125): (0.026, 0.744), (MK3, 0.25): (0.024, 0.724), (MK3, 0.5): (0.018, 0.546),
    ("fwe", 0.0625): (0.236, 0.414), ("fwe", 0.125): (0.162, 0.298), ("fwe", 0.25): (0.118, 0.224), ("fwe", 0.5): (0.064, 0.116),
    ("qa_1", 0.0625): (0.110, 0.110), ("qa_1", 0.125): (0.064, 0.064), ("qa_1", 0.25): (0.030, 0.030), ("qa_1", 0.5): (0.010, 0.010),
    ("vt", 0.0625): (0.684, 0.694), ("vt", 0.125): (0.108, 0.110), ("vt", 0.25): (0.034, 0.036), ("vt", 0.5): (0.026, 0.026),
}
PUB_T3 = {"drop_D": 1.000, "entropy_norm": 0.508, "topk_mass_share": 0.784, "max_share": 0.723, "keynorm_disp": 0.570, "entropy_drop": 0.805}
PUB_T3_MK23 = {"drop_D": 0.859, "entropy_norm": 0.670, "topk_mass_share": 0.574, "max_share": 0.708, "keynorm_disp": 0.560, "entropy_drop": 0.575}
PUB_T6 = {"signal_ablation_qwen14b_4k.jsonl": 0.899, "signal_ablation_llama31_4k.jsonl": 0.741, "signal_ablation_mistral7b_16k.jsonl": 0.978}
PUB_T12 = {
    "lcc": ("longbench_realistic_lcc_qwen14b.jsonl", 96, 0.292, 0.64, 0.031, 0.0543, 0.250, 0.146),
    "repobench-p": ("longbench_realistic_repobench-p_qwen14b.jsonl", 82, 0.317, 0.42, 0.061, 0.0541, 0.256, 0.049),
    "hotpotqa": ("longbench_realistic_hotpotqa_qwen14b.jsonl", 100, 0.560, 0.12, 0.020, 0.0973, 0.860, 0.000),
}
PUB_T24 = {"qwen15b": (0.75, 0.297, 0.437, 0.584), "qwen3b": (0.49, 0.541, 0.632, 0.728), "qwen14b": (0.75, 0.297, 0.437, 0.584), "mistral7b": (0.72, 0.327, 0.462, 0.602)}
PUB_T30 = {
    ("qwen15b", "snapkv"): (0.093, 0.151), ("qwen15b", "h2o"): (0.126, 0.199), ("qwen3b", "snapkv"): (0.225, 0.295),
    ("qwen14b", "snapkv"): (0.108, 0.181), ("mistral7b", "snapkv"): (0.154, 0.220), ("mistral7b", "pyramidkv"): (0.191, 0.272),
}
PUB_PARETO = {  # pareto_oracle_analysis.md: gated advantage at gated's achieved cache
    ("qwen15b", 0.0625): -0.093, ("qwen15b", 0.125): 0.098, ("qwen15b", 0.25): 0.127, ("qwen15b", 0.875): 0.031,
    ("qwen3b", 0.0625): -0.051, ("qwen3b", 0.25): 0.187, ("qwen14b", 0.0625): -0.164, ("qwen14b", 0.875): 0.003,
}


def signal_rows(fname):
    rows, _ = read_jsonl(fname)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    for t in by:
        by[t].sort(key=lambda r: r["id"])
    return by


def signal_auc(sig, signal, pos_tasks, pool):
    """Oriented AUC separating pos_tasks (low) from pool, as Tables 3 and 6 report it."""
    pos = [r[signal] for t in pos_tasks for r in sig[t]]
    neg = [r[signal] for t in pool for r in sig[t]]
    a = auc(neg, pos)
    return a


def a0(args):
    print("\n## A0. Reproduction of published values, then the errors\n")
    print(f"Logs: `{os.path.relpath(RES, REPO)}`\n")
    # Table 2
    pairs, pairs_d, pairs_nm = [], [], []
    for (pol, slug), (pp, gg, nm) in PUB_T2.items():
        c = cell(matrix_file(pol, slug))
        plain, gated = c.plain.mean(), c.gated().mean()
        pairs += [(f"{pol}/{slug}/plain", plain, pp), (f"{pol}/{slug}/gated", gated, gg)]
        pairs_d.append((f"{pol}/{slug}/delta", gated - plain, gg - pp))
        mask = c.task != MK3
        pairs_nm.append((f"{pol}/{slug}/noMK3", (c.gated()[mask].mean() - c.plain[mask].mean()), nm))
    check_many("Table 2 plain and gated (32 values)", pairs)
    check_many("Table 2 delta (16 values)", pairs_d, tol=0.0021)
    check_many("Table 2 no-MK3 row (16 values)", pairs_nm)
    # Table 14
    pairs = []
    for (pol, slug), want in PUB_T14.items():
        c = cell(matrix_file(pol, slug))
        m = c.task == MK3
        got = ((c.full[:, None] & m[:, None]).astype(float) + (c.plain & ~m[:, None]).astype(float)).mean() - c.plain.mean()
        pairs.append((f"{pol}/{slug}", got, want))
    check_many("Table 14 task-label oracle (16 values)", pairs)
    # Table 29
    files = ["gated_4k_qwen15b.jsonl", "gated_4k_qwen3b.jsonl", "gated_4k_qwen14b.jsonl", "gated_4k_mistral7b.jsonl",
             "gated_16k_qwen3b.jsonl", "gated_16k_mistral7b.jsonl", "gated_16k_qwen15b_sdpa.jsonl"]
    H = collections.defaultdict(lambda: [0, 0, 0])
    for fn in files:
        c = cell(fn)
        g = c.gated()
        for j, b in enumerate(c.budgets):
            for t in SUITE:
                m = c.task == t
                s = H[(t, b)]
                s[0] += int(m.sum())
                s[1] += int((c.full[m] & ~g[m, j]).sum())
                s[2] += int((c.full[m] & ~c.plain[m, j]).sum())
    pairs = []
    for (t, b), (gw, pw) in PUB_T29.items():
        s = H[(t, b)]
        pairs += [(f"{t}/{b}/gated", s[1] / s[0], gw), (f"{t}/{b}/plain", s[2] / s[0], pw)]
    check_many("Table 29 harm rates (32 values, 16 gated/plain pairs)", pairs)
    # Table 3
    sig = signal_rows("gate_signal_ablation.jsonl")
    pool = ["vt", "fwe", "qa_1", "niah_multivalue"]
    pairs, pairs2 = [], []
    for s, want in PUB_T3.items():
        a = signal_auc(sig, s, [MK3], pool)
        pairs.append((s, max(a, 1 - a), want))
    for s, want in PUB_T3_MK23.items():
        a = signal_auc(sig, s, [MK3, "niah_multikey_2"], pool)
        pairs2.append((s, max(a, 1 - a), want))
    check_many("Table 3 AUC MK3 (6 values)", pairs)
    check_many("Table 3 AUC MK3+MK2 (6 values)", pairs2)
    # Table 6
    pairs = []
    for f, want in PUB_T6.items():
        a = signal_auc(signal_rows(f), "drop_D", [MK3], pool)
        pairs.append((f, a, want))
    check_many("Table 6 held-out AUC(D) (3 values)", pairs)
    # Table 12
    pairs = []
    for t, (f, N, af, sens, rho, D, op, dl) in PUB_T12.items():
        c = cell(f)
        j = c.bi(0.0625)
        fc = c.full
        got_sens = (fc & ~c.plain[:, j]).sum() / fc.sum()
        rec = ((~fc) & c.plain.any(axis=1)).mean()
        delta = c.gated()[:, j].mean() - c.plain[:, j].mean()
        pairs += [(f"{t}/N", c.n, N), (f"{t}/A_full", fc.mean(), af), (f"{t}/sens", got_sens, sens), (f"{t}/rho", rec, rho),
                  (f"{t}/D", c.drop.mean(), D), (f"{t}/open", c.open().mean(), op), (f"{t}/delta", delta, dl)]
    check_many("Table 12 LongBench (21 values)", pairs, tol=0.0051)
    # Table 24
    pairs = []
    for slug, (po, k1, k4, km) in PUB_T24.items():
        c = cell(f"gated_4k_{slug}.jsonl")
        op = c.open()[:, None]
        kept = np.where(op, c.kept, 1.0)
        pairs += [(f"{slug}/p_open", c.open().mean(), po), (f"{slug}/kept1_16", kept[:, c.bi(0.0625)].mean(), k1),
                  (f"{slug}/kept1_4", kept[:, c.bi(0.25)].mean(), k4), (f"{slug}/keptgrid", kept.mean(), km)]
    check_many("Table 24 achieved compression (16 values)", pairs, tol=0.0051)
    # Table 30: same cluster bootstrap as the paper
    rng = np.random.default_rng(SEED)
    pairs = []
    for (slug, pol), (lo, hi) in PUB_T30.items():
        c = cell(matrix_file(pol, slug))
        d = (c.gated().astype(float) - c.plain.astype(float)).mean(axis=1)
        idx = rng.integers(0, c.n, (10000, c.n))
        boot = d[idx].mean(axis=1)
        a, b = ci(boot)
        pairs += [(f"{slug}/{pol}/lo", a, lo), (f"{slug}/{pol}/hi", b, hi)]
    check_many("Table 30 cluster-bootstrap intervals (12 endpoints)", pairs, tol=0.006)
    # pareto_oracle_analysis.md, the source of Figure 4
    pairs = []
    for (slug, b), want in PUB_PARETO.items():
        c = cell(f"gated_4k_{slug}.jsonl")
        got = matched_rows(c)[c.bi(b)]["delta"]
        pairs.append((f"{slug}/{b}", got, want))
    check_many("pareto_oracle_analysis.md matched-cache advantage (8 values)", pairs)

    print("\n### A0-T1: Table 4 as printed against the logs\n")
    print("Printed values are the paper's `open` column (`appendix_content.tex:185-188`).\n")
    printed = {
        ("niah_multiquery", "qwen15b"): 32, ("qa_2", "qwen15b"): 0, ("niah_single_1", "qwen15b"): 0, ("cwe", "qwen15b"): 0,
        ("niah_multiquery", "qwen3b"): 100, ("qa_2", "qwen3b"): 10, ("niah_single_1", "qwen3b"): 87, ("cwe", "qwen3b"): 0,
        ("niah_multiquery", "mistral7b"): 100, ("qa_2", "mistral7b"): 18, ("niah_single_1", "mistral7b"): 0, ("cwe", "mistral7b"): 0,
    }
    rows = []
    for (t, slug), pr in printed.items():
        ds = capbound_drops(slug, t)
        op, cl = np.mean(ds >= TAU) * 100, np.mean(ds < TAU) * 100
        check(f"Table4 printed equals closed {t}/{slug}", cl, pr, tol=0.5)
        rows.append((t, slug, len(ds), s3(float(ds.mean())), f"{pr}%", f"{op:.0f}%", f"{cl:.0f}%"))
    print("| task | model | N | mean D | printed `open` | actual open | actual closed |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")

    print("\n### A0-T2: cwe on Mistral-7B, shared-mask arm\n")
    rows_cwe, _ = read_jsonl("capbound_4k_mistral7b_cwe.jsonl")
    by = collections.defaultdict(list)
    for r in rows_cwe:
        by[r["budget"]].append(r)
    print("| b | full | plain | gated at tau=0.07 | gate open |")
    print("|---|---|---|---|---|")
    for b in sorted(by):
        x = by[b]
        n = len(x)
        full = sum(r["correct_full"] for r in x) / n
        plain = sum(r["correct_plain_shared"] for r in x) / n
        gate = sum((r["correct_plain_shared"] if r["drop"] >= TAU else r["correct_full"]) for r in x) / n
        print(f"| {bname(b)} | {full:.2f} | {plain:.2f} | {gate:.2f} | {sum(r['drop'] >= TAU for r in x) / n:.2f} |")

    print("\n### A0-T3: pre-registered control and floors\n")
    for slug, t in [("qwen3b", "niah_single_1"), ("qwen15b", "cwe"), ("qwen3b", "cwe")]:
        rows_b, _ = read_jsonl(f"capbound_4k_{slug}.jsonl")
        x = [r for r in rows_b if r["task"] == t and r["budget"] == 0.0625]
        n = len(x)
        print(f"- {slug} {t} at b=1/16: N={n}, full={sum(r['correct_full'] for r in x) / n:.2f}, "
              f"plain={sum(r['correct_plain_shared'] for r in x) / n:.2f}, mean D={np.mean([r['drop'] for r in x]):+.3f}, "
              f"open={sum(r['drop'] >= TAU for r in x) / n:.2f}")

    print("\n### A0-T4: the \"3-19pp\" range in `main.tex:314`\n")
    vals = []
    for slug, _ in MODELS:
        c = cell(f"gated_4k_{slug}.jsonl")
        vals += [(r["delta"], slug, r["b"]) for r in matched_rows(c)]
    pos = [v for v in vals if v[0] > 0]
    print(f"Positive matched-cache advantages across the four SnapKV 4K cells run from {min(pos)[0] * 100:+.1f}pp "
          f"({min(pos)[1]}, b={bname(min(pos)[2])}) to {max(pos)[0] * 100:+.1f}pp ({max(pos)[1]}, b={bname(max(pos)[2])}).")
    low = sorted((float(v[0]), v[1], bname(v[2])) for v in pos if v[0] < 0.03)
    print("Values below 3pp: " + ", ".join(f"{d * 100:+.1f}pp ({s}, b={b})" for d, s, b in low) + ".")


def capbound_drops(slug, task):
    files = [f"capbound_4k_{slug}.jsonl"]
    if slug == "mistral7b":
        files.append("capbound_4k_mistral7b_cwe.jsonl")
    per = {}
    for f in files:
        rows, _ = read_jsonl(f)
        for r in rows:
            if r["task"] == task:
                per[r["id"]] = r["drop"]
    return np.array(list(per.values()), float)


# ---------------------------------------------------------------- A1
A1_CELLS = [
    ("Qwen2.5-1.5B 4K", "gated_4k_qwen15b.jsonl"), ("Qwen2.5-3B 4K", "gated_4k_qwen3b.jsonl"),
    ("Qwen2.5-14B 4K", "gated_4k_qwen14b.jsonl"), ("Mistral-7B 4K", "gated_4k_mistral7b.jsonl"),
    ("Qwen2.5-1.5B 16K", "gated_16k_qwen15b_sdpa.jsonl"), ("Qwen2.5-3B 16K", "gated_16k_qwen3b.jsonl"),
    ("Qwen2.5-14B 16K", "gated_16k_qwen14b.jsonl"), ("Mistral-7B 16K", "gated_16k_mistral7b.jsonl"),
    ("Llama-3.1-8B 4K", "gated_4k_llama31_8b.jsonl"), ("Qwen2.5-32B 4K", "gated_4k_qwen25_32b.jsonl"),
    ("Qwen3-4B 4K", "gated_4k_qwen3_4b_merged.jsonl"),
]
A1_B = [0.0625, 0.125, 0.25, 0.5]


def within_auc(task, drop, harm, safe):
    """Pair-weighted AUC of -D for harm, using only same-task pairs."""
    num, den = 0.0, 0
    per = {}
    for t in np.unique(task):
        m = task == t
        h, s = -drop[m & harm], -drop[m & safe]
        if len(h) and len(s):
            a = auc(h, s)
            num += a * len(h) * len(s)
            den += len(h) * len(s)
            if len(h) >= 5 and len(s) >= 5:
                per[t] = a
    return (num / den if den else float("nan")), den, per


def logit_beta(task, zd, y, ridge=1e-3, iters=25):
    """Harm ~ task fixed effects + zD, IRLS with a small ridge. Tasks with one class are dropped."""
    keep = np.zeros(len(y), bool)
    tasks = []
    for t in np.unique(task):
        m = task == t
        if 0 < y[m].sum() < m.sum():
            keep |= m
            tasks.append(t)
    if not tasks or keep.sum() < 10:
        return float("nan")
    X = np.column_stack([(task[keep] == t).astype(float) for t in tasks] + [zd[keep]])
    yy = y[keep].astype(float)
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ beta
        p = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        w = np.clip(p * (1 - p), 1e-6, None)
        H = (X * w[:, None]).T @ X + ridge * np.eye(X.shape[1])
        g = X.T @ (yy - p) - ridge * beta
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.abs(step).max() < 1e-6:
            break
    return float(beta[-1])


def a1(args):
    print("\n## A1. Does D predict per-input harm? (S-Q1, S-W2)\n")
    print("Harm = correct at full cache and wrong at budget b, SnapKV. Score = -D, so above 0.5 is the direction the gate assumes. "
          "Exploratory: a first pass was looked at before this script existed.\n")
    rng = np.random.default_rng(SEED + 1)
    reps = args.reps
    rows, per_all = [], []
    summary = []
    for label, f in A1_CELLS:
        c = cell(f, label)
        groups = c.task_groups()
        for b in A1_B:
            if b not in c.budgets:
                continue
            j = c.bi(b)
            D, task, full, corr = c.drop, c.task, c.full, c.plain[:, j]
            harm, safe = full & ~corr, full & corr
            pooled = auc(-D[harm], -D[safe])
            rate = {t: (harm[task == t].sum() / max(1, full[task == t].sum())) for t in np.unique(task)}
            s = np.array([rate[t] for t in task])
            task_in = auc(s[harm], s[safe])
            # Split-half task baseline: rates come from one half and score the other, so
            # every evaluation input of a task shares one score. Leave-fold-out rates would
            # be biased low here: a fold with more harm gets a lower rate from the others,
            # which builds a negative correlation into the within-task ordering.
            sh_t, sh_d = [], []
            for _ in range(50):
                A = np.zeros(c.n, bool)
                for g in groups:
                    pm = rng.permutation(g)
                    A[pm[: len(pm) // 2]] = True
                B_ = ~A
                sc = np.zeros(c.n)
                for t in np.unique(task):
                    m = task == t
                    den = (full & A & m).sum()
                    sc[m] = (harm & A & m).sum() / den if den else 0.0
                sh_t.append(auc(sc[harm & B_], sc[safe & B_]))
                sh_d.append(auc(-D[harm & B_], -D[safe & B_]))
            task_cf = float(np.nanmean(sh_t))
            d_cf = float(np.nanmean(sh_d))
            wa, pairs, per = within_auc(task, D, harm, safe)
            nm = task != MK3
            pooled_nm = auc(-D[harm & nm], -D[safe & nm])
            succ = auc(D[corr], D[~corr])
            zd = (D[full] - D[full].mean()) / (D[full].std() or 1)
            beta = logit_beta(task[full], zd, harm[full])
            bp, bw, bb = [], [], []
            for _ in range(reps):
                idx = strat_idx(rng, groups)
                dd, tt, hh, ss = D[idx], task[idx], harm[idx], safe[idx]
                bp.append(auc(-dd[hh], -dd[ss]))
                bw.append(within_auc(tt, dd, hh, ss)[0])
            for _ in range(min(300, reps)):
                idx = strat_idx(rng, groups)
                fm = full[idx]
                if fm.sum() < 10:
                    continue
                dd = D[idx][fm]
                bb.append(logit_beta(task[idx][fm], (dd - dd.mean()) / (dd.std() or 1), harm[idx][fm]))
            rows.append((label, bname(b), int(full.sum()), int(harm.sum()), f3(pooled), ci3(ci(bp)), f3(task_in), f3(task_cf), f3(d_cf),
                         f3(wa), ci3(ci(bw)), pairs, f3(pooled_nm), f3(succ), s3(beta), ci3(ci(bb), True)))
            summary.append((label, b, pooled, task_in, task_cf, wa, ci(bw), d_cf))
            for t, a in per.items():
                per_all.append((label, b, t, a))
    table("A1-T1", "D against per-input harm",
          ["cell", "b", "full correct", "harmed", "pooled AUC(D)", "95% CI", "task id (in-sample)", "task id (split-half)", "D (same splits)",
           "within-task AUC", "95% CI", "same-task pairs", "pooled AUC without MK3", "success AUC(D)", "beta per SD of D", "95% CI"], rows,
          "Beta is from harm ~ task fixed effects + standardized D, so a negative value means higher D predicts less harm within a task.")

    four = [s for s in summary if s[0].endswith("4K") and s[0].split()[0] in ("Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-14B", "Mistral-7B")]
    tol = 5e-4
    below = sum(s[2] < s[3] - tol for s in four)
    tied = sum(abs(s[2] - s[3]) <= tol for s in four)
    above = sum(s[2] > s[3] + tol for s in four)
    print(f"\nFour SnapKV 4K models, four budgets (16 pairs): pooled AUC(D) range {min(s[2] for s in four):.3f} to {max(s[2] for s in four):.3f}. "
          f"Against in-sample task identity: {below} below, {tied} tied, {above} above. "
          f"Against split-half task identity on the same splits: {sum(s[7] < s[4] - tol for s in four)} below, "
          f"{sum(abs(s[7] - s[4]) <= tol for s in four)} tied, {sum(s[7] > s[4] + tol for s in four)} above.")
    print(f"All {len(summary)} (cell, budget) pairs: {sum(s[2] < s[3] - tol for s in summary)} below in-sample task identity, "
          f"{sum(abs(s[2] - s[3]) <= tol for s in summary)} tied, {sum(s[2] > s[3] + tol for s in summary)} above. "
          f"Split-half, same splits: {sum(s[7] < s[4] - tol for s in summary)} below, {sum(abs(s[7] - s[4]) <= tol for s in summary)} tied, {sum(s[7] > s[4] + tol for s in summary)} above. "
          f"Within-task CI excludes 0.5 above: {sum(s[6][0] > 0.5 for s in summary)}; below: {sum(s[6][1] < 0.5 for s in summary)}; includes 0.5: "
          f"{sum(s[6][0] <= 0.5 <= s[6][1] for s in summary)}.")

    v4 = [a for l, b, t, a in per_all if l.endswith("4K") and l.split()[0] in ("Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-14B", "Mistral-7B")]
    va = [a for _, _, _, a in per_all]
    print(f"\nPer-task AUCs with at least 5 inputs in each class. Four 4K models: {len(v4)} values, {min(v4):.3f} to {max(v4):.3f}, mean {np.mean(v4):.3f}, "
          f"{sum(a < 0.5 for a in v4)} below 0.5. All cells: {len(va)} values, {min(va):.3f} to {max(va):.3f}, mean {np.mean(va):.3f}, {sum(a < 0.5 for a in va)} below 0.5.")
    check("A1 per-task count", len(v4), 26, 0)
    check("A1 per-task below", sum(a < 0.5 for a in v4), 16, 0)
    check("A1 per-task mean", float(np.mean(v4)), 0.461)

    # pooled within-task across budgets, per model, clusters = inputs
    rows = []
    for label, f in A1_CELLS:
        c = cell(f, label)
        groups = c.task_groups()
        js = [c.bi(b) for b in A1_B if b in c.budgets]

        def stat(idx):
            num, den = 0.0, 0
            for j in js:
                full, corr = c.full[idx], c.plain[idx, j]
                a, p, _ = within_auc(c.task[idx], c.drop[idx], full & ~corr, full & corr)
                if p:
                    num += a * p
                    den += p
            return num / den if den else float("nan")

        pt = stat(np.arange(c.n))
        boot = [stat(strat_idx(rng, groups)) for _ in range(min(1000, reps))]
        rows.append((label, f3(pt), ci3(ci(boot))))
    table("A1-T2", "Within-task AUC pooled over the four budgets, input-clustered bootstrap", ["cell", "within-task AUC", "95% CI"], rows)

    rows = []
    for label, f in A1_CELLS[:4]:
        c = cell(f, label)
        for b in (0.0625, 0.25):
            j = c.bi(b)
            order = np.argsort(c.drop, kind="mergesort")
            for q in range(5):
                x = order[q * c.n // 5:(q + 1) * c.n // 5]
                fc = x[c.full[x]]
                nm = x[c.task[x] != MK3]
                nmfc = nm[c.full[nm]]
                rows.append((label, bname(b), f"Q{q + 1}", f"[{c.drop[x].min():+.3f}, {c.drop[x].max():+.3f}]", f3(c.plain[x, j].mean()),
                             f3((~c.plain[fc, j]).mean() if len(fc) else float("nan")), f"{(c.task[x] == MK3).mean():.2f}",
                             f3((~c.plain[nmfc, j]).mean() if len(nmfc) else float("nan"))))
    table("A1-T3", "Reliability: D quintile against success and harm",
          ["cell", "b", "D quintile", "D range", "success at b", "harm rate", "MK3 share", "harm rate without MK3"], rows)


# ---------------------------------------------------------------- A2
def a2_cells():
    out = []
    for pol in POLICIES:
        for slug, name in MODELS:
            out.append((f"{POLICY_NAME[pol]} {name}", matrix_file(pol, slug), TAU, "matrix"))
    return out


def a2_extras():
    return [
        ("SnapKV Qwen2.5-1.5B 16K", "gated_16k_qwen15b_sdpa.jsonl"), ("SnapKV Qwen2.5-3B 16K", "gated_16k_qwen3b.jsonl"),
        ("SnapKV Qwen2.5-14B 16K", "gated_16k_qwen14b.jsonl"), ("SnapKV Mistral-7B 16K", "gated_16k_mistral7b.jsonl"),
        ("SnapKV Llama-3.1-8B 4K", "gated_4k_llama31_8b.jsonl"), ("SnapKV Qwen2.5-32B 4K", "gated_4k_qwen25_32b.jsonl"),
        ("SnapKV Qwen3-4B 4K", "gated_4k_qwen3_4b_merged.jsonl"),
    ]


def decomp_boot(c, tau, rng, reps):
    f, p = c.full.astype(float), c.plain.astype(float).mean(axis=1)
    g = c.gated(tau).astype(float).mean(axis=1)
    o = c.open(tau).astype(float)
    idx = rng.integers(0, c.n, (reps, c.n))
    P, G, F, O = p[idx].mean(1), g[idx].mean(1), f[idx].mean(1), o[idx].mean(1)
    page, rnd = G - P, (1 - O) * (F - P)
    return page, rnd


def a2(args):
    print("\n## A2. Fallback vs selection (S-W2, R-Q1)\n")
    print("Random gate: opens at random at PAGE's open rate, so it uses about the same cache. Gain = (1 - open) x (full - plain). "
          "Selection = PAGE gain - random gain. Cluster bootstrap over inputs, 10,000 replicates, the paper's Table 30 method.\n")
    rng = np.random.default_rng(SEED + 2)
    reps = 10000
    rows, pvals, stats = [], [], []
    for label, f, tau, _ in a2_cells():
        c = cell(f)
        page, rnd = decomp_boot(c, tau, rng, reps)
        sel = page - rnd
        P, G, F = c.plain.mean(), c.gated(tau).mean(), c.full.mean()
        po = c.open(tau).mean()
        gp, gr = G - P, (1 - po) * (F - P)
        p2 = min(1.0, 2 * min((sel <= 0).mean() + 1 / reps, (sel >= 0).mean() + 1 / reps))
        pvals.append(p2)
        m = c.task != MK3
        pom = c.open(tau)[m].mean()
        gpm = c.gated(tau)[m].mean() - c.plain[m].mean()
        grm = (1 - pom) * (c.full[m].mean() - c.plain[m].mean())
        stats.append((gp, gr, gpm, grm))
        rows.append([label, len(c.budgets), f3(po), f3(P), f3(G), f3(F), s3(gp), s3(gr), s3(gp - gr), ci3(ci(sel), True), f"{p2:.4f}", None, s3(gpm), s3(grm), f"{gr / gp * 100:.0f}%"])
    keep = bh(pvals)
    for r, k in zip(rows, keep):
        r[11] = "yes" if k else "no"
    table("A2-T1", "The 16 matrix cells, tau = 0.07",
          ["cell", "budgets", "open", "plain", "gated", "full", "PAGE gain", "random gain", "selection", "95% CI", "p", "BH q=.05", "no-MK3 PAGE", "no-MK3 random", "fallback share"], rows)
    st = np.array(stats)
    print(f"\nGrand means over 16 cells: PAGE {st[:, 0].mean():+.3f}, random {st[:, 1].mean():+.3f}, selection {st[:, 0].mean() - st[:, 1].mean():+.3f}, "
          f"fallback share {st[:, 1].mean() / st[:, 0].mean() * 100:.1f}%. Without MK3: PAGE {st[:, 2].mean():+.3f}, random {st[:, 3].mean():+.3f}.")
    check("A2 grand PAGE gain", float(st[:, 0].mean()), 0.229)
    n_sel = sum(float(ci_lo) > 0 for ci_lo in [re.findall(r"[-+]?\d*\.\d+", r[9])[0] for r in rows])
    print(f"Cells whose selection CI excludes 0 on the positive side: {n_sel} of 16. Surviving BH at q=0.05: {int(keep.sum())} of 16.")

    # Joint bootstrap of the grand means: same resampled inputs across evictors within a model.
    G_page, G_rand, Gm_page, Gm_rand = np.zeros(reps), np.zeros(reps), np.zeros(reps), np.zeros(reps)
    for slug, _ in MODELS:
        cs = [cell(matrix_file(pol, slug)) for pol in POLICIES]
        assert all(x.keys == cs[0].keys for x in cs), slug
        idx = rng.integers(0, cs[0].n, (reps, cs[0].n))
        m = (cs[0].task != MK3).astype(float)
        M = m[idx]
        for c in cs:
            f, p = c.full.astype(float), c.plain.astype(float).mean(1)
            g, o = c.gated().astype(float).mean(1), c.open().astype(float)
            P, Gg, F, O = p[idx].mean(1), g[idx].mean(1), f[idx].mean(1), o[idx].mean(1)
            G_page += Gg - P
            G_rand += (1 - O) * (F - P)
            den = M.sum(1)
            Pm, Ggm, Fm, Om = (p[idx] * M).sum(1) / den, (g[idx] * M).sum(1) / den, (f[idx] * M).sum(1) / den, (o[idx] * M).sum(1) / den
            Gm_page += Ggm - Pm
            Gm_rand += (1 - Om) * (Fm - Pm)
    G_page, G_rand, Gm_page, Gm_rand = G_page / 16, G_rand / 16, Gm_page / 16, Gm_rand / 16
    print(f"\nA2-T2. Grand means with joint bootstrap (inputs resampled together across evictors within a model):")
    print(f"- PAGE gain {ci3(ci(G_page), True)}; random gain {ci3(ci(G_rand), True)}; selection {ci3(ci(G_page - G_rand), True)}; fallback share "
          f"[{ci(G_rand / G_page)[0] * 100:.1f}%, {ci(G_rand / G_page)[1] * 100:.1f}%].")
    print(f"- Without MK3: PAGE {ci3(ci(Gm_page), True)}; random {ci3(ci(Gm_rand), True)}; selection {ci3(ci(Gm_page - Gm_rand), True)}.")

    rows, pvals = [], []
    for label, f in a2_extras():
        c = cell(f)
        for gate in ("tau", "z"):
            tau = TAU if gate == "tau" else c.drop.mean() + THETA_Z * c.drop.std()
            page, rnd = decomp_boot(c, tau, rng, reps)
            sel = page - rnd
            P, G, F = c.plain.mean(), c.gated(tau).mean(), c.full.mean()
            po = c.open(tau).mean()
            gp, gr = G - P, (1 - po) * (F - P)
            p2 = min(1.0, 2 * min((sel <= 0).mean() + 1 / reps, (sel >= 0).mean() + 1 / reps))
            pvals.append(p2)
            kept = np.where(c.open(tau)[:, None], c.kept, 1.0).mean()
            rows.append([label, "tau=0.07" if gate == "tau" else f"z-gate ({tau:.3f})", c.n, len(c.budgets), f3(po), f3(kept), f3(P), f3(G), f3(F), s3(gp), s3(gr), s3(gp - gr), ci3(ci(sel), True), f"{p2:.4f}"])
    table("A2-T3", "Other cells, fixed tau and z-gate",
          ["cell", "gate", "N", "budgets", "open", "mean kept", "plain", "gated", "full", "PAGE gain", "random gain", "selection", "95% CI", "p"], rows,
          "Yi-1.5-9B has no full-cache rows, so it cannot enter this table. The z-gate here standardizes over all of the cell's inputs, as the fit did.")


# ---------------------------------------------------------------- A3
def matched_rows(c, tau=TAU, idx=None):
    if idx is None:
        idx = np.arange(c.n)
    full, plain, kept, op = c.full[idx], c.plain[idx], c.kept[idx], c.open(tau)[idx]
    xs = np.r_[kept.mean(axis=0), 1.0]
    ys = np.r_[plain.mean(axis=0), full.mean()]
    harm_curve = np.r_[(full[:, None] & ~plain).mean(axis=0), 0.0]
    order = np.argsort(xs)
    xs, ys, harm_curve = xs[order], ys[order], harm_curve[order]
    p_open = op.mean()
    out = []
    for j, b in enumerate(c.budgets):
        kg = np.where(op, kept[:, j], 1.0).mean()
        gated = np.where(op, plain[:, j], full)
        ag = gated.mean()
        ap = float(np.interp(kg, xs, ys))
        hg = (full & ~gated).mean()
        hp = float(np.interp(kg, xs, harm_curve))
        rnd = p_open * plain[:, j].mean() + (1 - p_open) * full.mean()
        out.append({"b": b, "kept_g": kg, "gated": ag, "plain_at": ap, "delta": ag - ap, "harm_g": hg, "harm_p_at": hp, "random": rnd})
    return out


def a3(args):
    print("\n## A3. Matched achieved cache (S-W2)\n")
    print("Plain accuracy is linearly interpolated at the gated arm's achieved kept fraction, which a mix of two adjacent budgets can achieve. "
          "Harm here is a share of all inputs, not of full-correct inputs.\n")
    rng = np.random.default_rng(SEED + 3)
    reps = min(args.reps, 2000)
    cells8 = [(f"{name} 4K", f"gated_4k_{slug}.jsonl") for slug, name in MODELS] + [
        ("Qwen2.5-1.5B 16K", "gated_16k_qwen15b_sdpa.jsonl"), ("Qwen2.5-3B 16K", "gated_16k_qwen3b.jsonl"),
        ("Qwen2.5-14B 16K", "gated_16k_qwen14b.jsonl"), ("Mistral-7B 16K", "gated_16k_mistral7b.jsonl")]
    rows, summ = [], []
    for label, f in cells8:
        c = cell(f)
        base = matched_rows(c)
        boots = [matched_rows(c, TAU, rng.integers(0, c.n, c.n)) for _ in range(reps)]
        means = []
        for j, r in enumerate(base):
            d = [bb[j]["delta"] for bb in boots]
            h = [bb[j]["harm_g"] - bb[j]["harm_p_at"] for bb in boots]
            rows.append((label, bname(r["b"]), f3(r["kept_g"]), f3(r["gated"]), f3(r["plain_at"]), s3(r["delta"]), ci3(ci(d), True), f3(r["random"]), f3(r["harm_g"]), f3(r["harm_p_at"]), s3(r["harm_g"] - r["harm_p_at"]), ci3(ci(h), True)))
        grid = [np.mean([bb[j]["delta"] for j in range(len(base))]) for bb in boots]
        ds = [r["delta"] for r in base]
        cross = None
        for a, bb2 in zip(base, base[1:]):
            if a["delta"] < 0 <= bb2["delta"]:
                cross = a["kept_g"] + (0 - a["delta"]) * (bb2["kept_g"] - a["kept_g"]) / (bb2["delta"] - a["delta"])
        summ.append((label, s3(float(np.mean(ds))), ci3(ci(grid), True), s3(min(ds)), s3(max(ds)), f"{sum(x < 0 for x in ds)}/{len(ds)}", f3(cross) if cross else "none"))
    table("A3-T1", "Matched-cache comparison per budget, SnapKV",
          ["cell", "b", "gated kept", "gated acc", "plain acc at that kept", "delta", "95% CI", "random gate acc", "gated harm", "plain harm at that kept", "harm delta", "95% CI"], rows)
    table("A3-T2", "Summary per cell", ["cell", "mean delta", "95% CI", "min", "max", "negative budgets", "crossover kept"], summ)
    rows = []
    for pol in POLICIES[1:]:
        for slug, name in MODELS:
            c = cell(matrix_file(pol, slug))
            base = matched_rows(c)
            ds = [r["delta"] for r in base]
            rows.append((POLICY_NAME[pol], name, ",".join(bname(b) for b in c.budgets), s3(float(np.mean(ds))), s3(min(ds)), s3(max(ds))))
    table("A3-T3", "Other evictors, coarse grids, read with care", ["evictor", "model", "budgets", "mean delta", "min", "max"], rows,
          "These plain curves have no points between b = 1/2 and full cache, so interpolation uses a long chord and favours the gate. StreamingLLM keeps a constant cache, so its curve is a single chord. These are not reliable matched-cache estimates.")


# ---------------------------------------------------------------- A4
CAL_FILES = [("Qwen1.5B-4K", "drops_qwen15b_4k_n100.jsonl"), ("Qwen3B-4K", "drops_qwen3b_4k_n100.jsonl"), ("Qwen3B-16K", "drops_qwen3b_16k_n100.jsonl"),
             ("Mistral7B-4K", "drops_mistral7b_4k_n100.jsonl"), ("Mistral7B-16K", "drops_mistral7b_16k_n100.jsonl")]


def load_drops(f, dedupe_mk3=False):
    rows, bad = read_jsonl(f)
    t, d, e, l_ = [], [], [], []
    mk = 0
    for r in rows:
        if dedupe_mk3 and r["task"] == MK3:
            mk += 1
            if mk > 100:
                continue
        pl = np.array(r["head_agreement_per_layer"], float)
        L = len(pl)
        t.append(r["task"])
        e.append(pl[: L // 3].mean())
        l_.append(pl[2 * L // 3:].mean())
        d.append(e[-1] - l_[-1])
    return np.array(t), np.array(d), np.array(e), np.array(l_)


def zscore(x):
    return (x - x.mean()) / x.std()


def sweep(cells_z, strict, grid=None):
    """Reimplements normalized_predictor.sweep: maximize (#cells separated, balanced acc)."""
    allv = np.unique(np.concatenate([z for _, z in cells_z]))
    cand = (allv[1:] + allv[:-1]) / 2 if grid is None else grid
    excl = (MK3,) if strict else (MK3, "niah_multivalue")
    intervals = []
    for t, z in cells_z:
        means = {u: z[t == u].mean() for u in np.unique(t)}
        others = [v for u, v in means.items() if u not in excl]
        intervals.append((means[MK3], min(others)))
    nok = np.zeros(len(cand), int)
    for lo, hi in intervals:
        nok += ((lo < cand) & (cand < hi)).astype(int)
    mk3 = np.sort(np.concatenate([z[t == MK3] for t, z in cells_z]))
    rest = np.sort(np.concatenate([z[~np.isin(t, excl)] for t, z in cells_z]))
    bal = 0.5 * (np.searchsorted(mk3, cand, "left") / len(mk3) + (len(rest) - np.searchsorted(rest, cand, "left")) / len(rest))
    # first maximum in ascending order, as the original's strict `>` does
    best = np.lexsort((-bal, -nok))[0]
    best_score = (nok[best], bal[best])
    first = next(i for i in range(len(cand)) if nok[i] == best_score[0] and bal[i] == best_score[1])
    return cand[first], int(nok[first]), float(bal[first]), cand, nok, bal


def a4(args):
    print("\n## A4. How -0.69 was selected, and threshold stability (R-Q2, G-W2)\n")
    rng = np.random.default_rng(SEED + 4)
    data = {}
    for lab, f in CAL_FILES:
        t, d, _, _ = load_drops(f, dedupe_mk3="mistral7b_16k" in f)
        data[lab] = (t, d)
    cells_z = [(t, zscore(d)) for t, d in data.values()]
    th, nok, bal, cand, noks, bals = sweep(cells_z, strict=False)
    print(f"Re-derived fit (relaxed, niah_multivalue excluded): theta_z = {th:+.4f}, cells separated {nok}/5, balanced accuracy {bal:.3f}.")
    check("A4 theta_z", th, -0.6934, 0.001)
    th_s, nok_s, bal_s, *_ = sweep(cells_z, strict=True)
    print(f"Strict fit (niah_multivalue kept): theta_z = {th_s:+.4f}, cells separated {nok_s}/5, balanced accuracy {bal_s:.3f}.")
    ok_range = cand[noks == nok]
    print(f"Thresholds with {nok}/5 cells separated span [{ok_range.min():+.3f}, {ok_range.max():+.3f}].")
    near = cand[(noks == nok) & (bals >= bal - 0.01)]
    print(f"Within 0.01 balanced accuracy of the best, they span [{near.min():+.3f}, {near.max():+.3f}].")
    rows = []
    for i, (lab, _) in enumerate(CAL_FILES):
        th_i, n_i, b_i, *_ = sweep([x for k, x in enumerate(cells_z) if k != i], strict=False)
        tt, dd = data[lab]
        z = zscore(dd)
        mk = z[tt == MK3].mean()
        oth = min(z[tt == u].mean() for u in np.unique(tt) if u not in (MK3, "niah_multivalue"))
        rows.append((lab, s3(th_i), f"{n_i}/4", f3(b_i), "yes" if mk < th_i < oth else "no"))
    table("A4-T1", "Leave-one-cell-out fit", ["held-out cell", "theta_z", "cells separated", "balanced acc", "held-out cell separated at that theta"], rows)
    grid = np.arange(-2.0, 2.0, 0.005)
    reps = min(args.reps, 500)
    ths, nks = [], []
    for _ in range(reps):
        cz = []
        for t, d in data.values():
            idx = np.concatenate([np.where(t == u)[0][rng.integers(0, (t == u).sum(), (t == u).sum())] for u in np.unique(t)])
            cz.append((t[idx], zscore(d[idx])))
        a, n, *_ = sweep(cz, strict=False, grid=grid)
        ths.append(a)
        nks.append(n)
    print(f"\nBootstrap of the fit ({reps} resamples of inputs within task within cell): theta_z mean {np.mean(ths):+.3f}, SD {np.std(ths):.3f}, "
          f"95% range {ci3(ci(ths), True)}, resamples reaching 5/5: {np.mean(np.array(nks) == 5) * 100:.0f}%.")

    print("\nThe fit standardizes over all of each cell's inputs from six tasks, so it uses task labels (in the objective) and a mix that is not a separate pilot.")

    # pilot stability, held-out evaluation
    rows, rows2 = [], []
    reps = min(args.reps, 400)
    for slug, name in MODELS:
        c = cell(f"gated_4k_{slug}.jsonl")
        mk_i, ot_i = np.where(c.task == MK3)[0], np.where(c.task != MK3)[0]
        j1 = c.bi(0.0625)
        pa = c.plain.astype(float).mean(1)
        settings = [(100, s) for s in (0.0, 0.05, 0.10, 0.25, 0.50)] + [(n, 0.25) for n in (20, 50, 200)] + [(n, 0.05) for n in (20, 50, 200)]
        for n, share in settings:
            res = collections.defaultdict(list)
            for _ in range(reps):
                pm = rng.permutation(mk_i)
                po = rng.permutation(ot_i)
                # keep tasks balanced in the split of the other tasks
                ev_mk, pl_mk = pm[: len(pm) // 2], pm[len(pm) // 2:]
                ev_ot, pl_ot = po[: len(po) // 2], po[len(po) // 2:]
                k = int(round(n * share))
                pilot = np.r_[pl_mk[rng.integers(0, len(pl_mk), k)], pl_ot[rng.integers(0, len(pl_ot), n - k)]]
                tau = c.drop[pilot].mean() + THETA_Z * c.drop[pilot].std()
                ev = np.r_[ev_mk, ev_ot]
                op = c.drop[ev] >= tau
                gate1 = np.where(op, c.plain[ev, j1], c.full[ev])
                fc = c.full[ev]
                res["tau"].append(tau)
                res["open"].append(op.mean())
                res["mk3open"].append((c.drop[ev_mk] >= tau).mean())
                res["otheropen"].append((c.drop[ev_ot] >= tau).mean())
                res["acc1"].append(gate1.mean())
                res["harm1"].append((fc & ~gate1)[fc].mean() if fc.any() else np.nan)
                res["kept1"].append(np.where(op, c.kept[ev, j1], 1.0).mean())
                # deployment mix equal to the pilot mix
                w = share
                g_all = np.where((c.drop >= tau)[:, None], c.plain, c.full[:, None]).astype(float).mean(1)
                kept_all = np.where((c.drop >= tau)[:, None], c.kept, 1.0).mean(1)
                gain_m, gain_o = (g_all[ev_mk] - pa[ev_mk]).mean(), (g_all[ev_ot] - pa[ev_ot]).mean()
                res["gain"].append(w * gain_m + (1 - w) * gain_o)
                res["kept"].append(w * kept_all[ev_mk].mean() + (1 - w) * kept_all[ev_ot].mean())
                po_w = w * (c.drop[ev_mk] >= tau).mean() + (1 - w) * (c.drop[ev_ot] >= tau).mean()
                full_w = w * c.full[ev_mk].mean() + (1 - w) * c.full[ev_ot].mean()
                plain_w = w * pa[ev_mk].mean() + (1 - w) * pa[ev_ot].mean()
                res["rand"].append((1 - po_w) * (full_w - plain_w))
            m = {k_: float(np.nanmean(v)) for k_, v in res.items()}
            rows.append((name, n, f"{share * 100:.0f}%", f"{m['tau']:.3f}", f"{np.std(res['tau']):.3f}", f3(m["open"]), f3(m["mk3open"]), f3(m["otheropen"]), f3(m["acc1"]), f3(m["harm1"]), f3(m["kept1"])))
            rows2.append((name, n, f"{share * 100:.0f}%", s3(m["gain"]), s3(m["rand"]), s3(m["gain"] - m["rand"]), f3(m["kept"])))
    table("A4-T2", "z-gate from a pilot, evaluated on held-out inputs at the suite's equal task mix, b = 1/16",
          ["model", "pilot size", "pilot MK3 share", "threshold", "SD", "open", "MK3 open", "others open", "accuracy", "harm (of full-correct)", "kept"], rows,
          "Each resample splits inputs in half, draws the pilot from one half with replacement, and evaluates on the other. Other tasks are pooled in the pilot.")
    table("A4-T3", "Same pilots, evaluated at a deployment mix equal to the pilot's MK3 share, eight-budget grid",
          ["model", "pilot size", "MK3 share", "PAGE gain", "random gain at same open", "selection", "mean kept"], rows2)
    # fixed tau and 40-input midpoint
    rows = []
    for slug, name in MODELS:
        c = cell(f"gated_4k_{slug}.jsonl")
        mk, vt = c.drop[c.task == MK3], c.drop[c.task == "vt"]
        ts = [(mk[rng.integers(0, len(mk), 20)].mean() + vt[rng.integers(0, len(vt), 20)].mean()) / 2 for _ in range(2000)]
        rows.append((name, f"{np.mean(ts):.3f}", f"{np.std(ts):.3f}", ci3(ci(ts)), f"{(c.drop >= 0.07).mean():.3f}", f"{(c.drop >= np.mean(ts)).mean():.3f}"))
    table("A4-T4", "40-input midpoint recipe (20 MK3 + 20 VT, needs task labels), resampling spread", ["model", "threshold", "SD", "95% range", "open at tau=0.07", "open at midpoint"], rows)

    # Is the z-gate a fixed-quantile rule? Phi(-0.69) = 0.245.
    q = 0.2451
    rows = []
    cells_q = [(f"{name} 4K", f"gated_4k_{slug}.jsonl") for slug, name in MODELS] + a2_extras()
    opens = []
    for label, f in cells_q:
        c = cell(f)
        tz = c.drop.mean() + THETA_Z * c.drop.std()
        tq = np.quantile(c.drop, q)
        gz, gq = c.gated(tz).mean() - c.plain.mean(), c.gated(tq).mean() - c.plain.mean()
        mz = (c.drop[c.task == MK3] < tz).mean()
        same = (c.open(tz) == c.open(tq)).mean()
        opens.append(c.open(tz).mean())
        rows.append((label, c.n, f3(c.open(tz).mean()), f3(c.open(tq).mean()), f3(same), s3(gz), s3(gq), f3(mz)))
    table("A4-T5", "The z-gate against a rule that closes the lowest 24.5% of inputs (Phi(-0.69) = 0.245), cell's own inputs",
          ["cell", "N", "z-gate open", "quantile gate open", "same decision on", "z-gate gain", "quantile gain", "MK3 closed by z-gate"], rows,
          f"Across these {len(rows)} cells the z-gate's open rate runs from {min(opens):.3f} to {max(opens):.3f}. MK3 is 25% of the four-task suites and 1/6 of the Llama suite.")



# ---------------------------------------------------------------- A6
def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_tokens(pred, gold):
    common = collections.Counter(pred) & collections.Counter(gold)
    same = sum(common.values())
    if same == 0:
        return 0.0
    p, r = same / len(pred), same / len(gold)
    return 2 * p * r / (p + r)


def qa_f1(pred, gold):
    return f1_tokens(normalize_answer(pred).split(), normalize_answer(gold).split())


def code_sim(pred, gold):
    # LongBench uses fuzzywuzzy's ratio, which falls back to difflib when python-Levenshtein is absent.
    chosen = ""
    for line in pred.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            chosen = line
            break
    return round(100 * difflib.SequenceMatcher(None, chosen, gold).ratio()) / 100


def count_score(pred, gold):
    nums = re.findall(r"\d+", pred)
    return 0.0 if not nums else sum(n == str(gold) for n in nums) / len(nums)


def retrieval_score(pred, gold):
    m = re.findall(r"Paragraph (\d+)", gold)
    if not m:
        return 0.0
    nums = re.findall(r"\d+", pred)
    return 0.0 if not nums else sum(n == m[0] for n in nums) / len(nums)


OFFICIAL = {"qasper": qa_f1, "multifieldqa_en": qa_f1, "hotpotqa": qa_f1, "triviaqa": qa_f1, "lcc": code_sim, "repobench-p": code_sim, "passage_count": count_score, "passage_retrieval_en": retrieval_score}
FIRST_LINE = {"triviaqa"}


def official(task, pred, answers):
    fn = OFFICIAL.get(task)
    if fn is None:
        return float("nan")
    if task in FIRST_LINE:
        pred = pred.lstrip("\n").split("\n")[0]
    return max((fn(pred, a) for a in answers), default=0.0)


LB_FILES = [
    ("Qwen2.5-14B", "longbench_realistic_lcc_qwen14b.jsonl"), ("Qwen2.5-14B", "longbench_realistic_repobench-p_qwen14b.jsonl"),
    ("Qwen2.5-14B", "longbench_realistic_hotpotqa_qwen14b.jsonl"), ("Qwen2.5-14B", "longbench_passcount_qwen14b.jsonl"),
    ("Qwen2.5-14B", "longbench_qwen14b.jsonl"), ("Qwen2.5-1.5B", "longbench_qwen15b.jsonl"), ("Qwen2.5-3B", "longbench_qwen3b.jsonl"),
    ("Mistral-7B", "longbench_mistral7b.jsonl"), ("Qwen2.5-1.5B", "longbench_passret_qwen15b_n100.jsonl"), ("Qwen2.5-1.5B", "longbench_passcount_qwen15b_n50.jsonl"),
]


def a6(args):
    print("\n## A6. Beyond RULER (S-Q2, G-W3, G-Q1)\n")
    print("Accuracy is the logged substring match. Official = LongBench's own metric recomputed on stored predictions (QA F1, code similarity, count, retrieval); trec needs a class list and is not re-scored. "
          "Stored predictions are cut at 200 characters and came from a generic prompt rather than LongBench's own, so official scores are indicative. Selection = PAGE - random at the same open rate, both at the lowest logged budget.\n")
    rng = np.random.default_rng(SEED + 6)
    reps = min(args.reps, 2000)
    rows, rows_off, pool = [], [], collections.defaultdict(list)
    for model, f in LB_FILES:
        rows_raw, _ = read_jsonl(f)
        tasks = sorted({r["task"] for r in rows_raw})
        for t in tasks:
            c = Cell(f, tasks=[t])
            if c.n == 0:
                continue
            b = c.budgets[0]
            j = c.bi(b)
            op = c.open()
            full, plain = c.full.astype(float), c.plain.astype(float)
            page = np.where(op, plain[:, j], full)
            po = op.mean()
            rnd = po * plain[:, j].mean() + (1 - po) * full.mean()
            h, s = c.full & ~c.plain[:, j], c.full & c.plain[:, j]
            a = auc(-c.drop[h], -c.drop[s])
            sels, aucs = [], []
            for _ in range(reps):
                idx = rng.integers(0, c.n, c.n)
                sels.append(page[idx].mean() - (op[idx].mean() * plain[idx, j].mean() + (1 - op[idx].mean()) * full[idx].mean()))
                aucs.append(auc(-c.drop[idx][h[idx]], -c.drop[idx][s[idx]]))
            rows.append((model, t, c.n, bname(b), f3(po), f3(full.mean()), f3(plain[:, j].mean()), f3(page.mean()), f3(rnd), s3(page.mean() - plain[:, j].mean()), s3(page.mean() - rnd), ci3(ci(sels), True), f"{h.sum()}/{s.sum()}", f3(a), ci3(ci(aucs))))
            pool[model].append((t, c, j))
            # official metric
            if t in OFFICIAL:
                sf = np.array([official(t, c.by[k][1.0]["pred_plain"], c.by[k][1.0]["answers"]) for k in c.keys])
                sp = np.array([official(t, c.by[k][b]["pred_plain"], c.by[k][b]["answers"]) for k in c.keys])
                sg = np.where(op, sp, sf)
                rnd_o = po * sp.mean() + (1 - po) * sf.mean()
                m = sf > 0
                rho = spearman(c.drop[m], (sp - sf)[m])
                bs = []
                for _ in range(reps):
                    idx = rng.integers(0, c.n, c.n)
                    bs.append(sg[idx].mean() - (op[idx].mean() * sp[idx].mean() + (1 - op[idx].mean()) * sf[idx].mean()))
                br = []
                mi = np.where(m)[0]
                for _ in range(min(reps, 1000)):
                    idx = mi[rng.integers(0, len(mi), len(mi))]
                    br.append(spearman(c.drop[idx], (sp - sf)[idx]))
                rows_off.append((model, t, c.n, bname(b), f3(float(sf.mean())), f3(float(sp.mean())), f3(float(sg.mean())), f3(float(rnd_o)), s3(float(sg.mean() - sp.mean())), s3(float(sg.mean() - rnd_o)), ci3(ci(bs), True), s3(rho), ci3(ci(br), True)))
    table("A6-T1", "LongBench, substring accuracy", ["model", "subtask", "N", "b", "open", "full", "plain", "PAGE", "random", "PAGE - plain", "selection", "95% CI", "harmed/survived", "AUC(D)", "95% CI"], rows)
    table("A6-T2", "LongBench, official metrics", ["model", "subtask", "N", "b", "full", "plain", "PAGE", "random", "PAGE - plain", "selection", "95% CI", "Spearman D vs score change", "95% CI"], rows_off,
          "A positive Spearman means higher D goes with smaller loss, the direction the gate assumes. Inputs with zero full-cache score are excluded from it.")

    # Pooled over subtasks per model at a common budget (b = 1/8, or the lowest shared)
    rows = []
    for model, items in pool.items():
        common = 0.125
        arrays = []
        for t, c, _ in items:
            if common in c.budgets:
                j = c.bi(common)
                arrays.append((t, c.drop, c.full.astype(float), c.plain[:, j].astype(float), c.open().astype(float)))
        if not arrays:
            continue
        task = np.concatenate([[a[0]] * len(a[1]) for a in arrays])
        D, F, P, O = (np.concatenate([a[q] for a in arrays]) for q in (1, 2, 3, 4))
        G = np.where(O > 0, P, F)
        groups = [np.where(task == t)[0] for t in np.unique(task)]
        sels = []
        for _ in range(reps):
            i = strat_idx(rng, groups)
            sels.append(G[i].mean() - (O[i].mean() * P[i].mean() + (1 - O[i].mean()) * F[i].mean()))
        rows.append((model, len(arrays), len(D), f3(O.mean()), f3(F.mean()), f3(P.mean()), f3(G.mean()), f3(O.mean() * P.mean() + (1 - O.mean()) * F.mean()), s3(G.mean() - P.mean()), s3(G.mean() - (O.mean() * P.mean() + (1 - O.mean()) * F.mean())), ci3(ci(sels), True)))
    table("A6-T3", "Pooled over logged LongBench subtasks per model, b = 1/8, substring accuracy",
          ["model", "subtasks", "N", "open", "full", "plain", "PAGE", "random", "PAGE - plain", "selection", "95% CI"], rows)

    # E9 tasks
    rows = []
    for slug, name, files in [("qwen15b", "Qwen2.5-1.5B", ["capbound_4k_qwen15b.jsonl"]), ("qwen3b", "Qwen2.5-3B", ["capbound_4k_qwen3b.jsonl"]), ("mistral7b", "Mistral-7B", ["capbound_4k_mistral7b.jsonl", "capbound_4k_mistral7b_cwe.jsonl"])]:
        by = collections.defaultdict(lambda: collections.defaultdict(dict))
        for f in files:
            rr, _ = read_jsonl(f)
            for r in rr:
                by[r["task"]][r["id"]][r["budget"]] = r
        for t in ["niah_multiquery", "qa_2", "niah_single_1", "cwe"]:
            ids = sorted(by[t])
            if not ids:
                continue
            budgets = sorted({b for i in ids for b in by[t][i]})
            D = np.array([by[t][i][budgets[0]]["drop"] for i in ids])
            F = np.array([bool(by[t][i][budgets[0]]["correct_full"]) for i in ids])
            P = np.array([[bool(by[t][i][b]["correct_plain_shared"]) for b in budgets] for i in ids])
            j = 0
            op = D >= TAU
            G = np.where(op, P[:, j], F)
            po = op.mean()
            rnd = po * P[:, j].mean() + (1 - po) * F.mean()
            h, s = F & ~P[:, j], F & P[:, j]
            ag = np.where(op[:, None], P, F[:, None]).mean() - P.mean()
            rows.append((name, t, len(ids), f3(po), f3(F.mean()), f3(P[:, j].mean()), f3(G.mean()), f3(rnd), s3(G.mean() - rnd), f"{h.sum()}/{s.sum()}", f3(auc(-D[h], -D[s])), s3(ag)))
    table("A6-T4", "Extra RULER tasks (E9), shared-mask arm, b = 1/16", ["model", "task", "N", "open", "full", "plain", "PAGE", "random", "selection", "harmed/survived", "AUC(D)", "PAGE - plain, grid"], rows)

    # E11 prevalence recomputed
    panel = [("longbench_qwen14b.jsonl", None), ("longbench_realistic_lcc_qwen14b.jsonl", None), ("longbench_realistic_repobench-p_qwen14b.jsonl", None), ("longbench_realistic_hotpotqa_qwen14b.jsonl", None), ("longbench_passcount_qwen14b.jsonl", None)]
    tot, n_floor, n_native, n_harm, n_closed, n_fc, n_fc_closed = 0, 0, 0, 0, 0, 0, 0
    rows = []
    for f, _ in panel:
        rr, _b = read_jsonl(f)
        for t in sorted({r["task"] for r in rr}):
            c = Cell(f, tasks=[t])
            fl = [j for j, b in enumerate(c.budgets) if b >= 0.125]
            low = c.drop < TAU
            harmed = c.full & (~c.plain[:, fl]).any(axis=1)
            k_f = int((harmed & low).sum())
            k_n = int((c.full & low & (~c.plain).any(axis=1)).sum())
            rows.append((t, c.n, f3(c.full.mean()), s3(float(c.drop.mean())), f3(c.open().mean()), k_f, k_n, int(harmed.sum()), f3(float(low[c.full].mean()) if c.full.any() else float("nan"))))
            tot += c.n
            n_floor += k_f
            n_native += k_n
            n_harm += int(harmed.sum())
            n_fc += int(c.full.sum())
            n_fc_closed += int((c.full & low).sum())
    table("A6-T5", "Capacity-bound share as E11 defines it (full correct, destroyed at some budget, D below tau), Qwen2.5-14B", ["subtask", "N", "A_full", "mean D", "open", "harmed and closed, b >= 1/8", "harmed and closed, native floor", "all harmed, b >= 1/8", "closed share of full-correct"], rows,
          f"Pooled: {n_floor}/{tot} = {n_floor / tot:.4f} at the b >= 1/8 floor, {n_native}/{tot} = {n_native / tot:.4f} at each subtask's native floor. This counts only inputs with D below tau, so it is not independent of the gate. "
          f"The gate closes on {n_floor} of the {n_harm} full-correct inputs harmed at b >= 1/8 ({n_floor / max(1, n_harm):.3f}), against {n_fc_closed} of all {n_fc} full-correct inputs ({n_fc_closed / max(1, n_fc):.3f}).")
    check("A6 E11 floor count", n_floor, 18, 0)
    check("A6 E11 native count", n_native, 28, 0)
    check("A6 E11 total", tot, 493, 0)


# ---------------------------------------------------------------- A7
def whole_task_gain(task, f, p, closed):
    tasks = np.unique(task)
    ben = {t: (f[task == t] - p[task == t]).mean() for t in tasks}
    share = {t: (task == t).mean() for t in tasks}
    rem, gain = closed, 0.0
    for t in sorted(tasks, key=lambda u: -ben[u]):
        take = min(rem, share[t])
        gain += take * ben[t]
        rem -= take
        if rem <= 1e-12:
            break
    return gain


def a7(args):
    print("\n## A7. Task-level ceiling (corrects `main.tex:326`, `appendix_content.tex:798-800`)\n")
    print("Whole-task policy: closes whole tasks in order of benefit, using labels and hindsight, until it reaches PAGE's closed fraction, taking the expected value of a partial last task. Out-of-sample version: tasks are ordered on one half of the inputs and evaluated on the other.\n")
    rng = np.random.default_rng(SEED + 7)
    reps = min(args.reps, 2000)
    rows, stats = [], []
    for pol in POLICIES:
        for slug, name in MODELS:
            c = cell(matrix_file(pol, slug))
            f, p = c.full.astype(float), c.plain.astype(float).mean(1)
            g = c.gated().astype(float).mean(1)
            closed = 1 - c.open().mean()
            page = g.mean() - p.mean()
            wt = whole_task_gain(c.task, f, p, closed)
            mk3_only = ((f - p) * (c.task == MK3)).mean()
            allfull = f.mean() - p.mean()
            # out of sample
            oos_w, oos_p = [], []
            for _ in range(40):
                a, b = [], []
                for gr in c.task_groups():
                    pm = rng.permutation(gr)
                    a.append(pm[: len(pm) // 2])
                    b.append(pm[len(pm) // 2:])
                A, B = np.concatenate(a), np.concatenate(b)
                tasks = np.unique(c.task)
                ben = {t: (f[A][c.task[A] == t] - p[A][c.task[A] == t]).mean() for t in tasks}
                share = {t: (c.task[B] == t).mean() for t in tasks}
                cl = 1 - c.open()[B].mean()
                rem, gain = cl, 0.0
                for t in sorted(tasks, key=lambda u: -ben[u]):
                    take = min(rem, share[t])
                    gain += take * (f[B][c.task[B] == t] - p[B][c.task[B] == t]).mean()
                    rem -= take
                oos_w.append(gain)
                oos_p.append(g[B].mean() - p[B].mean())
            d = []
            for _ in range(reps):
                i = rng.integers(0, c.n, c.n)
                d.append(whole_task_gain(c.task[i], f[i], p[i], 1 - c.open()[i].mean()) - (g[i].mean() - p[i].mean()))
            stats.append((page, wt, mk3_only, allfull, np.mean(oos_w), np.mean(oos_p)))
            rows.append((POLICY_NAME[pol], name, f3(closed), s3(page), s3(mk3_only), s3(wt), s3(wt - page), ci3(ci(d), True), s3(float(np.mean(oos_w))), s3(float(np.mean(oos_p))), s3(allfull)))
    table("A7-T1", "PAGE against task-level policies at the same closed fraction",
          ["evictor", "model", "closed", "PAGE gain", "MK3-only oracle (Table 14)", "whole-task, hindsight", "whole-task - PAGE", "95% CI", "whole-task, out-of-sample", "PAGE on the same splits", "always full"], rows)
    st = np.array(stats)
    print(f"\nGrand means: PAGE {st[:, 0].mean():+.3f}, MK3-only oracle {st[:, 2].mean():+.3f}, whole-task hindsight {st[:, 1].mean():+.3f}, whole-task out-of-sample {st[:, 4].mean():+.3f}, PAGE on those splits {st[:, 5].mean():+.3f}, always full {st[:, 3].mean():+.3f}.")
    print(f"PAGE as a share of the hindsight ceiling: {st[:, 0].mean() / st[:, 1].mean() * 100:.1f}%. Of the out-of-sample whole-task policy: {st[:, 5].mean() / st[:, 4].mean() * 100:.1f}%.")
    print(f"Hindsight whole-task >= PAGE (within 0.0005): {int((st[:, 1] >= st[:, 0] - 5e-4).sum())}/16. Exceeds by more than 0.005: {int((st[:, 1] - st[:, 0] > 0.005).sum())}/16. PAGE exceeds by more than 0.005: {int((st[:, 0] - st[:, 1] > 0.005).sum())}/16.")
    print(f"PAGE beats the MK3-only oracle by more than 0.005 in {int((st[:, 0] - st[:, 2] > 0.005).sum())}/16 cells; it matches within 0.0005 in {int((np.abs(st[:, 0] - st[:, 2]) <= 5e-4).sum())}/16.")
    # joint bootstrap of the share
    P_, W_ = np.zeros(reps), np.zeros(reps)
    for slug, _ in MODELS:
        cs = [cell(matrix_file(pol, slug)) for pol in POLICIES]
        for _r in range(reps):
            i = rng.integers(0, cs[0].n, cs[0].n)
            for c in cs:
                f, p = c.full.astype(float)[i], c.plain.astype(float).mean(1)[i]
                g = c.gated().astype(float).mean(1)[i]
                P_[_r] += g.mean() - p.mean()
                W_[_r] += whole_task_gain(c.task[i], f, p, 1 - c.open()[i].mean())
    print(f"Joint bootstrap ({reps} replicates): PAGE share of the hindsight ceiling {ci(P_ / W_)[0] * 100:.1f}% to {ci(P_ / W_)[1] * 100:.1f}%. Ceiling minus PAGE over 16 cells {ci3(ci((W_ - P_) / 16), True)}.")


SECTIONS = {"A0": a0, "A1": a1, "A2": a2, "A3": a3, "A4": a4, "A6": a6, "A7": a7}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", default=",".join(SECTIONS))
    ap.add_argument("--reps", type=int, default=2000, help="bootstrap replicates (A2 always uses 10,000)")
    args = ap.parse_args()
    print(f"# PAGE review analyses, seed {SEED}, default bootstrap replicates {args.reps}")
    for s in args.sections.split(","):
        SECTIONS[s.strip()](args)
    print("\n## Assertions\n")
    if FAILS:
        print(f"CHECK: FAIL ({len(FAILS)} assertions): {FAILS}")
        sys.exit(1)
    print("CHECK: PASS (all assertions against published values hold)")


if __name__ == "__main__":
    main()
