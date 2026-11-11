#!/usr/bin/env python3
"""Analyze B8 (PAGE on realistic near-tie retrieval) under preregistration/b8_neartie_prereg.md. Read-only: prints markdown.

Workloads: each near-tie dataset (nq30, babilong) mixed with the complete B7 L-Eval-arm run of the same model, which used the
same prefill, D, SnapKV settings and budgets. Selection = PAGE minus a random gate at the same open rate.
Reuses the checked helpers in checks/verify_2026-11-07.py, as analyze_b.py and analyze_b7.py do.
A B8 run is analyzed only when its log has the runner's "b8 done" line and 5 rows per question.

Usage: python3 -I analyze_b8.py [results_dir] [logs_dir]
"""
import collections
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("verify", str(Path(__file__).resolve().parent / "checks" / "verify_2026-11-07.py"))
V = importlib.util.module_from_spec(spec)
sys.modules["verify"] = V
spec.loader.exec_module(V)
auc, ci = V.auc, V.ci
TAU, THETA = V.TAU, V.THETA_Z
RES = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[2] / "results" / "revision")
LOGS = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).resolve().parents[2] / "results" / "revision" / "logs")
REPS = 2000
BUDGETS = [0.5, 0.25, 0.125, 0.0625]
PRIMARY_B = 0.0625
MODELS = [("Qwen2.5-3B", "qwen3b"), ("Mistral-7B", "mistral7b"), ("Qwen2.5-14B", "qwen14b")]
NEAR = [("nq30", ["nq30"]), ("babilong", ["babilong_qa2", "babilong_qa3"])]
LEVAL_TASKS = ["tpo", "quality", "coursera", "topic_retrieval_longchat", "gsm100", "sci_fi"]
SHARES = [0.10, 0.25, 0.50]
SUBDRAWS = 200
HARM_SCREEN = 0.15


def table(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("|" + "---|" * len(headers))
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")


def f3(x):
    return f"{x:+.3f}"


def rows_of(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def b8_complete(stem):
    path, log = os.path.join(RES, stem + ".jsonl"), os.path.join(LOGS, stem + ".log")
    if not (os.path.exists(path) and os.path.exists(log)):
        print(f"- {stem}: no log yet")
        return False
    if not any(ln.startswith("b8 done") for ln in open(log)):
        print(f"- {stem}: running, not analyzed until complete")
        return False
    ids = collections.Counter(r["id"] for r in rows_of(path))
    if any(c != 5 for c in ids.values()):
        print(f"- {stem}: a question lacks a budget; not analyzed")
        return False
    return True


def b7_complete(stem):
    log = os.path.join(LOGS, stem + ".log")
    return os.path.exists(os.path.join(RES, stem + ".jsonl")) and os.path.exists(log) and any(ln.startswith("b7 done") for ln in open(log))


def load(path, unit_of):
    by = collections.defaultdict(dict)
    for r in rows_of(path):
        by[(r["task"], r["id"])][float(r["budget"])] = r
    keys = sorted(k for k in by if sorted(by[k]) == sorted([1.0] + BUDGETS))
    assert len(keys) == len(by), "a question is missing a budget"
    for k in keys:
        assert len({by[k][b]["drop"] for b in by[k]}) == 1 and len({by[k][b]["T"] for b in by[k]}) == 1
    g = lambda f: [by[k][1.0][f] for k in keys]
    return dict(task=np.array(g("task")), D=np.array(g("drop"), float), T=np.array(g("T"), float),
                F=np.array(g("correct_plain"), bool),
                P={b: np.array([by[k][b]["correct_plain"] for k in keys], bool) for b in BUDGETS},
                kept={b: np.array([by[k][b]["n_kept_plain"] / by[k][1.0]["T"] for k in keys], float) for b in BUDGETS},
                unit=np.array([unit_of(by[k][1.0], k) for k in keys]))


def cat(a, b):
    return dict(task=np.concatenate([a["task"], b["task"]]), D=np.concatenate([a["D"], b["D"]]), T=np.concatenate([a["T"], b["T"]]),
                F=np.concatenate([a["F"], b["F"]]), P={x: np.concatenate([a["P"][x], b["P"][x]]) for x in BUDGETS},
                kept={x: np.concatenate([a["kept"][x], b["kept"][x]]) for x in BUDGETS}, unit=np.concatenate([a["unit"], b["unit"]]))


def cut(c, m):
    return dict(task=c["task"][m], D=c["D"][m], T=c["T"][m], F=c["F"][m], P={x: c["P"][x][m] for x in BUDGETS},
                kept={x: c["kept"][x][m] for x in BUDGETS}, unit=c["unit"][m])


def strata(c):
    out = []
    for t in sorted(set(c["task"])):
        d = collections.defaultdict(list)
        for i in np.flatnonzero(c["task"] == t):
            d[c["unit"][i]].append(i)
        out.append([np.array(v) for v in d.values()])
    return out


def resample(st, rng):
    return np.concatenate([np.concatenate([gs[j] for j in rng.integers(0, len(gs), len(gs))]) for gs in st])


def sel(op, F, P):
    G = np.where(op, P, F)
    po = op.mean()
    return G.mean() - (po * P.mean() + (1 - po) * F.mean())


def boot(c, gate, b, rng):
    """Point estimate, 95% CI and two-sided bootstrap p of selection. The z threshold is re-fit in each replicate."""
    op = gate_open(c["D"], gate)
    est = sel(op, c["F"], c["P"][b])
    st = strata(c)
    reps = []
    for _ in range(REPS):
        i = resample(st, rng)
        reps.append(sel(gate_open(c["D"][i], gate), c["F"][i], c["P"][b][i]))
    reps = np.array(reps)
    p = min(1.0, 2 * min((reps <= 0).mean(), (reps >= 0).mean()))
    return est, ci(reps), p, op


def gate_open(D, gate):
    thr = TAU if gate == "tau" else D.mean() + THETA * D.std()
    return D >= thr


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def main():
    print("## B8. PAGE on realistic near-tie retrieval mixed into L-Eval (S-Q2, G-W3, G-Q1, R-Q1)\n")
    data = {}
    for name, slug in MODELS:
        b7 = f"b7_leval_{slug}"
        if not b7_complete(b7):
            print(f"- {b7}: B7 background not complete; {name} not analyzed")
            continue
        bg = load(os.path.join(RES, b7 + ".jsonl"), lambda r, k: r["doc"] if r["task"] != "gsm100" else f"q{k[1]}")
        assert set(bg["task"]) == set(LEVAL_TASKS)
        for ds, _ in NEAR:
            stem = f"b8_{ds}_{slug}"
            if b8_complete(stem):
                data[(name, ds)] = (load(os.path.join(RES, stem + ".jsonl"), lambda r, k: r["doc"]), bg)

    # T1: each near-tie dataset alone: accuracy, harm, D against L-Eval tasks (all gate-free except AUC)
    rows = []
    screen = {}
    for (name, ds), (c, bg) in data.items():
        for t in sorted(set(c["task"])):
            m = c["task"] == t
            F, P = c["F"][m], c["P"][PRIMARY_B][m]
            harmed = (F & ~P).sum()
            hr = harmed / max(F.sum(), 1)
            lev_means = {lt: bg["D"][bg["task"] == lt].mean() for lt in LEVAL_TASKS}
            rank = 1 + sum(v < c["D"][m].mean() for v in lev_means.values())
            Dm = c["D"][m]
            a = auc(-Dm[F & ~P], -Dm[F & P])  # low D read as harm; NaN if a side is empty
            rows.append([name, t, int(m.sum()), f"{F.mean():.3f}"] + [f"{c['P'][b][m].mean():.3f}" for b in [0.0625, 0.125, 0.25, 0.5]]
                        + [f"{harmed}/{F.sum()}", f"{hr:.3f}", f"{c['D'][m].mean():+.4f}", f"{min(lev_means.values()):+.4f}", f"{rank} of 7", f"{a:.3f}"])
        Fa, Pa = c["F"], c["P"][PRIMARY_B]
        screen[(name, ds)] = (Fa & ~Pa).sum() / max(Fa.sum(), 1)
    print("### B8-T1: Near-tie datasets alone: accuracy by budget, harm at 1/16, and mean D against the six L-Eval tasks\n")
    table(["model", "task", "N", "full", "1/16", "1/8", "1/4", "1/2", "harmed/full correct", "harm rate", "mean D", "lowest L-Eval mean D",
           "rank of mean D (1 = lowest)", "AUC(D) vs harm, within task"], rows)
    print("\nHarm = correct at full cache and wrong under plain SnapKV at 1/16. AUC reads low D as more harm. Gate-free screen "
          f"(pre-registered): a dataset is capacity-bound for a model if its pooled harm rate is at least {HARM_SCREEN}.\n")

    # T2: primary. Mixed workload, z-gate and tau, b = 1/16 (primary) and 1/8
    print("### B8-T2: Mixed workloads (near-tie dataset + the model's 920 L-Eval questions): selection against a random gate\n")
    rows, prim, keys = [], [], []
    for (name, ds), (c, bg) in data.items():
        rng = np.random.default_rng(20261110 + 10 * [m for m, _ in MODELS].index(name) + [d for d, _ in NEAR].index(ds))
        w = cat(c, bg)
        near = np.isin(w["task"], dict(NEAR)[ds])
        for b in (0.0625, 0.125):
            for gate in ("z", "tau"):
                est, (lo, hi), p, op = boot(w, gate, b, rng)
                full, plain = w["F"].mean(), w["P"][b].mean()
                page = np.where(op, w["P"][b], w["F"]).mean()
                closed_near = (~op[near]).mean()
                closed_bg = (~op[~near]).mean()
                r = [name, ds, f"{near.mean():.3f}", bname(b), gate, f"{op.mean():.3f}", f"{closed_near:.3f}", f"{closed_bg:.3f}",
                     f"{full:.3f}", f"{plain:.3f}", f"{page:.3f}", f3(est), f"[{lo:+.3f}, {hi:+.3f}]", f"{p:.4f}"]
                rows.append(r)
                if b == PRIMARY_B and gate == "z":
                    prim.append(p)
                    keys.append((name, ds, est, lo))
    adj = holm(np.array(prim)) if prim else []
    table(["model", "near-tie dataset", "near-tie share", "b", "gate", "open", "closed on near-tie", "closed on L-Eval", "full", "plain", "PAGE",
           "selection", "95% CI", "bootstrap p"], rows)
    print("\nz threshold = mean D + (-0.69) SD over the whole mixed workload, re-fit in every bootstrap replicate. Bootstrap: within each "
          "task, documents resampled (NQ and BABILong: each question; gsm100: each question), 2000 replicates.\n")
    print("### B8-T3: Primary test, z-gate at 1/16, Holm across the (model, dataset) tests\n")
    table(["model", "near-tie dataset", "selection", "CI lower", "Holm-adjusted p", "harm screen passed"],
          [[k[0], k[1], f3(k[2]), f"{k[3]:+.3f}", f"{a:.4f}", "yes" if screen[(k[0], k[1])] >= HARM_SCREEN else "no"] for k, a in zip(keys, adj)])
    for ds, _ in NEAR:
        wins = [k for k, a in zip(keys, adj) if k[1] == ds and a < 0.05 and k[2] > 0]
        print(f"\n- {ds}: positive selection with Holm-adjusted p < 0.05 on {len(wins)} of {sum(k[1] == ds for k in keys)} models. "
              f"Pre-registered win: at least 2 of 3. {'WIN' if len(wins) >= 2 else 'no win'}.")

    # T4: share sweep, z-gate at 1/16
    print("\n### B8-T4: Near-tie share sweep, z-gate, 1/16 (mean selection over subsample draws)\n")
    rows = []
    for (name, ds), (c, bg) in data.items():
        rng = np.random.default_rng(777 + 10 * [m for m, _ in MODELS].index(name) + [d for d, _ in NEAR].index(ds))
        for s in SHARES:
            vals = []
            for _ in range(SUBDRAWS):
                nn, nb = len(c["D"]), len(bg["D"])
                if s * (nn + nb) <= nn:  # keep all background, subsample the near-tie set
                    k = int(round(s * nb / (1 - s)))
                    w = cat(cut(c, rng.choice(nn, k, replace=False)), bg)
                else:  # keep all near-tie, subsample the background
                    k = int(round(nn * (1 - s) / s))
                    w = cat(c, cut(bg, rng.choice(nb, k, replace=False)))
                op = gate_open(w["D"], "z")
                vals.append(sel(op, w["F"], w["P"][PRIMARY_B]))
            vals = np.array(vals)
            rows.append([name, ds, f"{s:.2f}", f3(vals.mean()), f"[{np.percentile(vals, 2.5):+.3f}, {np.percentile(vals, 97.5):+.3f}]"])
    table(["model", "near-tie dataset", "near-tie share", "mean selection", "2.5 to 97.5 percentile over draws"], rows)
    print("\nThe interval here is the spread over subsample draws, not a sampling CI.\n")

    # T5: near-tie dataset alone, z within the dataset (per-input selection)
    print("### B8-T5: Near-tie dataset alone, z threshold within the dataset (per-input selection)\n")
    rows = []
    for (name, ds), (c, bg) in data.items():
        rng = np.random.default_rng(555 + 10 * [m for m, _ in MODELS].index(name) + [d for d, _ in NEAR].index(ds))
        for b in (0.0625, 0.125):
            est, (lo, hi), p, op = boot(c, "z", b, rng)
            rows.append([name, ds, bname(b), f"{op.mean():.3f}", f3(est), f"[{lo:+.3f}, {hi:+.3f}]"])
    table(["model", "dataset", "b", "open", "selection", "95% CI"], rows)


def bname(b):
    return {0.0625: "1/16", 0.125: "1/8", 0.25: "1/4", 0.5: "1/2"}[b]


if __name__ == "__main__":
    main()
