#!/usr/bin/env python3
"""Analyze B1 (Table 41), B2, B2b and B2c (Table 47), B3 (Table 43), B5 and B6 (Table 44). Read-only: prints markdown tables.

Reuses the checked helpers in checks/verify_2026-11-07.py. Each section
seeds its own generator, so its intervals do not depend on which sections ran first.
Logs are read from experiments/results/revision/ (override with PAGE_REV_RESULTS).
B2's gov_report rows need b2_longbench_qwen14b_rougeL.jsonl from score_rouge.py.

Usage: python3 -I analyze_b.py [b1] [b2] [b2c] [b3] [b5] [b6]
"""
import collections
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("verify", str(HERE / "checks" / "verify_2026-11-07.py"))
V = importlib.util.module_from_spec(spec)
sys.modules["verify"] = V
spec.loader.exec_module(V)
auc, ci, f3, s3, ci3, table, within_auc, strat_idx = V.auc, V.ci, V.f3, V.s3, V.ci3, V.table, V.within_auc, V.strat_idx
MK3, TAU, THETA = V.MK3, V.TAU, V.THETA_Z
OUT = os.environ.get("PAGE_REV_RESULTS", str(Path(__file__).resolve().parents[2] / "results" / "revision"))
REPS = 2000
rng = np.random.default_rng(20261108)

B1_MODELS = [("Qwen2.5-1.5B", "qwen15b"), ("Qwen2.5-3B", "qwen3b"), ("Mistral-7B", "mistral7b"), ("Qwen2.5-14B", "qwen14b")]
MAIN = {"Qwen2.5-1.5B", "Qwen2.5-3B", "Mistral-7B"}
BUDGET_SIGNALS = ["snapkv", "h2o", "obcache_v", "obcache_k", "obcache_vk"]


def rows_of(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def join_b1(slug):
    path = os.path.join(OUT, f"b1_signals_{slug}.jsonl")
    if not os.path.exists(path):
        return None
    rows = rows_of(path)
    c = V.cell(f"gated_4k_{slug}.jsonl")
    idx_by_task = {}
    for t in np.unique(c.task):
        ii = np.where(c.task == t)[0]
        idx_by_task[t] = ii[np.argsort(c.id[ii])]
    keep, sig, mism = [], [], 0
    for r in rows:
        ii = idx_by_task[r["task"]]
        if r["id"] >= len(ii):
            continue
        i = ii[r["id"]]
        if int(c.T[i]) != int(r["T"]):
            mism += 1
            continue
        keep.append(i)
        sig.append(r)
    keep = np.array(keep)
    return c, keep, sig, mism


def signal_table(sig):
    """name -> values, for one model."""
    d = {"D (logged)": None, "D (this run)": np.array([r["drop"] for r in sig]), "a_early": np.array([r["a_early"] for r in sig]),
         "entropy_drop": np.array([r["entropy_drop"] for r in sig]), "top32_share": np.array([r["topk_share"] for r in sig])}
    for s in BUDGET_SIGNALS:
        for b in ("0.0625", "0.125", "0.25", "0.5"):
            d[f"{s}@{b}"] = np.array([r["shares"][s][b] for r in sig], float)
    for b in ("0.0625", "0.125", "0.25", "0.5"):
        d[f"act_change@{b}"] = np.array([r["act_change"][b] for r in sig], float)
        d[f"kl@{b}"] = np.array([r["kl"][b] for r in sig], float)
        d[f"top1_same@{b}"] = np.array([float(r["top1_same"][b]) for r in sig], float)
    return d


PRIMARY = ["a_early", "entropy_drop", "top32_share"] + [f"{s}@0.125" for s in BUDGET_SIGNALS] + ["act_change@0.125", "kl@0.125", "top1_same@0.125"]


def b1():
    global rng
    rng = np.random.default_rng(20261108)
    print("\n## B1. Scorer-based gates at a matched open rate (R-W4, R-Q1)\n")
    data = {}
    for name, slug in B1_MODELS:
        j = join_b1(slug)
        if j is None:
            print(f"- {name}: no log yet")
            continue
        c, keep, sig, mism = j
        d = signal_table(sig)
        d["D (logged)"] = c.drop[keep]
        full = c.full[keep].astype(float)
        plain = c.plain[keep].astype(float).mean(1)
        task = c.task[keep]
        nc = int((d["D (logged)"] < TAU).sum())
        data[name] = dict(c=c, keep=keep, d=d, full=full, plain=plain, task=task, nc=nc)
        rd = np.corrcoef(d["D (logged)"], d["D (this run)"])[0, 1]
        print(f"- {name}: joined {len(keep)} inputs, T mismatches {mism}, closures at tau {nc}, |D run - logged| max {np.abs(d['D (logged)'] - d['D (this run)']).max():.4f}, correlation {rd:.4f}.")
    if "Qwen2.5-1.5B" not in data:
        return
    # Orientation fixed on the fitting model: the direction that puts MK3 low (closes MK3).
    fit = data["Qwen2.5-1.5B"]
    orient = {}
    for s in PRIMARY + ["D (logged)"]:
        v = fit["d"][s]
        m = fit["task"] == MK3
        orient[s] = 1 if auc(v[~m], v[m]) >= 0.5 else -1

    def gain(D, s_vals, sign, row, idx=None):
        """Close the nc lowest values of sign*signal. Inputs tied at the boundary are
        closed fractionally (expected value over random tie-breaking), so binary
        signals such as top1_same do not depend on input order."""
        n = len(row["full"])
        ii = np.arange(n) if idx is None else idx
        v = sign * s_vals[ii]
        benefit = (row["full"] - row["plain"])[ii]
        mk = (row["task"][ii] == MK3).astype(float)
        nc = row["nc"]
        if nc == 0:
            return 0.0, 0.0
        thr = np.sort(v, kind="mergesort")[nc - 1]
        below, tie = v < thr, v == thr
        frac = (nc - below.sum()) / tie.sum()
        g = (benefit[below].sum() + frac * benefit[tie].sum()) / n
        m = mk[below].sum() + frac * mk[tie].sum()
        return float(g), float(m)

    rows, verdict = [], collections.defaultdict(list)
    for name, row in data.items():
        groups = [np.where(row["task"] == t)[0] for t in np.unique(row["task"])]
        base, base_m = gain(None, row["d"]["D (logged)"], 1, row)
        po = 1 - row["nc"] / len(row["full"])
        rnd = (1 - po) * (row["full"].mean() - row["plain"].mean())
        for s in PRIMARY:
            v = row["d"][s]
            g, m = gain(None, v, orient[s], row)
            g_h = max(gain(None, v, 1, row)[0], gain(None, v, -1, row)[0])
            diffs = []
            for _ in range(REPS):
                idx = strat_idx(rng, groups)
                diffs.append(gain(None, row["d"]["D (logged)"], 1, row, idx)[0] - gain(None, v, orient[s], row, idx)[0])
            lo, hi = ci(diffs)
            verdict[s].append((name, lo, hi))
            mk = row["task"] == MK3
            a = auc(v[~mk], v[mk])
            rows.append((name, s, "low closes" if orient[s] == 1 else "high closes", s3(g), f"{m:.0f}", s3(g_h), s3(base - g), ci3((lo, hi), True), f3(max(a, 1 - a))))
        rows.append((name, "D (logged)", "low closes", s3(base), f"{base_m:.0f}", s3(base), "", "", f3(auc(row["d"]["D (logged)"][~(row["task"] == MK3)], row["d"]["D (logged)"][row["task"] == MK3]))))
        rows.append((name, "random gate", "", s3(rnd), "", "", s3(base - rnd), "", ""))
    table("B1-T1", "Gate gain over plain SnapKV at D's closed count",
          ["model", "signal", "orientation (fixed on Qwen2.5-1.5B)", "gain", "MK3 closed", "gain, hindsight orientation", "D minus signal", "95% CI", "AUC for MK3"], rows)
    rows = []
    for s in PRIMARY:
        main = [x for x in verdict[s] if x[0] in MAIN]
        up = sum(x[1] > 0 for x in main)
        down = sum(x[2] < 0 for x in main)
        rows.append((s, f"{up}/{len(main)}", f"{down}/{len(main)}", "D has an advantage" if up >= 2 else ("signal has an advantage" if down >= 2 else "no clear advantage")))
    table("B1-T2", "Pre-registered decision rule, three main models", ["signal", "models with D above", "models with signal above", "verdict"], rows)
    # Against harm
    rows = []
    for name, row in data.items():
        c, keep = row["c"], row["keep"]
        for b in (0.0625, 0.25):
            j = c.bi(b)
            full = c.full[keep]
            corr = c.plain[keep, j]
            h, s_ = full & ~corr, full & corr
            for s in ["D (logged)", "a_early", "top32_share", f"snapkv@{b}", f"h2o@{b}", f"obcache_v@{b}", f"obcache_k@{b}", f"obcache_vk@{b}", f"act_change@{b}", f"kl@{b}"]:
                v = row["d"][s]
                sign = orient.get(s, None)
                if sign is None:
                    base_name = s.split("@")[0] + "@0.125"
                    sign = orient.get(base_name, 1)
                vv = sign * v
                rows.append((name, V.bname(b), s, int(h.sum()), f3(auc(-vv[h], -vv[s_])), f3(within_auc(row["task"], vv, h, s_)[0])))
    table("B1-T3", "Against per-input harm, in the orientation that closes MK3", ["model", "b", "signal", "harmed", "pooled AUC", "within-task AUC"], rows,
          "A pooled AUC well below 0.5 means the opposite orientation predicts pooled harm. That is a secondary reading, outside the pre-registered rule.")
    # Secondary: budget-specific signals at b = 1/16 used as gates, both orientations
    rows = []
    for name, row in data.items():
        base = gain(None, row["d"]["D (logged)"], 1, row)[0]
        for s in [f"{x}@0.0625" for x in BUDGET_SIGNALS] + ["act_change@0.0625", "kl@0.0625"]:
            v = row["d"][s]
            lo_g, lo_m = gain(None, v, 1, row)
            hi_g, hi_m = gain(None, v, -1, row)
            rows.append((name, s, s3(lo_g), f"{lo_m:.0f}", s3(hi_g), f"{hi_m:.0f}", s3(base)))
    table("B1-T4", "Secondary: b = 1/16 versions of the signals as gates, both orientations, D's closed count",
          ["model", "signal", "gain, low closes", "MK3 closed", "gain, high closes", "MK3 closed", "D gain"], rows,
          "Not part of the pre-registered rule. Either orientation is chosen with hindsight here.")


def load_b2(path):
    by = collections.defaultdict(dict)
    for r in rows_of(path):
        by[(r["task"], r["id"])][float(r["budget"])] = r
    keys = [k for k in sorted(by) if 1.0 in by[k]]
    return by, keys


def b2_cell(by, keys, task, officials=None):
    ks = [k for k in keys if k[0] == task]
    budgets = sorted({b for k in ks for b in by[k] if b < 1})
    ks = [k for k in ks if all(b in by[k] for b in budgets)]
    D = np.array([by[k][1.0]["drop"] for k in ks], float)
    T = np.array([by[k][1.0]["T"] for k in ks], float)
    F = np.array([bool(by[k][1.0]["correct_plain"]) for k in ks])
    P = np.array([[bool(by[k][b]["correct_plain"]) for b in budgets] for k in ks])
    pf = np.array([[bool(by[k][b].get("parse_fail")) for b in budgets + [1.0]] for k in ks])
    kept = np.array([[by[k][b]["n_kept_plain"] / by[k][1.0]["T"] for b in budgets] for k in ks], float)
    doc = np.array([by[k][1.0].get("doc") or f"i{k[1]}" for k in ks])
    Ae = []
    for k in ks:
        pl = by[k][1.0].get("per_layer")
        Ae.append(np.mean(pl[: max(1, len(pl) // 3)]) if pl else np.nan)
    return dict(keys=ks, budgets=budgets, D=D, T=T, F=F, P=P, pf=pf, kept=kept, doc=doc, Ae=np.array(Ae, float))


def doc_groups(doc):
    d = collections.defaultdict(list)
    for i, x in enumerate(doc):
        d[x].append(i)
    return [np.array(v) for v in d.values()]


def doc_resample(groups):
    pick = rng.integers(0, len(groups), len(groups))
    return np.concatenate([groups[i] for i in pick])


def sel_ci(groups, op, F, Pj, reps=REPS):
    """Selection (PAGE minus a random gate at the same open rate), bootstrapped by document.
    F and Pj are per-input scores at full cache and at the budget: 0/1 or continuous."""
    G = np.where(op, Pj, F)
    out = []
    for _ in range(reps):
        idx = doc_resample(groups)
        po = op[idx].mean()
        out.append(G[idx].mean() - (po * Pj[idx].mean() + (1 - po) * F[idx].mean()))
    return ci(out)


def auc_diff_ci(groups, a, d, h, s_, reps=REPS):
    """AUC(a) minus AUC(d) against harm, both read as low value means harm, by document."""
    out = []
    for _ in range(reps):
        idx = doc_resample(groups)
        hh, ss = h[idx], s_[idx]
        if hh.sum() == 0 or ss.sum() == 0:
            continue
        out.append(auc(-a[idx][hh], -a[idx][ss]) - auc(-d[idx][hh], -d[idx][ss]))
    return ci(out)


def official_scorer(stem, task, by):
    """Per-input LongBench official score function (key, budget) -> score, or None."""
    if task == "gov_report":
        path = os.path.join(OUT, stem + "_rougeL.jsonl")
        if not os.path.exists(path):
            return None
        sc = {(r["task"], r["id"], float(r["budget"])): r["rougeL"] for r in rows_of(path)}
        return lambda k, b: sc[(k[0], k[1], b)]
    if task not in V.OFFICIAL and task not in ("2wikimqa", "musique"):
        return None
    fn = V.OFFICIAL.get(task, V.qa_f1)

    def score(k, b):
        r = by[k][b]
        p = r["pred_plain"]
        if task in V.FIRST_LINE:
            p = p.lstrip("\n").split("\n")[0]
        return max((fn(p, a) for a in r["answers"]), default=0.0)

    return score


B2_FILES = [  # (arm, model, stem)
    ("B2 letter", "Qwen2.5-3B", "b2_quality_qwen3b"), ("B2 letter", "Mistral-7B", "b2_quality_mistral7b"),
    ("B2 letter", "Qwen2.5-3B", "b2_lbv2_qwen3b"), ("B2 letter", "Mistral-7B", "b2_lbv2_mistral7b"),
    ("B2b quote", "Qwen2.5-3B", "b2b_quality_qwen3b"), ("B2b quote", "Mistral-7B", "b2b_quality_mistral7b"),
    ("B2b quote", "Qwen2.5-3B", "b2b_lbv2_qwen3b"), ("B2b quote", "Mistral-7B", "b2b_lbv2_mistral7b"),
    ("B2 LongBench", "Qwen2.5-3B", "b2_longbench_qwen3b"), ("B2 LongBench", "Qwen2.5-14B", "b2_longbench_qwen14b"),
]


B2C_FILES = [("B2c letter", "Qwen2.5-14B", "b2c_quality_qwen14b_letter"), ("B2c quote", "Qwen2.5-14B", "b2c_quality_qwen14b_quote")]


def complete(path, n_rows):
    """Analyze a new run only once its log is complete, so no interim result is ever printed."""
    n = sum(1 for _ in open(path)) if os.path.exists(path) else 0
    if n < n_rows:
        print(f"- {os.path.basename(path)}: {n} of {n_rows} rows, not analyzed until complete")
        return False
    return True


def b2_cells(files):
    """B2's per-cell computation, shared by b2() and b2c(). Uses the caller's rng, in file order."""
    rows, off, harm_rows, chk, chk_off = [], [], [], [], []
    for arm, model, stem in files:
        path = os.path.join(OUT, stem + ".jsonl")
        if not os.path.exists(path):
            print(f"- {stem}: no log yet")
            continue
        by, keys = load_b2(path)
        for task in sorted({k[0] for k in keys}):
            c = b2_cell(by, keys, task)
            n = len(c["keys"])
            if n == 0:
                continue
            groups = doc_groups(c["doc"])
            tz = c["D"].mean() + THETA * c["D"].std()
            gates = (("tau=0.07", TAU), (f"z-gate ({tz:.3f})", tz))
            F = c["F"].astype(float)
            for b in (0.0625, 0.125):
                if b not in c["budgets"]:
                    continue
                j = c["budgets"].index(b)
                Pj = c["P"][:, j].astype(float)
                h, s_ = c["F"] & ~c["P"][:, j], c["F"] & c["P"][:, j]
                aD, aE = auc(-c["D"][h], -c["D"][s_]), auc(-c["Ae"][h], -c["Ae"][s_])
                dlo, dhi = auc_diff_ci(groups, c["Ae"], c["D"], h, s_) if h.sum() and s_.sum() else (float("nan"), float("nan"))
                if b == 0.0625:
                    harm_rows.append((arm, model, task, n, int(c["F"].sum()), int(h.sum()), f3(h.sum() / max(1, c["F"].sum())), f"{c['pf'][:, j].mean():.3f}", f"{c['pf'][:, -1].mean():.3f}"))
                for gate, tau in gates:
                    op = c["D"] >= tau
                    G = np.where(op, Pj, F)
                    po = op.mean()
                    rnd = po * Pj.mean() + (1 - po) * F.mean()
                    kept = np.where(op, c["kept"][:, j], 1.0).mean()
                    lo, hi = sel_ci(groups, op, F, Pj)
                    rows.append((arm, model, task, n, len(groups), V.bname(b), gate, f3(po), f3(kept), f3(F.mean()), f3(Pj.mean()), f3(G.mean()), f3(rnd),
                                 s3(G.mean() - Pj.mean()), s3(G.mean() - rnd), ci3((lo, hi), True), f"{h.sum()}/{s_.sum()}", f3(aD), f3(aE), ci3((dlo, dhi), True),
                                 f"{c['pf'][:, j].mean():.3f}"))
                    chk.append(dict(arm=arm, model=model, task=task, b=b, gate=gate.split()[0], po=po, closed=int((~op).sum()), harmed=int(h.sum()),
                                    fc=int(c["F"].sum()), pmp=float(G.mean() - Pj.mean()), lo=lo, hi=hi, aD=aD, dlo=dlo))
            scorer = official_scorer(stem, task, by)
            if scorer is None:
                continue
            sf = np.array([scorer(k, 1.0) for k in c["keys"]])
            for b in (0.0625, 0.125):
                j = c["budgets"].index(b)
                sp = np.array([scorer(k, b) for k in c["keys"]])
                m = sf > 0
                rho = V.spearman(c["D"][m], (sp - sf)[m])
                for gate, tau in gates:
                    op = c["D"] >= tau
                    sg = np.where(op, sp, sf)
                    po = op.mean()
                    rnd = po * sp.mean() + (1 - po) * sf.mean()
                    lo, hi = sel_ci(groups, op, sf, sp)
                    off.append((model, task, n, V.bname(b), gate, f3(po), f3(float(sf.mean())), f3(float(sp.mean())), f3(float(sg.mean())), f3(float(rnd)),
                                s3(float(sg.mean() - sp.mean())), s3(float(sg.mean() - rnd)), ci3((lo, hi), True), s3(rho)))
                    chk_off.append(dict(gate=gate.split()[0], lo=lo, hi=hi))
    return rows, off, harm_rows, chk, chk_off


B2_T1_COLS = ["arm", "model", "dataset", "N", "docs", "b", "gate", "open", "kept", "full", "plain", "PAGE", "random", "PAGE - plain", "selection",
              "95% CI (by document)", "harmed/survived", "AUC(D)", "AUC(a_early)", "AUC(a_early) - AUC(D), 95% CI", "parse failures"]
B2_T3_COLS = ["arm", "model", "dataset", "N", "full correct", "harmed", "harm rate", "parse failures at 1/16", "parse failures at full"]


def b2():
    global rng
    rng = np.random.default_rng(20261108 + 2)
    print("\n## B2. Beyond RULER (S-Q2, G-W3, G-Q1)\n")
    rows, off, harm_rows, chk, chk_off = b2_cells(B2_FILES)
    table("B2-T1", "Accuracy and selection against a random gate at the same open rate, SnapKV",
          ["arm", "model", "dataset", "N", "docs", "b", "gate", "open", "kept", "full", "plain", "PAGE", "random", "PAGE - plain", "selection", "95% CI (by document)",
           "harmed/survived", "AUC(D)", "AUC(a_early)", "AUC(a_early) - AUC(D), 95% CI", "parse failures"], rows,
          "QuALITY and LongBench-v2 are multiple choice. The letter-only arm (B2) scores the first standalone letter A to D; the quote-then-answer arm (B2b) "
          "scores the last `Answer: X`. LongBench subtasks use substring match here, and official metrics in B2-T2. Parse failures count as wrong. "
          "Harm = correct at full cache and wrong at b; both AUCs read a low value as more harm, which is the gate's assumption. "
          f"The z-gate threshold is mean D + ({THETA}) SD over the row's (run, dataset) inputs. Bootstraps resample documents, because QuALITY questions "
          f"share articles; each LongBench input is its own document. {REPS} replicates.")
    table("B2-T2", "LongBench official metrics, both gates",
          ["model", "subtask", "N", "b", "gate", "open", "full", "plain", "PAGE", "random", "PAGE - plain", "selection", "95% CI", "Spearman D vs score change"], off,
          "Official LongBench metrics on the stored predictions: token F1 for hotpotqa, 2wikimqa and musique; edit similarity on the first code line for lcc and repobench-p; "
          "ROUGE-L F (rouge 1.0.1, via score_rouge.py) for gov_report. Spearman uses inputs with a nonzero full-cache score.")
    table("B2-T3", "Harm at b = 1/16, by arm", ["arm", "model", "dataset", "N", "full correct", "harmed", "harm rate", "parse failures at 1/16", "parse failures at full"], harm_rows,
          "Harm rate = harmed / full correct. gov_report has no full-correct input under substring match; B2-T2 scores it with ROUGE-L.")

    # Pre-registered predictions. A unit is (dataset or subtask, model, budget), at b = 1/16 and 1/8.
    # Units with no full-correct input cannot test anything and are left out.
    def count(arms, gate, cond, b=None):
        vals = [cond(x) for x in chk if x["arm"] in arms and x["gate"] == gate and x["fc"] > 0 and (b is None or x["b"] == b)]
        vals = [v for v in vals if v is not None]
        return int(sum(vals)), len(vals)

    def most(k, n):
        return "holds" if n and k > n / 2 else "fails"

    def every(k, n):
        return "holds" if k == n else f"fails on {n - k} of {n}"

    inc0 = lambda x: x["lo"] <= 0 <= x["hi"]
    low_auc = lambda x: None if np.isnan(x["aD"]) else bool(x["aD"] < 0.65)
    no_clear = lambda x: None if np.isnan(x["dlo"]) else bool(not x["dlo"] > 0)
    nonneg = lambda x: bool(x["pmp"] >= -1e-12) if (x["closed"] > 0 and x["harmed"] > 0) else None
    B2A, B2B = ("B2 letter", "B2 LongBench"), ("B2b quote",)
    rows = []
    for pid, arms, text, gate, cond, b, rule in [
        ("B2 P1", B2A, "selection CI includes 0", "tau=0.07", inc0, None, most),
        ("B2 P1", B2A, "selection CI includes 0", "z-gate", inc0, None, most),
        ("B2 P2", B2A, "AUC(D) against harm below 0.65", "tau=0.07", low_auc, None, most),
        ("B2 P3", B2A, "a_early does not clearly beat D (difference CI not above 0)", "tau=0.07", no_clear, None, most),
        ("B2 P4", B2A, "PAGE - plain >= 0 at 1/16 where the gate closes some inputs and eviction harms some", "tau=0.07", nonneg, 0.0625, every),
        ("B2 P4", B2A, "PAGE - plain >= 0 at 1/16 where the gate closes some inputs and eviction harms some", "z-gate", nonneg, 0.0625, every),
        ("B2b P2", B2B, "selection CI includes 0", "tau=0.07", inc0, None, most),
        ("B2b P2", B2B, "selection CI includes 0", "z-gate", inc0, None, most),
        ("B2b P3", B2B, "AUC(D) against harm below 0.65", "tau=0.07", low_auc, None, most),
    ]:
        k, n = count(arms, gate, cond, b)
        rows.append((pid, " + ".join(arms), text, "gate-free" if cond in (low_auc, no_clear) else gate, f"{k}/{n}", rule(k, n)))
    for gate in ("tau=0.07", "z-gate"):
        xs = [x for x in chk_off if x["gate"] == gate]
        k = sum(x["lo"] <= 0 <= x["hi"] for x in xs)
        rows.append(("B2 P1, official metrics", "B2 LongBench", "selection CI includes 0", gate, f"{k}/{len(xs)}", most(k, len(xs))))
    table("B2-T4", "Pre-registered predictions, counted over (dataset, model, budget) units at b = 1/16 and 1/8",
          ["prediction", "arms", "test", "gate", "units meeting it", "verdict"], rows,
          "\"Most\" means more than half of the units. B2b P1 (harm clearly above zero) and B2b P4 (few openings at fixed tau on Qwen2.5-3B) are read from B2-T3 and B2-T1.")
    q3 = [x for x in chk if x["arm"] == "B2b quote" and x["model"] == "Qwen2.5-3B" and x["gate"] == "tau=0.07" and x["b"] == 0.0625]
    print("\nB2b P4 read-out, Qwen2.5-3B at tau = 0.07: open rate " + ", ".join(f"{x['task']} {x['po']:.3f}" for x in q3) + ".")


def harm_rows_a1(name, c):
    """A1-T1's comparators for one model: pooled and within-task AUC of -D against harm,
    task identity in-sample and split-half (50 splits), bootstrap within task."""
    rows = []
    groups = c.task_groups()
    for b in (0.0625, 0.25):
        if b not in c.budgets:
            continue
        j = c.bi(b)
        D, task, full, corr = c.drop, c.task, c.full, c.plain[:, j]
        harm, safe = full & ~corr, full & corr
        rate = {t: harm[task == t].sum() / max(1, full[task == t].sum()) for t in np.unique(task)}
        s = np.array([rate[t] for t in task])
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
        wa, pairs, per = within_auc(task, D, harm, safe)
        bp, bw = [], []
        for _ in range(REPS):
            idx = strat_idx(rng, groups)
            dd, tt, hh, ss = D[idx], task[idx], harm[idx], safe[idx]
            bp.append(auc(-dd[hh], -dd[ss]))
            bw.append(within_auc(tt, dd, hh, ss)[0])
        rows.append((name, V.bname(b), int(full.sum()), int(harm.sum()), f3(auc(-D[harm], -D[safe])), ci3(ci(bp)), f3(auc(s[harm], s[safe])),
                     f3(float(np.nanmean(sh_t))), f3(float(np.nanmean(sh_d))), f3(wa), ci3(ci(bw)), pairs,
                     ", ".join(f"{t} {a:.2f}" for t, a in sorted(per.items())) or "none"))
    return rows


def suite_model(tag, name, fname, kept_offset=0):
    """B3, B5 and B6: one model on the RULER 4K suite, with A1's and A2's metrics. The z-gate
    standardizes D over all of the cell's inputs (ddof 0), as A2-T3 and the fit did.
    Bootstraps resample inputs within each task."""
    path = os.path.join(OUT, fname)
    if not os.path.exists(path):
        print(f"- {fname}: no log yet")
        return None
    c = V.Cell(path, name)
    if kept_offset:
        c.kept = np.array([[c.by[k][b]["n_kept_plain"] / (c.by[k][1.0]["T"] - kept_offset) for b in c.budgets] for k in c.keys], float)
    D, F, task = c.drop, c.full, c.task
    mk = task == MK3
    Pm = c.plain.astype(float).mean(1)
    print(f"{name}: {c.n} inputs (" + ", ".join(f"{t} {int((task == t).sum())}" for t in np.unique(task)) + f"), budgets {', '.join(V.bname(b) for b in c.budgets)}, "
          f"inputs dropped for missing budgets {c.n_dropped}, unreadable lines {c.bad}.")
    print(f"AUC of D for NIAH-MK3 (MK3 low) against the other three tasks: {auc(D[~mk], D[mk]):.3f}. Mean D by task: "
          + ", ".join(f"{t} {D[task == t].mean():+.3f}" for t in np.unique(task)) + ".")
    tz = D.mean() + THETA * D.std()
    groups = c.task_groups()
    rows = []
    for gate, tau in (("tau=0.07", TAU), (f"z-gate ({tz:.3f})", tz)):
        op = c.open(tau)
        G = c.gated(tau).astype(float).mean(1)
        po = op.mean()
        page, rnd = G.mean() - Pm.mean(), (1 - po) * (F.mean() - Pm.mean())
        sel = []
        for _ in range(REPS):
            i = strat_idx(rng, groups)
            o = op[i].mean()
            sel.append((G[i].mean() - Pm[i].mean()) - (1 - o) * (F[i].mean() - Pm[i].mean()))
        kg = np.where(op[:, None], c.kept, 1.0).mean()
        rows.append((gate, f3(po), f3(float((~op[mk]).mean())), f3(float((~op[~mk]).mean())), f3(kg), f3(F.mean()), f3(Pm.mean()), f3(G.mean()),
                     s3(page), s3(rnd), s3(page - rnd), ci3(ci(sel), True)))
        # Kept for B5 and B6 summaries; printing and the generator's draws are unchanged.
        c.summary = getattr(c, "summary", {})
        c.summary["tau" if gate.startswith("tau") else "z"] = dict(thr=tau, po=float(po), mk_closed=float((~op[mk]).mean()),
                                                                   other_closed=float((~op[~mk]).mean()), sel=float(page - rnd), ci=ci(sel))
    c.auc_mk3 = float(auc(D[~mk], D[mk]))
    table(f"{tag}-T1", f"{name}: gate gain over plain SnapKV, averaged over the budget grid",
          ["gate", "open", "MK3 closed", "other tasks closed", "mean kept", "full", "plain", "gated", "PAGE gain", "random gain", "selection", "95% CI"], rows,
          "Random gain = (1 - open) x (full - plain): a gate that opens at random at the same rate. Selection = PAGE gain - random gain. "
          f"Bootstrap over inputs within task, {REPS} replicates. The z-gate threshold is mean D + ({THETA}) SD over all {c.n} inputs.")
    rows = []
    for t in np.unique(task):
        m = task == t
        rows.append((t, int(m.sum()), s3(float(D[m].mean())), f3(float(F[m].mean())))
                    + tuple(f3(float(c.plain[m, j].mean())) for j in range(len(c.budgets))) + (f3(float(c.gated(tz)[m].mean())),))
    table(f"{tag}-T2", f"{name}: accuracy by task and budget, plain SnapKV", ["task", "N", "mean D", "full"] + [V.bname(b) for b in c.budgets] + ["z-gate, grid mean"], rows)
    table(f"{tag}-T3", f"{name}: D against per-input harm, with A1-T1's comparators",
          ["model", "b", "full correct", "harmed", "pooled AUC(D)", "95% CI", "task id (in-sample)", "task id (split-half)", "D (same splits)",
           "within-task AUC", "95% CI", "same-task pairs", "per task (at least 5 in each class)"], harm_rows_a1(name, c),
          "Harm = correct at full cache and wrong at b. AUC of -D, so above 0.5 is the direction the gate assumes.")
    return c


def b3():
    global rng
    rng = np.random.default_rng(20261108 + 3)
    print("\n## B3. Newer model, Qwen3-14B with thinking off, RULER 4K suite (G-W4, G-Q2)\n")
    suite_model("B3", "Qwen3-14B", "b3_qwen3_14b.jsonl")


def b2c():
    """B2c, preregistration/revision/b2c_quality_qwen14b_prereg.md: QuALITY on Qwen2.5-14B, both arms."""
    global rng
    rng = np.random.default_rng(20261108 + 5)
    print("\n## B2c. QuALITY on Qwen2.5-14B, letter and quote-then-answer arms (replication test)\n")
    if not all(complete(os.path.join(OUT, stem + ".jsonl"), 1500) for _, _, stem in B2C_FILES):
        return
    rows, off, harm_rows, chk, chk_off = b2_cells(B2C_FILES)
    if not rows:
        return
    table("B2c-T1", "Qwen2.5-14B on QuALITY: accuracy and selection against a random gate at the same open rate, SnapKV", B2_T1_COLS, rows,
          "As B2-T1: bootstraps resample documents, 2000 replicates; the z-gate threshold is mean D + (-0.69) SD over the arm's inputs.")
    table("B2c-T3", "Qwen2.5-14B on QuALITY: harm at b = 1/16, by arm", B2_T3_COLS, harm_rows)
    q = [x for x in chk if x["arm"] == "B2c quote"]
    p1 = [x for x in q if x["gate"] == "z-gate" and x["b"] == 0.125]
    hr = {r[0]: float(r[6]) for r in harm_rows}
    out = []
    if p1:
        x = p1[0]
        rep = x["lo"] > 0
        out.append(("P1", "quote arm, z-gate, b = 1/8: selection positive with 95% CI above 0 (the Mistral result replicates)",
                    f"CI [{x['lo']:+.3f}, {x['hi']:+.3f}]", "replicated, so P1 (no replication) fails" if rep else "not replicated, so P1 holds"))
    if "B2c quote" in hr and "B2c letter" in hr:
        out.append(("P2", "harm at 1/16 higher in the quote arm than in the letter arm", f"{hr['B2c quote']:.3f} vs {hr['B2c letter']:.3f}",
                    "holds" if hr["B2c quote"] > hr["B2c letter"] else "fails"))
    rest = [x for x in q if not (x["gate"] == "z-gate" and x["b"] == 0.125)]
    if rest:
        k = sum(x["lo"] <= 0 <= x["hi"] for x in rest)
        out.append(("P3a", "quote arm: selection CI includes 0 in the other gate and budget cells", f"{k}/{len(rest)}", "holds" if k == len(rest) else "fails"))
        aucs = [x["aD"] for x in q if x["gate"] == "tau=0.07" and not np.isnan(x["aD"])]
        out.append(("P3b", "quote arm: AUC(D) against harm below 0.65, at 1/16 and 1/8", ", ".join(f"{a:.3f}" for a in aucs),
                    "holds" if aucs and all(a < 0.65 for a in aucs) else "fails"))
    t = [x for x in chk if x["gate"] == "tau=0.07" and x["b"] == 0.0625]
    if t:
        out.append(("P4", "fixed tau opens on at most 12% of inputs, both arms", ", ".join(f"{x['arm']} {x['po']:.3f}" for x in t),
                    "holds" if all(x["po"] <= 0.12 for x in t) else "fails"))
    table("B2c-T4", "B2c's pre-registered predictions", ["prediction", "test", "value", "verdict"], out)


B5_MODELS = [  # (tag, name, design, log in experiments/results/revision)
    ("B5a", "Olmo-Hybrid-7B", "linear + full attention", "b5_olmo_hybrid_7b.jsonl"),
    ("B5b", "Qwen3-Next-80B-A3B", "linear + full attention", "b5_qwen3_next_80b.jsonl"),
    ("B5c", "LFM2-24B-A2B", "short convolution + attention", "b5_lfm2_24b_a2b.jsonl"),
]
B6_MODELS = [("B6a", "DeepSeek-V2-Lite-Chat", "MLA", "b6_deepseek_v2_lite.jsonl"), ("B6b", "GLM-4.7-Flash", "MLA", "b6_glm47_flash.jsonl")]


def sel_by_budget(c, tau):
    """Selection at each budget, bootstrapped within task: (PAGE - plain) - (1 - open)(full - plain)."""
    op, F, groups = c.open(tau), c.full.astype(float), c.task_groups()
    out = {}
    for b in c.budgets:
        P = c.plain[:, c.bi(b)].astype(float)
        G = np.where(op, P, F)

        def stat(i):
            return (G[i].mean() - P[i].mean()) - (1 - op[i].mean()) * (F[i].mean() - P[i].mean())

        out[b] = (stat(np.arange(c.n)), ci([stat(strat_idx(rng, groups)) for _ in range(REPS)]))
    return out


def arch_section(models, label):
    """B5 and B6: each model as B3 (tables {tag}-T1 to T3, then T4 by budget), then one summary row per model."""
    summ = []
    for tag, name, design, fname in models:
        if not complete(os.path.join(OUT, fname), 1000):  # 4 tasks x 50 inputs x 5 budgets
            continue
        print(f"\n### {tag}. {name} ({design})\n")
        c = suite_model(tag, name, fname, kept_offset=1)
        sb = sel_by_budget(c, c.summary["z"]["thr"])
        sbt = sel_by_budget(c, TAU)
        table(f"{tag}-T4", f"{name}: selection against a random gate at each budget", ["b", "z-gate selection", "95% CI", "tau=0.07 selection", "95% CI"],
              [(V.bname(b), s3(sb[b][0]), ci3(sb[b][1], True), s3(sbt[b][0]), ci3(sbt[b][1], True)) for b in c.budgets],
              f"Bootstrap over inputs within task, {REPS} replicates.")
        fa = float(c.full.mean())
        keep = {b: float(c.plain[:, c.bi(b)].mean()) / fa if fa > 0 else float("nan") for b in c.budgets}
        accs = ", ".join(f"{t} {c.full[c.task == t].mean():.2f}" + (" (below 0.2: harm numbers rest on few inputs)" if c.full[c.task == t].mean() < 0.2 else "")
                         for t in np.unique(c.task))
        summ.append(dict(tag=tag, name=name, design=design, n=c.n, auc=c.auc_mk3, tau=c.summary["tau"], z=c.summary["z"], sel16=sb[0.0625],
                         keep=keep, accs=accs))
    if summ:
        table(f"{label}-T0", f"{label} summary: D, gates, selection at b = 1/16, and accuracy kept by plain SnapKV",
              ["model", "design", "N", "full accuracy by task", "AUC(D) for MK3", "tau open", "tau MK3 closed", "z open", "z MK3 closed",
               "z selection at 1/16", "95% CI", "kept 1/2", "kept 1/4", "kept 1/8", "kept 1/16"],
              [(s["name"], s["design"], s["n"], s["accs"], f3(s["auc"]), f3(s["tau"]["po"]), f3(s["tau"]["mk_closed"]), f3(s["z"]["po"]),
                f3(s["z"]["mk_closed"]), s3(s["sel16"][0]), ci3(s["sel16"][1], True)) + tuple(f3(s["keep"].get(b, float("nan"))) for b in (0.5, 0.25, 0.125, 0.0625))
               for s in summ],
              "Kept = plain SnapKV accuracy at b over full-cache accuracy, all four tasks. Eviction applies only to layers with a key-value cache, "
              "so for hybrids nominal budgets are not comparable with full-attention models. Kept fractions use T - 1 positions (last prompt token held back).")
    return summ


def b5():
    global rng
    rng = np.random.default_rng(20261108 + 6)
    print("\n## B5. Hybrid models, D and eviction on the layers with a key-value cache (G-Q3, G-Q2)\n")
    return arch_section(B5_MODELS, "B5")


def b6():
    global rng
    rng = np.random.default_rng(20261108 + 7)
    print("\n## B6. MLA models (G-Q3)\n")
    return arch_section(B6_MODELS, "B6")


if __name__ == "__main__":
    which = sys.argv[1:] or ["b1", "b2", "b2c", "b3", "b5", "b6"]
    for section in ("b1", "b2", "b3", "b2c", "b5", "b6"):
        if section in which:
            globals()[section]()
