#!/usr/bin/env python3
"""Analyze B7 (PAGE on L-Eval) under preregistration/revision/b7_leval_prereg.md. Read-only: prints markdown tables.

Reuses the checked helpers in checks/verify_2026-11-07.py (AUC, CIs, tables), as analyze_b.py does.
A run is analyzed only when its log ends with the runner's "b7 done" line and the row count matches it.

Usage: python3 -I analyze_b7.py [results_dir] [logs_dir]
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
auc, ci, f3, s3, ci3, table, within_auc, bname = V.auc, V.ci, V.f3, V.s3, V.ci3, V.table, V.within_auc, V.bname
TAU, THETA = V.TAU, V.THETA_Z
RES = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[2] / "results" / "revision")
LOGS = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).resolve().parents[2] / "results" / "revision" / "logs")
REPS = 2000
BUDGETS = [0.5, 0.25, 0.125, 0.0625]
PRIMARY_B = [0.0625, 0.125]
MODELS = [("Qwen2.5-3B", "qwen3b"), ("Mistral-7B", "mistral7b"), ("Qwen2.5-14B", "qwen14b")]
EXT_MODELS = [("Qwen3-14B", "qwen3_14b"), ("Llama-3.1-8B", "llama31_8b")]  # Amendment 3, reported separately
ARMS = [("leval", "L-Eval prompts"), ("quote", "quote-then-answer")]
TASK_ORDER = ["tpo", "quality", "coursera", "topic_retrieval_longchat", "gsm100", "sci_fi"]
SHORT = {"topic_retrieval_longchat": "topic_retrieval"}
rng = np.random.default_rng(20261109)


def rows_of(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def complete(stem):
    """True only if the log's last 'b7 done' line accounts for every question and the log has 5 rows per question."""
    path, log = os.path.join(RES, stem + ".jsonl"), os.path.join(LOGS, stem + ".log")
    if not (os.path.exists(path) and os.path.exists(log)):
        print(f"- {stem}: no log yet")
        return False
    done = [ln for ln in open(log) if ln.startswith("b7 done")]
    if not done:
        print(f"- {stem}: running, not analyzed until complete")
        return False
    m = dict(re.findall(r"(\w+)=(\d+)", done[-1]))
    q, new, res, sk, oom = (int(m[k]) for k in ("questions", "new", "resumed", "skipped", "oom"))
    n = sum(1 for _ in open(path))
    if new + res + sk + oom != q or n != 5 * (new + res):
        print(f"- {stem}: counts do not add up (questions {q}, new {new}, resumed {res}, skipped {sk}, oom {oom}, rows {n}); not analyzed")
        return False
    return True


def load(stem):
    by = collections.defaultdict(dict)
    for r in rows_of(os.path.join(RES, stem + ".jsonl")):
        by[(r["task"], r["id"])][float(r["budget"])] = r
    keys = [k for k in sorted(by, key=lambda k: (TASK_ORDER.index(k[0]), k[1])) if sorted(by[k]) == sorted([1.0] + BUDGETS)]
    assert len(keys) == len(by), "a question is missing a budget"
    g = lambda f: [by[k][1.0][f] for k in keys]
    c = dict(keys=keys, task=np.array(g("task")), D=np.array(g("drop"), float), T=np.array(g("T"), float),
             F=np.array(g("correct_plain"), bool), Fs=np.array(g("leval_score"), float),
             P={b: np.array([by[k][b]["correct_plain"] for k in keys], bool) for b in BUDGETS},
             Ps={b: np.array([by[k][b]["leval_score"] for k in keys], float) for b in BUDGETS},
             kept={b: np.array([by[k][b]["n_kept_plain"] / by[k][1.0]["T"] for k in keys], float) for b in BUDGETS},
             pf={b: np.array([by[k][b]["parse_fail"] for k in keys], bool) for b in [1.0] + BUDGETS},
             doc=np.array([by[k][1.0]["doc"] if by[k][1.0]["task"] != "gsm100" else f"q{k[1]}" for k in keys]))
    for k in keys:  # integrity: D, T and the document are per question, not per budget
        assert len({by[k][b]["drop"] for b in by[k]}) == 1 and len({by[k][b]["T"] for b in by[k]}) == 1
    return c


def strata(c, mask=None):
    """Per task, the index arrays of each document's questions (gsm100: each question alone; it has one shared prompt).
    With a mask, indices are positions within the masked subset, so they index arrays already cut by that mask."""
    sel_idx = np.arange(len(c["keys"])) if mask is None else np.flatnonzero(mask)
    task, doc = c["task"][sel_idx], c["doc"][sel_idx]
    out = []
    for t in TASK_ORDER:
        it = np.flatnonzero(task == t)
        if len(it):
            d = collections.defaultdict(list)
            for i in it:
                d[doc[i]].append(i)
            out.append([np.array(v) for v in d.values()])
    return out


def resample(st):
    return np.concatenate([np.concatenate([gs[j] for j in rng.integers(0, len(gs), len(gs))]) for gs in st])


def sel(op, F, P):
    G = np.where(op, P, F)
    po = op.mean()
    return G.mean() - (po * P.mean() + (1 - po) * F.mean())


def sel_ci(st, op, F, P):
    out = []
    for _ in range(REPS):
        i = resample(st)
        out.append(sel(op[i], F[i], P[i]))
    return ci(out)


def zthr(D):
    return D.mean() + THETA * D.std()


def run_stems():
    for models in (MODELS, EXT_MODELS):  # the original runs first, so their seeds are unchanged by Amendment 3
        for arm, _ in ARMS:
            for name, slug in models:
                yield arm, name, f"b7_{arm}_{slug}"


SEED = {(arm, name): 20261109 + i for i, (arm, name, _) in enumerate(run_stems())}  # one generator per run (Amendment 1)


def run_counts(stem):
    m = dict(re.findall(r"(\w+)=(\d+)", [ln for ln in open(os.path.join(LOGS, stem + ".log")) if ln.startswith("b7 done")][-1]))
    return int(m["questions"]), int(m["new"]) + int(m["resumed"]), int(m["skipped"]), int(m["oom"])


def quality_overlap():
    """How much of L-Eval's quality task B2 already used: articles matched on a 200-character passage from the middle of
    each article (punctuation and case removed), questions on their first line."""
    try:
        from datasets import load_dataset
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        norm = lambda x: re.sub(r"[^a-z0-9]+", " ", x.lower()).strip()
        lev = [json.loads(line) for line in open(os.environ.get("PAGE_LEVAL_DIR", str(Path(__file__).resolve().parents[2] / "data" / "revision" / "leval" / "Exam")) + "/quality.jsonl")]
        b2 = load_dataset("emozilla/quality", split="validation").shuffle(seed=0).select(range(300))
        arts = [norm(a) for a in b2["article"]]
        qs = {norm(q)[:120] for q in b2["question"]}
        docs = [norm(d["input"]) for d in lev]
        k = sum(any(d[len(d) // 2: len(d) // 2 + 200] in a for a in arts) for d in docs)
        lq = [norm(i.split("\n")[0])[:120] for d in lev for i in d["instructions"]]
        return (f"L-Eval's quality task overlaps B2's QuALITY sample: {k} of its {len(docs)} articles are among B2's {len(set(arts))} articles, "
                f"and {sum(q in qs for q in lq)} of its {len(lq)} questions are among B2's 300 questions (B7 uses L-Eval's prompt, B2 its own).")
    except Exception as e:  # noqa: BLE001
        return f"Quality overlap with B2 not computed here ({type(e).__name__}: {e})."


def main():
    global rng
    print("\n## B7. PAGE on L-Eval, closed-ended tasks (S-Q2, G-W3, G-Q1)\n")
    loaded, t0 = {}, []
    for arm, name, stem in run_stems():
        if complete(stem):
            loaded[(arm, name)] = load(stem)
            q, done, sk, oom = run_counts(stem)
            t0.append((arm, name, q, done, sk, oom))
    t1, t2, t3, t4, t5 = [], [], [], [], []
    units = []  # for the predictions
    for (arm, name), c in loaded.items():
        rng = np.random.default_rng(SEED[(arm, name)])
        D, F, n = c["D"], c["F"], len(c["keys"])
        Ff = F.astype(float)
        st = strata(c)
        ndoc = sum(len(g) for g in st)
        gates = (("tau=0.07", TAU), (f"z-gate ({zthr(D):.3f})", zthr(D)))
        wa = {}
        for b in sorted(BUDGETS):  # 1/16 first
            P = c["P"][b]
            h, s_ = F & ~P, F & P
            pooled = auc(-D[h], -D[s_])
            w, pairs, per = within_auc(c["task"], D, h, s_)
            wa[b] = (w, pairs, per)
            for gate, thr in gates:
                op = D >= thr
                Pf = P.astype(float)
                G = np.where(op, Pf, Ff)
                po = op.mean()
                rnd = po * Pf.mean() + (1 - po) * Ff.mean()
                lo, hi = sel_ci(st, op, Ff, Pf)
                t3.append((arm, name, bname(b), gate.split()[0], s3(G.mean() - rnd), ci3((lo, hi), True)))
                if b in PRIMARY_B:
                    kept = np.where(op, c["kept"][b], 1.0).mean()
                    t1.append((arm, name, n, ndoc, bname(b), gate, f3(po), f3(kept), f3(Ff.mean()), f3(Pf.mean()), f3(G.mean()), f3(rnd),
                               s3(G.mean() - Pf.mean()), s3(G.mean() - rnd), ci3((lo, hi), True), f"{h.sum()}/{s_.sum()}", f3(pooled), f3(w),
                               f"{c['pf'][b].mean():.3f}"))
                    units.append(dict(arm=arm, model=name, b=b, gate=gate.split()[0], po=po, lo=lo, hi=hi, pmp=float(G.mean() - Pf.mean()),
                                      closed=int((~op).sum()), harmed=int(h.sum())))
        # Per task, pooled gate (the threshold over all of the run's questions), at the primary budgets
        zt = zthr(D)
        means = {t: D[c["task"] == t].mean() for t in TASK_ORDER if (c["task"] == t).any()}
        lowest = min(means, key=means.get)
        for t in TASK_ORDER:
            m = c["task"] == t
            if not m.any():
                continue
            stt = strata(c, m)
            op = D[m] >= zt
            row = [arm, name, SHORT.get(t, t), int(m.sum()), sum(len(g) for g in stt), s3(means[t]), f3(F[m].mean())]
            row += [f3(c["P"][b][m].mean()) for b in sorted(BUDGETS)]
            hm = F[m] & ~c["P"][0.0625][m]
            row += [f"{hm.sum()}/{F[m].sum()}", f"{c['pf'][1.0][m].mean():.3f}", f"{c['pf'][0.0625][m].mean():.3f}", f3(1 - op.mean()), f3((D[m] >= TAU).mean())]
            for b in PRIMARY_B:
                P = c["P"][b][m].astype(float)
                lo, hi = sel_ci(stt, op, F[m].astype(float), P)
                row += [s3(sel(op, F[m].astype(float), P)), ci3((lo, hi), True)]
                units.append(dict(arm=arm, model=name, task=t, b=b, kind="task", lo=lo, hi=hi))
            h, s_ = F[m] & ~c["P"][0.0625][m], F[m] & c["P"][0.0625][m]
            a = auc(-D[m][h], -D[m][s_])
            row.append(f3(a))
            units.append(dict(arm=arm, model=name, task=t, kind="auc", auc=a, nh=int(h.sum()), ns=int(s_.sum())))
            # the B2 convention, for comparison: z-gate standardized within the task
            opt = D[m] >= zthr(D[m])
            row.append(s3(sel(opt, F[m].astype(float), c["P"][0.0625][m].astype(float))))
            t2.append(tuple(row))
        units.append(dict(arm=arm, model=name, kind="lowest", lowest=lowest, topic_closed=float(1 - (D[c["task"] == "topic_retrieval_longchat"] >= zt).mean())
                          if (c["task"] == "topic_retrieval_longchat").any() else float("nan"), tau_open=float((D >= TAU).mean())))
        # L-Eval-style scores (x100): coursera with its 0.25 partial credit, sci_fi loyalty only
        for t in TASK_ORDER:
            m = c["task"] == t
            if m.any():
                zop = D[m] >= zt
                t4.append((arm, name, SHORT.get(t, t), int(m.sum()), f"{100 * c['Fs'][m].mean():.1f}")
                          + tuple(f"{100 * c['Ps'][b][m].mean():.1f}" for b in sorted(BUDGETS))
                          + (f"{100 * np.where(zop, c['Ps'][0.0625][m], c['Fs'][m]).mean():.1f}",))
        t5.append((arm, name, n, ndoc, f3(wa[0.0625][0]), wa[0.0625][1], ", ".join(f"{SHORT.get(k, k)} {v:.2f}" for k, v in wa[0.0625][2].items()) or "none",
                   f3(wa[0.25][0]), wa[0.25][1]))

    if not loaded:
        print("No complete B7 run yet.")
        return
    note = (f"Selection = PAGE minus a random gate at the same open rate. Bootstrap: within each task, documents resampled with replacement "
            f"(gsm100: each question, since all share one prompt), {REPS} replicates. The z-gate threshold is mean D + ({THETA}) SD over all of the "
            "run's questions, pooled across tasks, so it needs no task labels. Correct = L-Eval score 1. Harm = correct at full cache and wrong at b. "
            "AUCs read low D as more harm.")
    table("B7-T0", "Runs", ["arm", "model", "questions", "analyzed", "over the 16,000-token cap", "out of memory"], t0,
          quality_overlap())
    table("B7-T1", "Pooled over tasks: accuracy and selection against a random gate at the same open rate, SnapKV",
          ["arm", "model", "N", "bootstrap units", "b", "gate", "open", "kept", "full", "plain", "PAGE", "random", "PAGE - plain", "selection", "95% CI",
           "harmed/survived", "pooled AUC(D)", "within-task AUC(D)", "parse failures"], t1, note)
    table("B7-T2", "Per task: accuracy by budget, harm, and selection with the pooled z-gate",
          ["arm", "model", "task", "N", "bootstrap units", "mean D", "full", "1/16", "1/8", "1/4", "1/2", "harmed/full correct at 1/16",
           "parse failures at full", "parse failures at 1/16", "z closed", "tau open",
           "selection 1/16", "95% CI", "selection 1/8", "95% CI", "AUC(D) at 1/16", "selection 1/16, z within task (B2 convention)"], t2,
          "z closed uses the pooled threshold of B7-T1. The last column standardizes D within the task instead, as B2 did per dataset; it needs task labels. "
          "Bootstrap units are documents, except gsm100, where each question is one. Parse failures: in the L-Eval arm, an option-task output with no "
          "A to D letter (L-Eval scores it as answer A), a gsm100 output with no number, or a sci_fi output with neither true nor false; in the quote arm, "
          "no readable Answer line, counted wrong.")
    table("B7-T3", "Pooled selection by budget, both gates", ["arm", "model", "b", "gate", "selection", "95% CI"], t3)
    table("B7-T4", "L-Eval-style scores (x100), plain SnapKV by budget, and PAGE (pooled z-gate) at 1/16",
          ["arm", "model", "task", "N", "full", "1/16", "1/8", "1/4", "1/2", "PAGE at 1/16"], t4,
          "Mean L-Eval score: coursera gets 0.25 when the predicted letters are a strict subset of the gold letters; sci_fi is the loyalty half only "
          "(L-Eval also averages in a no-document fact question, which eviction cannot affect).")
    table("B7-T5", "D against per-input harm within a task", ["arm", "model", "N", "bootstrap units", "within-task AUC at 1/16", "same-task pairs",
          "per task (at least 5 in each class)", "within-task AUC at 1/4", "same-task pairs at 1/4"], t5)

    predictions(units, t2, [n for n, _ in MODELS], "B7-T6", "Pre-registered predictions, the three pre-registered models")
    if any(name in {n for n, _ in EXT_MODELS} for _, name in loaded):
        predictions(units, t2, [n for n, _ in EXT_MODELS], "B7-T7", "The same predictions on the two models added by Amendment 3, counted separately")

def predictions(units, t2, models, tid, title):
    """The pre-registered predictions, counted over the units of the given models only."""
    units = [u for u in units if u.get("model") in models]
    t2 = [r for r in t2 if r[1] in models]
    most = lambda k, n: "holds" if n and k > n / 2 else "fails"
    rows = []
    L = [u for u in units if u.get("arm") == "leval" and "gate" in u and u.get("kind") is None]
    for gate in ("z-gate", "tau=0.07"):
        xs = [u for u in L if u["gate"] == gate]
        k = sum(u["lo"] <= 0 <= u["hi"] for u in xs)
        rows.append(("P1" if gate == "z-gate" else "P1, secondary gate", f"L-Eval arm, pooled selection CI includes 0, {gate} (units: model x budget 1/16, 1/8)",
                     f"{k}/{len(xs)}", most(k, len(xs))))
    xs = [u for u in units if u.get("kind") == "auc" and u["arm"] == "leval" and u["nh"] >= 10 and u["ns"] >= 10]
    k = sum(u["auc"] < 0.65 for u in xs)
    rows.append(("P2", "L-Eval arm, AUC(D) against harm at 1/16 below 0.65 (units: task x model, at least 10 harmed and 10 survived)", f"{k}/{len(xs)}", most(k, len(xs))))
    xs = [u for u in units if u.get("kind") == "lowest"]
    k = sum(u["tau_open"] < 0.20 for u in xs)
    rows.append(("P3", "fixed tau opens on under 20% of questions (units: model x arm)", f"{k}/{len(xs)}", most(k, len(xs))))
    xs = [u for u in units if u.get("kind") == "lowest" and u["arm"] == "leval"]
    k = sum(u["lowest"] != "topic_retrieval_longchat" for u in xs)
    rows.append(("P4", "L-Eval arm, topic_retrieval does not have the lowest mean D of the six tasks (units: model) -- low confidence",
                 f"{k}/{len(xs)} ({'; '.join(SHORT.get(u['lowest'], u['lowest']) + ' lowest on ' + u['model'] for u in xs)})", most(k, len(xs))))
    Q = [u for u in units if u.get("arm") == "quote" and u.get("gate") == "z-gate" and u.get("kind") is None]
    k = sum(u["lo"] <= 0 <= u["hi"] for u in Q)
    rows.append(("Q2", "quote arm, pooled z-gate selection CI includes 0 (units: model x budget)", f"{k}/{len(Q)}", most(k, len(Q))))
    table(tid, title, ["prediction", "test", "units meeting it", "verdict"], rows,
          "\"Most\" means more than half of the units. Q1 (harm in the quote arm) is read from B7-T2's harm column: at least 15% of full-correct answers "
          "lost at 1/16 on most (task, model) units. Units missing a run are not counted; the verdict is final only when all runs of these models are complete.")
    q1 = [r for r in t2 if r[0] == "quote"]
    if q1:
        k = sum(int(r[11].split("/")[0]) >= 0.15 * max(1, int(r[11].split("/")[1])) for r in q1)
        print(f"\nQ1 read-out: harm at 1/16 is at least 15% of full-correct answers on {k} of {len(q1)} quote-arm (task, model) units: {most(k, len(q1))}.")



if __name__ == "__main__":
    main()
