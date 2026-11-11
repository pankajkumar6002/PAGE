"""Architecture-normalized head-agreement-drop predictor: separation + gated re-evaluation.

Reproduces every number in experiments/results/normalized_predictor.md.

Per-input D = mean(head_agreement_per_layer[:L//3]) - mean(head_agreement_per_layer[2L//3:])
(same convention as aggregate_drops_n100.py). Normalization variants tested:
  raw         D itself (paper baseline, fixed tau = 0.07)
  a_taskmax   D / max_task(mean D)              -- task-LABELED, diagnostic only, NOT deployable
  b_z         (D - mu_pooled) / sd_pooled       -- deployable with unlabeled pilot
  c_minmax    (D - min) / (max - min)           -- deployable with pilot
  d_quantile  quantile rank of D in pooled dist -- deployable with pilot
  shape variants (e): relD = D/mean(pl); corr = -Pearson(pl, depth); slope_rel;
  coreD (early bin skips first L/6 layers); lateshare = 1 - a_late/max(binned means)

Usage: .venv/bin/python experiments/scripts/normalized_predictor.py
"""
import json
import collections
import numpy as np

RES = "experiments/results/"
CELLS = [
    ("Qwen1.5B-4K", "drops_qwen15b_4k_n100.jsonl"),
    ("Qwen3B-4K", "drops_qwen3b_4k_n100.jsonl"),
    ("Qwen3B-16K", "drops_qwen3b_16k_n100.jsonl"),
    ("Mistral7B-4K", "drops_mistral7b_4k_n100.jsonl"),
    ("Mistral7B-16K", "drops_mistral7b_16k_n100.jsonl"),  # mk3 block duplicated in file; deduped below
    ("Yi1.5-9B-4K", "drops_yi15_9b_4k.jsonl"),
    ("Llama3.1-8B-4K", "drops_llama31_8b_4k.jsonl"),
]
CAL = [l for l, _ in CELLS[:5]]          # threshold-fitting cells (Qwen + Mistral)
HELD = [l for l, _ in CELLS[5:]]         # held-out Llama-family cells
TASKS = ["qa_1", "qa_2", "vt", "niah_multivalue", "fwe", "niah_multikey_3"]
MATRIX = ["niah_multikey_3", "vt", "fwe", "qa_1"]  # tab:matrix mixed suite
BUDGETS = [0.5, 0.25, 0.125, 0.0625]
THETA_Z = -0.6934  # fit on the 5 CAL cells (relaxed criterion, see sweep below)


def rows(f):
    for line in open(RES + f):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            pass  # two files contain one truncated/blank line


def load(f, dedupe_mk3=False):
    per, mk = [], 0
    for r in rows(f):
        if dedupe_mk3 and r["task"] == "niah_multikey_3":
            mk += 1
            if mk > 100:
                continue
        pl = np.array(r["head_agreement_per_layer"])
        per.append((r["task"], r.get("T"), pl))
    return per


def rawD(pl):
    L = len(pl)
    return float(pl[: L // 3].mean() - pl[2 * L // 3:].mean())


# ---------------- shape variants (e) ----------------
def relD(pl):
    return rawD(pl) / pl.mean()


def corr(pl):
    x = np.arange(len(pl)) / (len(pl) - 1)
    return -np.corrcoef(pl, x)[0, 1]


def slope_rel(pl):
    x = np.arange(len(pl)) / (len(pl) - 1)
    return -np.polyfit(x, pl, 1)[0] / pl.mean()


def coreD(pl):
    L = len(pl)
    return float(pl[L // 6: L // 3].mean() - pl[2 * L // 3:].mean())


def lateshare(pl):
    L = len(pl)
    bins = [pl[i * L // 6:(i + 1) * L // 6].mean() for i in range(6)]
    return 1 - pl[2 * L // 3:].mean() / max(bins)


SHAPE = {"relD": relD, "corr": corr, "slope_rel": slope_rel, "coreD": coreD,
         "lateshare": lateshare}


def normalize(tv, kind):
    ts = [t for t, _ in tv]
    D = np.array([v for _, v in tv])
    if kind == "raw":
        V = D
    elif kind == "a_taskmax":
        tm = {t: np.mean([v for tt, v in tv if tt == t]) for t in set(ts)}
        V = D / max(tm.values())
    elif kind == "b_z":
        V = (D - D.mean()) / D.std()
    elif kind == "c_minmax":
        V = (D - D.min()) / (D.max() - D.min())
    elif kind == "d_quantile":
        s = np.sort(D)
        V = np.array([(np.searchsorted(s, v, "left") + np.searchsorted(s, v, "right"))
                      / 2 / len(D) for v in D])
    return list(zip(ts, V))


def auc(neg, pos):
    neg = np.sort(np.asarray(neg))
    pos = np.asarray(pos)
    return sum((neg < v).sum() + 0.5 * (neg == v).sum() for v in pos) / (len(neg) * len(pos))


def main():
    data = {lbl: load(f, dedupe_mk3="mistral7b_16k" in f) for lbl, f in CELLS}
    dvals = {lbl: [(t, rawD(pl)) for t, _, pl in data[lbl]] for lbl in data}

    # ---- (e) shape-statistic per-cell AUC (MK3 vs rest) ----
    print("=== shape variants: per-cell input-level AUC (MK3 low vs rest high) ===")
    print("stat".ljust(11) + "".join(l.ljust(16) for l, _ in CELLS))
    for name, fn in [("rawD", rawD)] + list(SHAPE.items()):
        row = name.ljust(11)
        for lbl, _ in CELLS:
            tv = [(t, float(fn(pl))) for t, _, pl in data[lbl]]
            row += ("%.3f" % auc([v for t, v in tv if t == "niah_multikey_3"],
                                 [v for t, v in tv if t != "niah_multikey_3"])).ljust(16)
        print(row)

    # ---- normalized variants a-d ----
    VAR = ["raw", "a_taskmax", "b_z", "c_minmax", "d_quantile"]
    norm = {(lbl, k): normalize(dvals[lbl], k) for lbl in dvals for k in VAR}

    def sweep(k, cells, strict):
        allv = sorted(set(v for lbl in cells for _, v in norm[(lbl, k)]))
        cand = [(a + b) / 2 for a, b in zip(allv, allv[1:])]
        best = None
        for th in cand:
            nok = 0
            for lbl in cells:
                tm = {t: np.mean([v for tt, v in norm[(lbl, k)] if tt == t])
                      for t in TASKS if any(tt == t for tt, _ in norm[(lbl, k)])}
                others = [t for t in tm if t != "niah_multikey_3"]
                if not strict:
                    others = [t for t in others if t != "niah_multivalue"]
                if tm["niah_multikey_3"] < th and all(tm[t] > th for t in others):
                    nok += 1
            excl = ("niah_multikey_3",) if strict else ("niah_multikey_3", "niah_multivalue")
            mk3 = np.array([v for lbl in cells for t, v in norm[(lbl, k)]
                            if t == "niah_multikey_3"])
            rest = np.array([v for lbl in cells for t, v in norm[(lbl, k)] if t not in excl])
            bal = 0.5 * ((mk3 < th).mean() + (rest >= th).mean())
            if best is None or (nok, bal) > best[0]:
                best = ((nok, bal), th)
        return best[1], best[0][0], best[0][1]

    print("\n=== global single-threshold sweep over ALL 7 cells ===")
    for k in VAR:
        s = sweep(k, [l for l, _ in CELLS], True)
        r = sweep(k, [l for l, _ in CELLS], False)
        mk3 = [v for lbl, _ in CELLS for t, v in norm[(lbl, k)] if t == "niah_multikey_3"]
        rest = [v for lbl, _ in CELLS for t, v in norm[(lbl, k)] if t != "niah_multikey_3"]
        rest5 = [v for lbl, _ in CELLS for t, v in norm[(lbl, k)]
                 if t not in ("niah_multikey_3", "niah_multivalue")]
        print(f"{k:11s} strict: th={s[0]:+.4f} cells={s[1]}/7 bal={s[2]:.3f} | "
              f"relaxed(no mv): th={r[0]:+.4f} cells={r[1]}/7 bal={r[2]:.3f} | "
              f"pooled AUC={auc(mk3, rest):.3f} (excl mv {auc(mk3, rest5):.3f})")

    print("\n=== threshold fit on 5 Qwen/Mistral cells (relaxed), applied to held-out cells ===")
    for k in VAR:
        th, nok, bal = sweep(k, CAL, False)
        line = f"{k:11s} theta*={th:+.4f} (calib {nok}/5, bal {bal:.3f})  ||"
        for lbl in HELD:
            tm = {t: np.mean([v for tt, v in norm[(lbl, k)] if tt == t]) for t in TASKS}
            ok = tm["niah_multikey_3"] < th and all(
                tm[t] > th for t in TASKS if t not in ("niah_multikey_3", "niah_multivalue"))
            mk3 = np.array([v for t, v in norm[(lbl, k)] if t == "niah_multikey_3"])
            line += f" {lbl}: task-sep={'OK' if ok else 'FAIL'} open(mk3)={(mk3 >= th).mean():.2f} |"
        print(line)

    # ---- per-cell z summary ----
    print("\n=== per-model pooled stats + where tau=0.07 lands in z units ===")
    for lbl, _ in CELLS:
        D = np.array([v for _, v in dvals[lbl]])
        mu, sd = D.mean(), D.std()
        print(f"{lbl:15s} N={len(D)} mu={mu:+.4f} sd={sd:.4f} z(tau=0.07)={(0.07 - mu) / sd:+.2f} "
              f"raw-equiv of theta_z={THETA_Z}: {mu + THETA_Z * sd:+.4f}")

    # ---- gated re-evaluation on Yi / Llama ----
    for name, df, gf in [("Yi-1.5-9B-4K", "drops_yi15_9b_4k.jsonl", "gated_4k_yi15_9b.jsonl"),
                         ("Llama-3.1-8B-4K", "drops_llama31_8b_4k.jsonl",
                          "gated_4k_llama31_8b.jsonl")]:
        print("\n" + "=" * 70 + f"\nGATED RE-EVAL {name} at theta_z={THETA_Z}")
        D = np.array([v for _, v in dvals[{"Yi-1.5-9B-4K": "Yi1.5-9B-4K",
                                           "Llama-3.1-8B-4K": "Llama3.1-8B-4K"}[name]]])
        z = (D - D.mean()) / D.std()
        g = collections.defaultdict(dict)
        for r in rows(gf):
            g[r["id"]][r["budget"]] = r
        md = max(abs(g[i][0.5]["drop"] - D[i]) for i in g)
        flips = sum((g[i][0.5]["drop"] >= 0.07) != (D[i] >= 0.07) for i in g)
        print(f"stored-vs-recomputed drop: max|diff|={md:.2e}, tau=0.07 flips={flips}/{len(g)}")
        full = {}
        for i in g:
            if 1.0 in g[i]:
                full[i] = g[i][1.0]["correct_plain"]
            else:
                cg = [g[i][b]["correct_gated"] for b in BUDGETS if not g[i][b]["gate_open"]]
                if cg:
                    full[i] = sum(cg) >= len(cg) / 2
        task_of = {i: g[i][0.5]["task"] for i in g}
        new_open = {i: z[i] >= THETA_Z for i in g}
        nclosed_unknown = sum(1 for i in g if i not in full and not new_open[i])
        print(f"full-KV known {len(full)}/{len(g)}; unknown-and-new-closed = {nclosed_unknown} "
              "(must be 0 for exact re-eval)")
        print("task            old_open new_open")
        for t in TASKS:
            ids = [i for i in g if task_of[i] == t]
            print(f"{t:15s} {np.mean([g[i][0.5]['gate_open'] for i in ids]):8.2f} "
                  f"{np.mean([new_open[i] for i in ids]):8.2f}")

        def acc(ids, b, mode):
            out = []
            for i in ids:
                cp = g[i][b]["correct_plain"]
                if mode == "plain":
                    out.append(cp)
                elif mode == "old":
                    out.append(g[i][b]["correct_gated"])
                elif mode == "new":
                    out.append(cp if new_open[i] else full.get(i, cp))
                elif mode == "oracle":
                    out.append(cp or full.get(i, cp))
            return np.mean(out)

        for suite, lab in [(TASKS, "6-task"), (MATRIX, "4-task tab:matrix")]:
            ids = [i for i in g if task_of[i] in suite]
            print(f"--- {lab} (N={len(ids)}) ---")
            print("budget  plain  old-gated  new-gated  oracle | keptKV old -> new")
            dl = collections.defaultdict(list)
            for b in BUDGETS:
                r_ = {m: acc(ids, b, m) for m in ("plain", "old", "new", "oracle")}
                for m in ("old", "new", "oracle"):
                    dl[m].append(r_[m] - r_["plain"])
                ko = np.mean([b if g[i][b]["gate_open"] else 1.0 for i in ids])
                kn = np.mean([b if new_open[i] else 1.0 for i in ids])
                print(f"{b:6} {r_['plain']:6.3f} {r_['old']:9.3f} {r_['new']:10.3f} "
                      f"{r_['oracle']:7.3f} | {ko:.3f} -> {kn:.3f}")
            print("grand-mean delta vs plain: old %+.3f  new %+.3f  oracle %+.3f"
                  % tuple(np.mean(dl[m]) for m in ("old", "new", "oracle")))

        # theta_z operating-point sweep
        print("theta_z sweep: matrixD / keptKV@b=0.0625 / open(mk3)")
        for th in [-1.0, -0.8, THETA_Z, -0.5, -0.3, 0.0]:
            no = {i: z[i] >= th for i in g}
            ids = [i for i in g if task_of[i] in MATRIX]
            dm = np.mean([np.mean([(g[i][b]["correct_plain"] if no[i] else full.get(i, g[i][b]["correct_plain"]))
                                   - g[i][b]["correct_plain"] for i in ids]) for b in BUDGETS])
            kept = np.mean([0.0625 if no[i] else 1.0 for i in g])
            mk3o = np.mean([no[i] for i in g if task_of[i] == "niah_multikey_3"])
            print(f"  th={th:+.3f}: matrixD={dm:+.3f} kept={kept:.3f} open(mk3)={mk3o:.2f}")

        # unlabeled-pilot bootstrap
        rng = np.random.default_rng(0)
        mk3 = [i for i in g if task_of[i] == "niah_multikey_3"]
        rest = [i for i in g if task_of[i] != "niah_multikey_3"]
        for n in (40, 100):
            o = []
            for _ in range(500):
                s = rng.choice(len(D), n, replace=False)
                zz = (D - D[s].mean()) / D[s].std()
                o.append((np.mean([zz[i] >= THETA_Z for i in mk3]),
                          np.mean([zz[i] >= THETA_Z for i in rest])))
            o = np.array(o)
            print(f"pilot n={n}: open(mk3)={o[:, 0].mean():.3f}+-{o[:, 0].std():.3f} "
                  f"open(rest)={o[:, 1].mean():.3f}+-{o[:, 1].std():.3f}")


if __name__ == "__main__":
    main()
