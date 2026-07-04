"""Pareto / achieved-compression / oracle-gap analysis for gated KV eviction.

Post-hoc gating convention (used EVERYWHERE, incl. Qwen3B whose stored run
used tau=0.04): at threshold tau, for input i at budget b < 1:
    gated_correct = correct_plain(i, b)  if drop_i >= tau else full_correct(i)
    gated_kept    = n_kept_plain(i,b)/T_i if drop_i >= tau else 1.0
Full-KV outcome: correct_plain at budget 1.0 when available; for Yi (no
b=1.0 rows) recovered from correct_gated of stored-gate-closed rows.

Outputs:
  paper/figs/pareto_gated.pdf/.png
  (tables printed to stdout; prose report: experiments/results/pareto_oracle_analysis.md)

Run: .venv/bin/python experiments/scripts/pareto_oracle_analysis.py
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- style matched to paper/make_figures.py ----
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "lines.linewidth": 1.4,
    "patch.linewidth": 0.6,
    "legend.frameon": False,
    "savefig.bbox": "tight",
})

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(exist_ok=True)

TAU = 0.07
MATRIX_TASKS = {"niah_multikey_3", "vt", "fwe", "qa_1"}

MODELS = [
    ("gated_4k_qwen15b", "Qwen2.5-1.5B"),
    ("gated_4k_qwen3b", "Qwen2.5-3B"),
    ("gated_4k_qwen14b", "Qwen2.5-14B"),
    ("gated_4k_mistral7b", "Mistral-7B"),
]
LLAMA_FAM = [
    ("gated_4k_yi15_9b", "Yi-1.5-9B"),
    ("gated_4k_llama31_8b", "Llama-3.1-8B"),
]


def load(stem):
    return [json.loads(l) for l in open(RESULTS / f"{stem}.jsonl") if l.strip()]


def index(recs, tasks=None):
    """Return per-input dict: key -> {budget: rec}, plus full-KV outcome map."""
    by_id = defaultdict(dict)
    for r in recs:
        if tasks and r["task"] not in tasks:
            continue
        by_id[(r["id"], r["task"])][r["budget"]] = r
    full = {}
    for k, bmap in by_id.items():
        if 1.0 in bmap:
            full[k] = bool(bmap[1.0]["correct_plain"])
        else:
            # recover from a stored-gate-closed row (n_kept_gated == T)
            for r in bmap.values():
                if not r["gate_open"]:
                    assert r["n_kept_gated"] == r["T"]
                    full[k] = bool(r["correct_gated"])
                    break
    return by_id, full


def analyze(stem, tasks=None, tau=TAU):
    by_id, full = index(load(stem), tasks)
    budgets = sorted({b for bm in by_id.values() for b in bm})
    sub = [b for b in budgets if b < 1.0]
    out = {"budgets": sub, "n_inputs": len(by_id), "has_full": len(full)}

    # per-budget aggregates
    plain_pts, gated_pts, rows = [], [], []
    ora_acc_all, gat_acc_all, pla_acc_all = [], [], []
    for b in sub:
        pa, pk, ga, gk, oa, unknown = [], [], [], [], [], 0
        for k, bm in by_id.items():
            r = bm.get(b)
            if r is None:
                continue
            cp = bool(r["correct_plain"])
            kf = r["n_kept_plain"] / r["T"]
            pa.append(cp)
            pk.append(kf)
            open_ = r["drop"] >= tau
            fk = full.get(k)
            if open_:
                ga.append(cp)
                gk.append(kf)
            else:
                if fk is None:
                    unknown += 1
                    ga.append(cp)  # will not happen for tau=0.07 (stored gate matches)
                else:
                    ga.append(fk)
                gk.append(1.0)
            # oracle: better of plain-evict vs full-KV
            if fk is None:
                oa.append(cp)  # lower bound
                unknown += 1
            else:
                oa.append(max(cp, fk))
        rows.append(dict(budget=b, plain_acc=np.mean(pa), plain_kept=np.mean(pk),
                         gated_acc=np.mean(ga), gated_kept=np.mean(gk),
                         oracle_acc=np.mean(oa), n=len(pa), unknown=unknown))
        plain_pts.append((np.mean(pk), np.mean(pa)))
        gated_pts.append((np.mean(gk), np.mean(ga)))
        pla_acc_all += pa; gat_acc_all += ga; ora_acc_all += oa

    # full-KV point
    afull = np.mean([v for v in full.values()]) if full else float("nan")
    out["A_full"] = afull
    out["rows"] = rows
    out["plain_pts"] = plain_pts + [(1.0, afull)]
    out["gated_pts"] = gated_pts + [(1.0, afull)]
    out["open_frac"] = np.mean([next(iter(bm.values()))["drop"] >= tau
                                for bm in by_id.values()])
    P, G, O = np.mean(pla_acc_all), np.mean(gat_acc_all), np.mean(ora_acc_all)
    out["pooled"] = dict(plain=P, gated=G, oracle=O,
                         delta_gated=G - P, delta_oracle=O - P,
                         recovery=(G - P) / (O - P) if O > P else float("nan"))
    out["pooled_kept_gated"] = np.mean([r["gated_kept"] for r in rows])
    out["pooled_kept_plain"] = np.mean([r["plain_kept"] for r in rows])
    return out, by_id, full


def per_task_oracle(stem, tau=TAU):
    """Per-task pooled (over budgets<1) plain/gated/oracle."""
    recs = load(stem)
    tasks = sorted({r["task"] for r in recs})
    res = {}
    for t in tasks:
        out, _, _ = analyze(stem, tasks={t}, tau=tau)
        res[t] = out
    return res


def sanity_check_posthoc(stem, tau=TAU):
    """For runs stored at tau=0.07, reconstruction must equal stored gated cols."""
    by_id, full = index(load(stem))
    mism = 0
    tot = 0
    for k, bm in by_id.items():
        for b, r in bm.items():
            if b == 1.0:
                continue
            tot += 1
            open_ = r["drop"] >= tau
            rec_c = bool(r["correct_plain"]) if open_ else full.get(k)
            rec_k = r["n_kept_plain"] if open_ else r["T"]
            if rec_c is not None and (rec_c != bool(r["correct_gated"]) or rec_k != r["n_kept_gated"]):
                mism += 1
    return mism, tot


def frontier_verdict(plain_pts, gated_pts):
    """For each gated point, plain accuracy at the same kept-KV via linear interp
    on the plain curve (kept ascending); and vice versa."""
    pp = sorted(plain_pts)
    px = np.array([p[0] for p in pp]); py = np.array([p[1] for p in pp])
    gp = sorted(gated_pts)
    res = []
    for x, a in gp:
        pa = float(np.interp(x, px, py))
        res.append((x, a, pa, a - pa))
    return res


def main():
    print("=" * 70)
    print("SANITY: post-hoc reconstruction vs stored gated columns (tau=0.07 runs)")
    for stem in ["gated_4k_qwen15b", "gated_4k_qwen14b", "gated_4k_mistral7b"]:
        m, t = sanity_check_posthoc(stem)
        print(f"  {stem}: {m}/{t} mismatches")
    m, t = sanity_check_posthoc("gated_4k_qwen3b", tau=0.04)
    print(f"  gated_4k_qwen3b @ stored tau=0.04: {m}/{t} mismatches")

    results = {}
    print("\n" + "=" * 70)
    print("MAIN 4K MATRIX MODELS (SnapKV, tau=0.07 post-hoc)")
    for stem, name in MODELS:
        out, _, _ = analyze(stem)
        results[name] = out
        p = out["pooled"]
        print(f"\n--- {name} (N={out['n_inputs']}, A_full={out['A_full']:.3f}, "
              f"gate-open frac={out['open_frac']:.3f})")
        print(f"  pooled b<1: plain={p['plain']:.3f} gated={p['gated']:.3f} "
              f"oracle={p['oracle']:.3f}  dG={p['delta_gated']:+.3f} "
              f"dO={p['delta_oracle']:+.3f} recovery={p['recovery']:.3f}")
        print(f"  {'b':>7} {'plain_kept':>10} {'plain_acc':>9} {'gated_kept':>10} "
              f"{'gated_acc':>9} {'oracle_acc':>10}")
        for r in out["rows"]:
            print(f"  {r['budget']:>7} {r['plain_kept']:>10.3f} {r['plain_acc']:>9.3f} "
                  f"{r['gated_kept']:>10.3f} {r['gated_acc']:>9.3f} {r['oracle_acc']:>10.3f}")
        print("  frontier check (gated pt vs plain interp at same kept-KV):")
        for x, a, pa, d in frontier_verdict(out["plain_pts"], out["gated_pts"]):
            print(f"    kept={x:.3f}: gated={a:.3f} plain@same-kept={pa:.3f} adv={d:+.3f}")

    print("\n" + "=" * 70)
    print("PER-TASK ORACLE (4 matrix models, pooled b<1)")
    per_task = {}
    for stem, name in MODELS:
        pt = per_task_oracle(stem)
        per_task[name] = pt
        for t, out in pt.items():
            p = out["pooled"]
            rec = p["recovery"]
            print(f"  {name:14s} {t:16s} plain={p['plain']:.3f} gated={p['gated']:.3f} "
                  f"oracle={p['oracle']:.3f} dG={p['delta_gated']:+.3f} "
                  f"dO={p['delta_oracle']:+.3f} rec={rec if rec==rec else float('nan'):.3f}"
                  if rec == rec else
                  f"  {name:14s} {t:16s} plain={p['plain']:.3f} gated={p['gated']:.3f} "
                  f"oracle={p['oracle']:.3f} dG={p['delta_gated']:+.3f} "
                  f"dO={p['delta_oracle']:+.3f} rec=n/a (oracle==plain)")

    # grand means over the 4 models (pooled level)
    gm_dG = np.mean([results[n]["pooled"]["delta_gated"] for _, n in MODELS])
    gm_dO = np.mean([results[n]["pooled"]["delta_oracle"] for _, n in MODELS])
    gm_rec = np.mean([results[n]["pooled"]["recovery"] for _, n in MODELS])
    print(f"\nGRAND MEAN over 4 models: dG={gm_dG:+.3f} dO={gm_dO:+.3f} "
          f"mean-recovery={gm_rec:.3f} ratio-of-means={gm_dG/gm_dO:.3f}")

    print("\n" + "=" * 70)
    print("LLAMA-FAMILY (SnapKV, tau=0.07 post-hoc; 4-task matrix suite)")
    for stem, name in LLAMA_FAM:
        out, by_id, full = analyze(stem, tasks=MATRIX_TASKS)
        results[name + " (4-task)"] = out
        p = out["pooled"]
        print(f"\n--- {name} 4-task (N={out['n_inputs']}, full-KV known for "
              f"{out['has_full']}/{out['n_inputs']}, A_full(known)={out['A_full']:.3f}, "
              f"open={out['open_frac']:.3f})")
        print(f"  pooled: plain={p['plain']:.3f} gated={p['gated']:.3f} "
              f"oracle>={p['oracle']:.3f} dG={p['delta_gated']:+.3f} "
              f"dO>={p['delta_oracle']:+.3f} recovery<={p['recovery']:.3f}")
        # upper-bound oracle: unknown full -> correct
        # recompute with optimistic assumption
        budgets = out["budgets"]
        oa_hi = []
        pa_all = []
        ga_all = []
        for b in budgets:
            for k, bm in by_id.items():
                r = bm.get(b)
                if r is None:
                    continue
                cp = bool(r["correct_plain"])
                fk = full.get(k)
                oa_hi.append(1.0 if fk is None else max(cp, fk))
                pa_all.append(cp)
                ga_all.append(cp if r["drop"] >= TAU else (fk if fk is not None else cp))
        P, G, Ohi = np.mean(pa_all), np.mean(ga_all), np.mean(oa_hi)
        print(f"  oracle upper bound (unknown full-KV := correct): "
              f"oracle<={Ohi:.3f} dO<={Ohi-P:+.3f} recovery>={(G-P)/(Ohi-P):.3f}")
        for r in out["rows"]:
            print(f"  b={r['budget']:>6} plain={r['plain_acc']:.3f} gated={r['gated_acc']:.3f} "
                  f"oracle_lb={r['oracle_acc']:.3f} kept_g={r['gated_kept']:.3f}")
        # 6-task version
        out6, _, _ = analyze(stem, tasks=None)
        p6 = out6["pooled"]
        print(f"  6-task pooled: plain={p6['plain']:.3f} gated={p6['gated']:.3f} "
              f"oracle_lb={p6['oracle']:.3f} dG={p6['delta_gated']:+.3f} "
              f"rec<={p6['recovery']:.3f} kept={out6['pooled_kept_gated']:.3f}")

    # ---- achieved compression summary ----
    print("\n" + "=" * 70)
    print("ACHIEVED COMPRESSION (gated kept-KV fraction, tau=0.07)")
    for _, name in MODELS:
        out = results[name]
        kept = " ".join(f"{r['budget']}:{r['gated_kept']:.3f}" for r in out["rows"])
        print(f"  {name}: open={out['open_frac']:.3f} pooled_kept={out['pooled_kept_gated']:.3f} | {kept}")

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.8), sharey=True)
    for ax, (_, name) in zip(axes.flat, MODELS):
        out = results[name]
        pp = sorted(out["plain_pts"]); gp = sorted(out["gated_pts"])
        ax.plot([p[0] for p in pp], [p[1] for p in pp], "o-", color="#EE6677",
                markersize=4.5, label="Plain SnapKV")
        ax.plot([p[0] for p in gp], [p[1] for p in gp], "s-", color="#4477AA",
                markersize=4.5, label=r"Gated SnapKV ($\tau{=}0.07$)")
        ax.plot(1.0, out["A_full"], marker="*", color="black", markersize=9,
                zorder=4, linestyle="none", label="Full KV")
        ax.axhline(out["A_full"], color="black", linewidth=0.6, alpha=0.35,
                   linestyle="--")
        ax.set_title(f"{name}  (gate open {out['open_frac']*100:.0f}\\%)"
                     if plt.rcParams.get("text.usetex") else
                     f"{name}  (gate open {out['open_frac']*100:.0f}%)",
                     fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(-0.03, 1.03)
    for ax in axes[1]:
        ax.set_xlabel("Mean kept-KV fraction (achieved)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mixed-suite accuracy")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGS / "pareto_gated.pdf")
    fig.savefig(FIGS / "pareto_gated.png", dpi=150)
    print(f"\nwrote {FIGS / 'pareto_gated.pdf'}")


if __name__ == "__main__":
    main()
