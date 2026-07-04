"""Matched-ACHIEVED-cache PAGE-vs-plain analysis (Part 1).

For each (model, base-evictor): build accuracy-vs-achieved-kept-KV curves for
  (a) plain eviction sweeping nominal budget b,
  (b) PAGE (gated, tau=0.07 post-hoc) sweeping b; achieved kept = mean_i[ drop_i>=tau ? n_kept_plain/T : 1.0 ].
Interpolate both at fixed achieved-kept targets {0.25,0.35,0.50}, report PAGE-minus-plain,
and find the crossover kept-KV (below which plain dominates).
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

RESULTS = Path("/home/smlab/projects/eff-nn/experiments/results")
TAU = 0.07
TARGETS = [0.25, 0.35, 0.50]

# (model_label, evictor_label, jsonl_stem)
CONFIGS = [
    ("Qwen2.5-1.5B", "SnapKV",       "gated_4k_qwen15b"),
    ("Qwen2.5-1.5B", "H2O",          "gated_h2o_qwen15b_4k"),
    ("Qwen2.5-1.5B", "StreamingLLM", "gated_streamingllm_qwen15b_4k"),
    ("Qwen2.5-1.5B", "PyramidKV",    "gated_pyramidkv_qwen15b_4k"),
    ("Mistral-7B",   "SnapKV",       "gated_4k_mistral7b"),
    ("Mistral-7B",   "H2O",          "gated_h2o_mistral7b_4k"),
    ("Mistral-7B",   "StreamingLLM", "gated_streamingllm_mistral7b_4k"),
    ("Mistral-7B",   "PyramidKV",    "gated_pyramidkv_mistral7b_4k"),
    # SnapKV extra models (context; pareto md covers them)
    ("Qwen2.5-3B",   "SnapKV",       "gated_4k_qwen3b"),
    ("Qwen2.5-14B",  "SnapKV",       "gated_4k_qwen14b"),
]


def load(stem):
    return [json.loads(l) for l in open(RESULTS / f"{stem}.jsonl") if l.strip()]


def build_curves(stem, tau=TAU):
    recs = load(stem)
    by_id = defaultdict(dict)
    for r in recs:
        by_id[(r["id"], r["task"])][r["budget"]] = r
    # full-KV outcome per input
    full = {}
    for k, bm in by_id.items():
        if 1.0 in bm:
            full[k] = bool(bm[1.0]["correct_plain"])
        else:
            for r in bm.values():
                if not r["gate_open"]:
                    full[k] = bool(r["correct_gated"])
                    break
    budgets = sorted({b for bm in by_id.values() for b in bm})
    sub = [b for b in budgets if b < 1.0]
    open_frac = np.mean([next(iter(bm.values()))["drop"] >= tau for bm in by_id.values()])
    A_full = np.mean(list(full.values())) if full else float("nan")

    plain, page = [], []
    mism = 0
    for b in sub:
        pa, pk, ga, gk = [], [], [], []
        for k, bm in by_id.items():
            r = bm.get(b)
            if r is None:
                continue
            cp = bool(r["correct_plain"]); kf = r["n_kept_plain"] / r["T"]
            pa.append(cp); pk.append(kf)
            opened = r["drop"] >= tau
            if opened:
                ga.append(cp); gk.append(kf)
            else:
                ga.append(full.get(k, cp)); gk.append(1.0)
            # sanity vs stored gated cols (only valid if stored tau==0.07)
            rec_c = cp if opened else full.get(k)
            if rec_c is not None and rec_c != bool(r["correct_gated"]):
                mism += 1
        plain.append((float(np.mean(pk)), float(np.mean(pa))))
        page.append((float(np.mean(gk)), float(np.mean(ga))))
    plain.append((1.0, A_full)); page.append((1.0, A_full))
    return dict(plain=sorted(plain), page=sorted(page), open_frac=float(open_frac),
                A_full=float(A_full), n_inputs=len(by_id), budgets=sub, mism=mism,
                page_floor=min(p[0] for p in page))


def interp_acc(curve, x):
    xs = np.array([p[0] for p in curve]); ys = np.array([p[1] for p in curve])
    if x < xs.min() or x > xs.max():
        return None
    return float(np.interp(x, xs, ys))


def is_degenerate(c):
    """True if the base evictor's achieved cache does not vary with nominal
    budget (e.g. StreamingLLM fixed sink+window): <3 distinct kept<1 points."""
    keptvals = {round(k, 4) for k, _ in c["plain"] if k < 0.999}
    return len(keptvals) < 3


def crossover(c):
    """Finest kept-KV x* in the PAGE-reachable overlap where PAGE_acc == plain_acc.
    Below x*: plain dominates. Endpoint x=1 (both curves meet at A_full) excluded."""
    lo = max(c["page_floor"], c["plain"][0][0])
    hi = 0.99  # exclude the shared full-KV endpoint at x=1.0
    if lo >= hi:
        return (lo, "PAGE floor at/above 0.99; no compression room")
    xs = np.linspace(lo, hi, 2001)
    diff = np.array([interp_acc(c["page"], x) - interp_acc(c["plain"], x) for x in xs])
    if np.all(diff >= 0):
        return (lo, "PAGE dominates whole reachable range; plain-only below floor %.3f" % c["page_floor"])
    if np.all(diff <= 0):
        return (hi, "plain dominates whole reachable range")
    sign = np.sign(diff)
    idx = np.where(np.diff(sign) > 0)[0]  # - to + transition
    if len(idx):
        i = idx[0]
        x0, x1, d0, d1 = xs[i], xs[i + 1], diff[i], diff[i + 1]
        xr = x0 - d0 * (x1 - x0) / (d1 - d0)
        return (float(xr), "crossover")
    return (float(xs[np.argmin(np.abs(diff))]), "ambiguous")


def main():
    print("=" * 100)
    print("PART 1: PAGE vs PLAIN at MATCHED ACHIEVED kept-KV (tau=0.07 post-hoc)")
    print("=" * 100)
    rows = []
    for model, evictor, stem in CONFIGS:
        try:
            c = build_curves(stem)
        except FileNotFoundError:
            print(f"  MISSING: {stem}")
            continue
        xr, kind = crossover(c)
        degen = is_degenerate(c)
        if degen:
            kind = "DEGENERATE (budget-invariant base evictor; comparison ill-posed)"
        line = dict(model=model, evictor=evictor, open_frac=c["open_frac"], degen=degen,
                    page_floor=c["page_floor"], A_full=c["A_full"], cross=xr, kind=kind,
                    n=c["n_inputs"], mism=c["mism"])
        for t in TARGETS:
            pl = interp_acc(c["plain"], t); pg = interp_acc(c["page"], t)
            line[f"plain@{t}"] = pl; line[f"page@{t}"] = pg
            line[f"d@{t}"] = (pg - pl) if (pl is not None and pg is not None) else None
        rows.append(line)
        print(f"\n--- {model} / {evictor}  (N={c['n_inputs']}, open={c['open_frac']:.3f}, "
              f"PAGE floor kept={c['page_floor']:.3f}, A_full={c['A_full']:.3f}, sanity_mism={c['mism']})")
        print("  plain curve:", [(round(k,3),round(a,3)) for k,a in c["plain"]])
        print("  PAGE  curve:", [(round(k,3),round(a,3)) for k,a in c["page"]])
        for t in TARGETS:
            pl, pg, d = line[f"plain@{t}"], line[f"page@{t}"], line[f"d@{t}"]
            def f(v): return "  n/a" if v is None else f"{v:.3f}"
            note = ""
            if pg is None and t < c["page_floor"]:
                note = f"  <- PAGE unreachable (floor {c['page_floor']:.3f}); plain-only"
            print(f"   kept={t}: plain={f(pl)} PAGE={f(pg)} PAGE-plain={('n/a' if d is None else f'{d:+.3f}')}{note}")
        print(f"   CROSSOVER kept-KV = {xr:.3f}  ({kind})")

    # markdown table
    print("\n\n===MARKDOWN===")
    print("| model | evictor | gate-open | PAGE floor | Δ@0.25 | Δ@0.35 | Δ@0.50 | crossover kept | verdict |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        def fmt(t):
            d = r[f"d@{t}"]; pg = r[f"page@{t}"]
            if pg is None: return "n/r*"
            return f"{d:+.3f}"
        if r["degen"]:
            v = "degenerate sweep"
        elif "crossover" == r["kind"]:
            v = f"plain wins below {r['cross']:.2f}, PAGE above"
        elif r["kind"].startswith("PAGE dominates"):
            v = f"PAGE wins wherever it operates (>= floor {r['page_floor']:.2f})"
        else:
            v = r["kind"]
        print(f"| {r['model']} | {r['evictor']} | {r['open_frac']:.2f} | {r['page_floor']:.3f} | "
              f"{fmt(0.25)} | {fmt(0.35)} | {fmt(0.50)} | {r['cross']:.3f} | {v} |")
    print("\n* n/r = PAGE cannot reach this cache size (target below its full-KV-fallback floor); plain is the only option there.")


if __name__ == "__main__":
    main()
