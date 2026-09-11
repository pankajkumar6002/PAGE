"""Expected compression as a function of batch size.

The paper states the mechanism ("realized compression is the minimum over the
sequences in a batch, and a single gate-closed sequence erases the benefit for
the entire batch") but never quantifies it. This computes the consequence.

Model: under a static allocator a batch is provisioned for its largest resident
cache, so with per-sequence open probability p_open and kept-fraction k_open
when open,

    E[kept](B) = p_open^B * k_open + (1 - p_open^B) * 1.0

MODELLING SCOPE, which the paper must state: this is STATIC batch
provisioning. Continuous batching with per-sequence paged allocation (vLLM,
PagedAttention) does not behave this way, because it pages per sequence rather
than provisioning the batch for its maximum, so a reader familiar with
PagedAttention could otherwise misread the table as alarmist. The honest claim
is that PAGE's memory benefit is a single-stream / small-batch property under
static provisioning.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paths import MODELS_HF as MODELS, TAU, released_cell, out_path
from gatelib import read_rows

BATCHES = [1, 2, 4, 8, 16, 32, 64]
AGGRESSIVE = 0.0625
TOL = 1e-2


def cell_stats(path):
    """p_open over unique inputs, and kept-fraction at the aggressive budget."""
    rows = read_rows(path)
    opened, kept = {}, []
    for r in rows:
        opened[(r["task"], r["id"])] = r["drop"] >= TAU
        if abs(r["budget"] - AGGRESSIVE) < 1e-9 and r.get("T"):
            nk = r.get("n_kept_plain")
            if nk:
                kept.append(nk / r["T"])
    if not opened or not kept:
        return None
    p_open = sum(opened.values()) / len(opened)
    k_open = sum(kept) / len(kept)
    return p_open, k_open, len(opened)


def expected_kept(p_open, k_open, B):
    p_all_open = p_open ** B
    return p_all_open * k_open + (1.0 - p_all_open) * 1.0


def main():
    cells = []
    for name, slug, _ in MODELS:
        path = released_cell(slug)
        if not os.path.exists(path):
            continue
        st = cell_stats(path)
        if st:
            cells.append((name, slug, *st))
    if not cells:
        raise SystemExit("no cells found; set PAGE_RESULTS")

    lines = [
        "# Expected compression vs batch size",
        "",
        "Under a **static** allocator a batch is provisioned for its largest",
        "resident cache, so one gate-closed sequence erases the benefit for the",
        "whole batch:",
        "",
        "    E[kept](B) = p_open^B * k_open + (1 - p_open^B) * 1.0",
        "",
        f"`p_open` is measured per cell at tau = {TAU}; `k_open` is the realized",
        f"kept-KV fraction at the aggressive budget b = {AGGRESSIVE} (nominal 16x).",
        "",
        "| cell | p_open | k_open | " + " | ".join(f"B={b}" for b in BATCHES) + " |",
        "|---|---:|---:|" + "---:|" * len(BATCHES),
    ]
    monotone_ok = True
    for name, slug, p_open, k_open, n in cells:
        comps = [1.0 / expected_kept(p_open, k_open, B) for B in BATCHES]
        for a, b in zip(comps, comps[1:]):
            if b > a + 1e-9:
                monotone_ok = False
        lines.append(f"| {name} | {p_open:.3f} | {k_open:.3f} | "
                     + " | ".join(f"{c:.2f}x" for c in comps) + " |")

    # E[compression] is nonlinear in p_open, so averaging p_open first and then
    # applying the model is NOT the mean of the per-cell compressions. Average
    # the outputs, not the inputs. (Averaging inputs gave 2.74x at B=1 against a
    # true 2.91x; the two coincide at B=32, which is why a B=32-only check
    # missed it.)
    mean_p = sum(c[2] for c in cells) / len(cells)
    mean_k = sum(c[3] for c in cells) / len(cells)
    mean_comps = [sum(1.0 / expected_kept(p, k, B) for _, _, p, k, _ in cells) / len(cells)
                  for B in BATCHES]
    lines.append(f"| **mean of per-cell compressions** | ({mean_p:.3f}) | ({mean_k:.3f}) | "
                 + " | ".join(f"**{c:.2f}x**" for c in mean_comps) + " |")
    lines.append("")
    lines.append("The p_open and k_open shown on the mean row are averages of the "
                 "inputs, given for reference only; the compressions are the mean "
                 "of the per-cell compressions, since the model is nonlinear in "
                 "p_open.")

    b32 = mean_comps[BATCHES.index(32)]
    lines += [
        "",
        "## What this settles",
        "",
        f"At the measured mean p_open = {mean_p:.2f}, the probability that every",
        f"sequence in a batch of 32 opens is {mean_p**32:.1e}, so expected",
        f"compression is {b32:.2f}x. **For a static batch-serving system the",
        "memory benefit is close to zero.**",
        "",
        "The paper should state this scope in the abstract rather than leaving",
        "the mechanism unquantified in Section 5. PAGE is a single-stream or",
        "small-batch safeguard under static provisioning.",
        "",
        "## Scope of the model",
        "",
        "This is a static max-over-batch allocator. **Continuous batching** with",
        "per-sequence paged allocation (PagedAttention/vLLM) does not have this",
        "property: it pages per sequence, so a gate-closed sequence costs its own",
        "pages rather than the batch's. The table above is therefore an upper",
        "bound on the damage, and the paper must say which regime it claims.",
        "The cost that does NOT go away under paging is the extra",
        "attention-materializing prefill pass, which is paid on 100% of requests",
        "to save cache on the fraction where the gate closes.",
    ]

    ok = True
    checks = [("compression monotonically decreasing in B", monotone_ok,
               "ok" if monotone_ok else "violated")]
    ok &= monotone_ok
    near1 = abs(b32 - 1.0) <= TOL
    checks.append(("mean compression at B=32 == 1.00", near1, f"{b32:.4f}"))
    ok &= near1
    for name, slug, p_open, k_open, n in cells:
        c1 = 1.0 / expected_kept(p_open, k_open, 1)
        good = c1 > 1.0
        checks.append((f"{name}: B=1 compression > 1x", good, f"{c1:.2f}x (N={n})"))
        ok &= good

    lines += ["", "## Checks", ""]
    for name, passed, got in checks:
        lines.append(f"- [{'PASS' if passed else 'FAIL'}] {name} (got {got})")

    path = out_path("batching_decay.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {path}")
    print("CHECK: PASS" if ok else "CHECK: FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()