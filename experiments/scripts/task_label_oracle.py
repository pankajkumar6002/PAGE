"""DA-6 / P-3: what would a gate that cheats on task identity achieve?

The gate is advertised as a *per-input* predictor, but most of its validation
is per task. The sharp version of the objection: if a gate keyed on
ground-truth task identity does as well as PAGE, then D buys label-freeness
rather than per-input resolution, and the contribution is narrower than the
abstract claims.

This settles it from the existing logs. The oracle closes on every
capacity-bound input and opens on every other one, which is the best any
task-level policy can do, and it needs a label PAGE never sees.

Reading the result:
  gap == 0   PAGE reproduces the oracle exactly. D bought label-freeness only.
  gap > 0    PAGE beats a policy that knows the task, which is only possible
             through within-task per-input variation.
  gap < 0    PAGE is worse, i.e. its per-input errors cost more than its
             per-input wins.
"""
import os

from paths import MODELS, POLICIES, MK3, cell_path, out_path
from gatelib import eviction_rows, rate

EXPECTED_PAGE = 0.229
EXPECTED_ORACLE = 0.201


def main():
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("# Task-label oracle vs PAGE (RULER 4K matrix, tau = 0.07)")
    emit()
    emit("Oracle: close the gate on every NIAH-MK3 input, open on every other. "
         "It is not implementable, since it needs the task label PAGE never "
         "observes; it is the ceiling for any task-level policy.")
    emit()
    emit("| cell | PAGE $\\Delta$ | oracle $\\Delta$ | gap | gate-open on MK3 |")
    emit("|---|---:|---:|---:|---:|")

    page, oracle = [], []
    for policy in POLICIES:
        for model_name, slug in MODELS:
            path = cell_path(slug, policy)
            if not os.path.exists(path):
                continue
            rows = eviction_rows(path, policy, slug)
            plain = rate([r["plain"] for r in rows])
            d_page = rate([r["gated"] for r in rows]) - plain
            # The oracle takes the full-cache outcome on MK3, plain elsewhere.
            d_orc = rate([r["full"] if r["task"] == MK3 else r["plain"]
                          for r in rows]) - plain
            open_mk3 = rate([r["open"] for r in rows if r["task"] == MK3])
            page.append(d_page)
            oracle.append(d_orc)
            emit(f"| {policy} {model_name} | {d_page:+.3f} | {d_orc:+.3f} "
                 f"| {d_page - d_orc:+.3f} | {open_mk3:.3f} |")

    gm_p = sum(page) / len(page)
    gm_o = sum(oracle) / len(oracle)
    gaps = [p - o for p, o in zip(page, oracle)]
    ties = sum(1 for g in gaps if abs(g) < 1e-12)
    wins = sum(1 for g in gaps if g > 1e-12)
    losses = sum(1 for g in gaps if g < -1e-12)

    emit(f"| **grand mean** | **{gm_p:+.3f}** | **{gm_o:+.3f}** "
         f"| **{gm_p - gm_o:+.3f}** | |")
    emit()
    emit(f"- PAGE reproduces the oracle exactly in **{ties} of {len(gaps)}** cells. "
         f"There the gate is a perfect task classifier and D buys label-freeness, "
         f"not resolution.")
    emit(f"- PAGE **beats** the oracle in **{wins}** cells, by up to "
         f"{max(gaps):+.3f}. A task-level policy cannot do this: the gain comes "
         f"from closing on individual dilution-prone inputs the oracle opens.")
    emit(f"- PAGE **loses** in **{losses}** cells, by at most {min(gaps):+.3f}, "
         f"from false positives on MK3.")
    emit()
    emit("Honest summary: on this suite most of the matrix Delta is reproducible "
         "by a task-label oracle, so the headline is largely a task-level "
         "effect. The per-input claim rests on the cells where PAGE exceeds the "
         "oracle, and on the Llama-3.1-8B MK3 cell where mean D sits above tau "
         "yet per-input dispersion still closes the gate on the inputs that "
         "need it.")

    with open(out_path("task_label_oracle.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    ok = (abs(gm_p - EXPECTED_PAGE) < 1e-3
          and abs(gm_o - EXPECTED_ORACLE) < 1e-3)
    if not ok:
        print(f"\nFAIL: got PAGE {gm_p:.4f} / oracle {gm_o:.4f}, "
              f"expected {EXPECTED_PAGE} / {EXPECTED_ORACLE}")
    print("\nCHECK: " + ("PASS" if ok else "FAIL"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
