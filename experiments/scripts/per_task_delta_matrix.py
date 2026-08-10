"""P0-2: per-task decomposition of the headline 4x4 gating matrix.

How much of the grand-mean Delta in tab:matrix comes from NIAH-MK3 alone?

Two conventions are pinned and must not drift:

  1. Budget: Delta averages over eviction budgets b < 1.0. This is what
     method_agnostic_matrix.py uses and what the +22.9pp caption reports.
     Alternatives are printed at the bottom so the choice stays auditable.
  2. Threshold: every cell is evaluated at tau = 0.07 through the post-hoc
     identity (see gatelib), so the Qwen2.5-3B SnapKV cell executed at
     tau = 0.04 is comparable with the rest.
"""
from paths import MODELS, POLICIES, MK3, cell_path, out_path
from gatelib import eviction_rows, delta, rate

import os


def main():
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("# Per-task decomposition of the 4x4 gating matrix (RULER 4K, tau=0.07)")
    emit()

    cells, tasks_seen = [], []
    for policy in POLICIES:
        for model_name, slug in MODELS:
            path = cell_path(slug, policy)
            if not os.path.exists(path):
                continue
            rows = eviction_rows(path, policy, slug)
            by_task = {}
            for r in rows:
                by_task.setdefault(r["task"], []).append(r)
            for t in by_task:
                if t not in tasks_seen:
                    tasks_seen.append(t)
            cells.append({
                "name": f"{policy} {model_name}",
                "per_task": {t: delta(v) for t, v in by_task.items()},
                "all": delta(rows),
                "nomk3": delta(rows, lambda t: t != MK3),
                "open_nomk3": rate([r["open"] for r in rows if r["task"] != MK3]),
                "open_mk3": rate([r["open"] for r in rows if r["task"] == MK3]),
            })

    order = [MK3] + [t for t in tasks_seen if t != MK3]
    emit(f"## Convention: b < 1.0, tau = 0.07  ({len(cells)} cells)")
    emit()
    emit("| cell | " + " | ".join(order)
         + " | all | no-MK3 | gate-open non-MK3 | gate-open MK3 |")
    emit("|---" * (len(order) + 5) + "|")

    zero = 0
    for c in cells:
        vals = [f"{c['per_task'][t]:+.3f}" if t in c["per_task"] else "--" for t in order]
        flag = ""
        if abs(c["nomk3"]) < 1e-12:
            zero += 1
            flag = " **"
        emit(f"| {c['name']} | " + " | ".join(vals)
             + f" | {c['all']:+.3f} | {c['nomk3']:+.3f}{flag}"
             f" | {c['open_nomk3']:.3f} | {c['open_mk3']:.3f} |")

    gm_all = sum(c["all"] for c in cells) / len(cells)
    gm_nomk3 = sum(c["nomk3"] for c in cells) / len(cells)
    emit()
    emit(f"- grand mean Delta, all four tasks: **{gm_all:+.4f}**")
    emit(f"- grand mean Delta, excluding NIAH-MK3: **{gm_nomk3:+.4f}**")
    emit(f"- cells with non-MK3 Delta exactly 0.000: **{zero}/{len(cells)}**")
    emit(f"- in those cells the gate opens on "
         f"{min(c['open_nomk3'] for c in cells if abs(c['nomk3']) < 1e-12):.0%}"
         f" of non-MK3 inputs, so gated == plain by construction")
    emit(f"- mean gate-open fraction: non-MK3 "
         f"{sum(c['open_nomk3'] for c in cells)/len(cells):.3f}, "
         f"MK3 {sum(c['open_mk3'] for c in cells)/len(cells):.3f}")
    emit()
    emit(f"Arithmetically the matrix is Delta ~ (1/4) * Delta_MK3: mean "
         f"Delta_MK3/4 = {sum(c['per_task'][MK3] for c in cells)/len(cells)/4:+.4f} "
         f"vs measured grand mean {gm_all:+.4f}.")

    emit()
    emit("## Budget-convention sensitivity (grand mean, all tasks)")
    emit()
    for name, filt in [
        ("b < 1.0 (PINNED)", lambda b: b < 1.0),
        ("all budgets incl. b = 1.0", lambda b: True),
        ("b in {0.125, 0.25, 0.5}", lambda b: b in (0.125, 0.25, 0.5)),
        ("4 shared budgets {0.0625, 0.125, 0.25, 0.5}",
         lambda b: b in (0.0625, 0.125, 0.25, 0.5)),
    ]:
        vals = []
        for policy in POLICIES:
            for _, slug in MODELS:
                path = cell_path(slug, policy)
                if not os.path.exists(path):
                    continue
                vals.append(delta(eviction_rows(path, policy, slug, budget_filter=filt)))
        emit(f"- {name}: {sum(vals)/len(vals):+.4f}  ({len(vals)} cells)")

    with open(out_path("per_task_delta_matrix.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    ok = True
    if abs(gm_all - 0.229) > 1e-3:
        print(f"\nFAIL: grand mean {gm_all:.4f} does not match the +22.9pp caption")
        ok = False
    if abs(gm_nomk3 - 0.045) > 1.5e-3:
        print(f"FAIL: no-MK3 grand mean {gm_nomk3:.4f} != 0.045")
        ok = False
    if zero != 8 or len(cells) != 16:
        print(f"FAIL: expected 8 exactly-zero cells of 16, got {zero}/{len(cells)}")
        ok = False
    print("\nCHECK: " + ("PASS" if ok else "FAIL"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()