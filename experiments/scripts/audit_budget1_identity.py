"""P0-1 justification: show that the recovery indicator r(x) is an identity.

App. `app:recovery-hist` defines

    r(x) = sign( max_b A_gated(b, x) - A_plain(1, x) ),  b over the full sweep,

and reports that the r = -1 bin is empty as evidence that gating never makes an
input strictly worse. The sweep includes b = 1.0. If at b = 1.0 the gated arm
keeps the whole cache and reproduces the plain full-cache outcome, then
max_b A_gated(b, x) >= A_gated(1.0, x) = A_plain(1, x), so r(x) >= 0 holds for
every input by construction and the empty bin carries no information about harm.

This script asserts that precondition on every matrix cell:

    at b = 1.0:  n_kept_plain == n_kept_gated == T  and
                 correct_gated == correct_plain     for every row.

Exit code is non-zero if any cell violates it, in which case the identity
argument is cell-specific and the paper's wording must weaken accordingly.
"""
import os

from paths import MODELS, POLICIES, RESULTS, cell_path, out_path
from gatelib import read_rows

EXTRA = [  # non-matrix SnapKV cells that also carry reported results
    "gated_16k_qwen15b_sdpa.jsonl",
    "gated_16k_qwen3b.jsonl",
    "gated_16k_qwen14b.jsonl",
    "gated_16k_mistral7b.jsonl",
]


def audit(path, name):
    rows = [r for r in read_rows(path) if r["budget"] == 1.0]
    if not rows:
        return name, 0, None, None, None
    kept_full = sum(1 for r in rows if r["n_kept_plain"] == r["T"]
                    and r["n_kept_gated"] == r["T"])
    same = sum(1 for r in rows if bool(r["correct_gated"]) == bool(r["correct_plain"]))
    closed = sum(1 for r in rows if not r["gate_open"])
    return name, len(rows), kept_full, same, closed


def main():
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("# b = 1.0 identity audit")
    emit()
    emit("At b=1.0 the gated arm must keep the full cache and match the plain "
         "outcome row for row. If it does, r(x) >= 0 is an identity and the "
         "empty r=-1 bin is vacuous.")
    emit()
    emit("| cell | rows at b=1.0 | n_kept == T | gated == plain | gate closed |")
    emit("|---|---:|---:|---:|---:|")

    targets = []
    for policy in POLICIES:
        for model_name, slug in MODELS:
            p = cell_path(slug, policy)
            if os.path.exists(p):
                targets.append((p, f"{policy} {model_name} 4K"))
    for fn in EXTRA:
        p = os.path.join(RESULTS, fn)
        if os.path.exists(p):
            targets.append((p, fn.replace("gated_", "").replace(".jsonl", "")))

    violations = []
    for path, name in targets:
        name, n, kept_full, same, closed = audit(path, name)
        if n == 0:
            emit(f"| {name} | 0 | -- | -- | -- |")
            violations.append((name, "no b=1.0 rows"))
            continue
        mark_k = "" if kept_full == n else " **X**"
        mark_s = "" if same == n else " **X**"
        emit(f"| {name} | {n} | {kept_full}/{n}{mark_k} | {same}/{n}{mark_s} "
             f"| {closed}/{n} |")
        if kept_full != n:
            violations.append((name, f"n_kept != T on {n - kept_full} rows"))
        if same != n:
            violations.append((name, f"gated != plain on {n - same} rows"))

    emit()
    emit(f"Cells audited: {len(targets)}")
    emit()
    if violations:
        emit("## Violations")
        emit()
        for name, why in violations:
            emit(f"- {name}: {why}")
        emit()
        emit("CHECK: FAIL - the r(x) >= 0 identity is not universal; the App. D "
             "wording must be weakened to the cells that satisfy it.")
        with open(out_path("budget1_identity_audit.md"), "w") as f:
            f.write("\n".join(lines) + "\n")
        raise SystemExit(1)
    emit("CHECK: PASS - r(x) >= 0 holds by construction on every audited cell, "
         "so the empty r=-1 bin is vacuous and cannot be cited as evidence of "
         "harmlessness.")
    with open(out_path("budget1_identity_audit.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
