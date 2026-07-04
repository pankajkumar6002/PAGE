"""Analyze the Garcia-style protection ablation (n_sink=410, obs_window=410)
against the paper's baseline runs (n_sink=4, obs_window=32).

Correctness follows reanalyze_ruler.py conventions:
  vt / fwe / cwe / niah_multivalue -> ALL golds substring in pred (lowercased)
  everything else                  -> ANY gold substring in pred

Per task we report:
  - accuracy vs (nominal) budget, plus mean EFFECTIVE budget n_kept/T
    (derive_keep_mask clamps budget to >= obs_window + n_sink + 4, so with
    410+410 protection the minimum cache is ~824 tokens ~= 0.20 * 4096)
  - rho = fraction of inputs where full-KV (b=1.0) is wrong AND some b<1.0
    is correct
and compare against the baseline files.
"""
import argparse
import json
from collections import defaultdict

BASE = "/home/smlab/projects/eff-nn/experiments/results"

BASELINE_FILES = {
    "niah_multikey_3": f"{BASE}/ruler_mk3_4k_snapkv.jsonl",
    "vt": f"{BASE}/ruler_vt_fwe_4k_snapkv.jsonl",
    "fwe": f"{BASE}/ruler_vt_fwe_4k_snapkv.jsonl",
    "qa_1": f"{BASE}/ruler_qa_4k_qwen.jsonl",
}

TASK_ORDER = ["niah_multikey_3", "vt", "fwe", "qa_1"]


def is_correct(pred, golds, task):
    pred_l = pred.lower()
    if task in ("vt", "fwe", "cwe", "niah_multivalue"):
        return all(g.strip().lower() in pred_l for g in golds)
    return any(g.strip().lower() in pred_l for g in golds)


def load(path, tasks=None):
    """-> {task: {id: {budget: (correct, n_kept, T)}}}"""
    d = defaultdict(lambda: defaultdict(dict))
    for line in open(path):
        r = json.loads(line)
        if tasks and r["task"] not in tasks:
            continue
        ok = is_correct(r["pred"], r["gold"], r["task"])
        d[r["task"]][r["id"]][r["budget"]] = (ok, r.get("n_kept"), r.get("T"))
    return d


def curve(task_d):
    """-> list of (budget, acc, mean_eff_budget), rho, n_full_wrong, N"""
    budgets = sorted({b for per in task_d.values() for b in per}, reverse=True)
    full_b = max(budgets)
    N = len(task_d)
    rows = []
    for b in budgets:
        recs = [per[b] for per in task_d.values() if b in per]
        acc = sum(ok for ok, _, _ in recs) / max(1, N)
        effs = [nk / t for _, nk, t in recs if nk and t]
        eff = sum(effs) / len(effs) if effs else float("nan")
        rows.append((b, acc, eff))
    n_full_wrong = sum(1 for per in task_d.values() if not per.get(full_b, (False,))[0])
    n_rho = sum(
        1 for per in task_d.values()
        if not per.get(full_b, (False,))[0]
        and any(per[b][0] for b in per if b < full_b)
    )
    return rows, n_rho / max(1, N), n_full_wrong, N


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--garcia", default=f"{BASE}/ruler_4k_qwen15b_garcia_protect.jsonl")
    args = p.parse_args()

    garcia = load(args.garcia)
    print(f"loaded garcia run: { {t: len(v) for t, v in garcia.items()} }")

    for task in TASK_ORDER:
        if task not in garcia:
            print(f"\n## {task}: MISSING in garcia run")
            continue
        g_rows, g_rho, g_fw, g_N = curve(garcia[task])
        base = load(BASELINE_FILES[task], tasks={task})
        b_rows, b_rho, b_fw, b_N = curve(base[task])
        b_acc = {b: a for b, a, _ in b_rows}

        print(f"\n## task = {task}  (garcia N={g_N}, baseline N={b_N})")
        print("| nominal b | eff b (protected) | acc protected | acc baseline |")
        print("|---:|---:|---:|---:|")
        for b, acc, eff in g_rows:
            base_acc = b_acc.get(b)
            ba = f"{base_acc:.3f}" if base_acc is not None else "-"
            flag = " *" if eff == eff and eff > b + 1e-6 else ""
            print(f"| {b:.4g} | {eff:.3f}{flag} | {acc:.3f} | {ba} |")
        print(f"rho protected = {g_rho:.4f} ({int(g_rho*g_N+0.5)}/{g_N}, wrong@full={g_fw})"
              f"   rho baseline = {b_rho:.4f} (wrong@full={b_fw})")
        print("(* = effective budget exceeds nominal: protection floor engaged)")


if __name__ == "__main__":
    main()
