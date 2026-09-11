"""E12 (Q2 / W4): a tiny learned probe over the per-layer agreement profile a_l.

The paper's gate signal D collapses the per-layer head-agreement profile a_l to
its endpoints (early - late). On Llama-3.1-8B the profile is non-monotone with a
mid-stack spike, so the endpoint contrast inverts and per-input AUC sits at
~0.80 under the z-scored variant. This script asks the reviewer's exact
question: does a <=3-parameter logistic regression over the INTERIOR of the
profile, z-normalized on an unlabeled pilot, raise the held-out Llama AUC?

Label (frozen in preregistration/E12_profile_probe_prereg.md): the paper's own
per-input AUC label from heldout_ablation.py -- separability of NIAH-MK3 inputs
(positive) from the dilution-prone-task pool (vt, fwe, qa_1, niah_multivalue;
negative), scored by the signal. Task identity IS the label and is present in
the profile dump, so features (interior profile summaries) and label both come
from drops_*.jsonl -- no eviction-log join.

Baseline reproduction guard: the drop_D AUC recomputed here (endpoint drop,
MK3-vs-dilution) must reproduce the paper's per-input AUC (Qwen2.5-1.5B 1.000;
Llama-arch ~0.80). A mismatch aborts, since it means our construction does not
match the paper's.

Zero-GPU: runs entirely on released logs. Exits `CHECK: PASS` on success.
"""
import json
import math
import os

import numpy as np

from paths import RESULTS, DATA, out_path, require

ANCHOR = "niah_multikey_3"
DIL = ["vt", "fwe", "qa_1", "niah_multivalue"]

# (label, released profile-dump filename, id-tagged re-dump filename, expected
# paper AUC for RAW drop_D). The released drops_* files are the offline default;
# run_e12_profiles.sh produces id-tagged profiles_* dumps in DATA which, if
# present, are preferred (more inputs + explicit ids). resolve() picks whichever
# exists. NB the ~0.80 in main.tex:382 is the *z-scored* D variant, NOT raw D;
# raw D on the Llama held-out cell is 0.741 (main.tex:244).
FIT_CELL = ("Qwen2.5-1.5B", "drops_qwen15b_4k_n100.jsonl", "profiles_qwen15b_4k.jsonl", 1.000)
EVAL_CELL = ("Llama-3.1-8B", "drops_llama31_8b_4k.jsonl", "profiles_llama31_4k.jsonl", 0.741)


def resolve(released_name, redump_name):
    """Prefer the id-tagged re-dump in DATA (from run_e12_profiles.sh) if it
    exists; otherwise fall back to the released profile dump in RESULTS. Return
    the released path if neither exists so require() reports it consistently."""
    d = os.path.join(DATA, redump_name)
    if os.path.exists(d):
        return d
    return os.path.join(RESULTS, released_name)

AUC_GUARD_TOL = 0.05   # recomputed drop_D AUC must land within this of the paper's.


def read_jsonl(path):
    """Rows plus a count of skipped corrupt lines (NUL / unparseable), mirroring
    layer_subsample.read_jsonl so a silently shortened log cannot bias a mean."""
    require(path)
    rows, corrupt = [], 0
    try:
        with open(path, errors="replace") as f:
            for line in f:
                if not line.strip() or "\x00" in line:
                    corrupt += int(bool(line.strip()))
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    corrupt += 1
    except OSError as e:
        raise SystemExit(f"cannot read {path}: {e}")
    if not rows:
        raise SystemExit(f"no usable rows in {path} (corrupt={corrupt})")
    return rows, corrupt


def endpoint_drop(profile):
    """D = mean(first third) - mean(last third), exactly as compute_drop does."""
    L = len(profile)
    third = max(1, L // 3)
    return sum(profile[:third]) / third - sum(profile[L - third:]) / third


def interior_features(profile):
    """The three interior summaries D discards: min, normalized argmin depth,
    mid-band mean. Returns None if the interior is empty."""
    L = len(profile)
    third = max(1, L // 3)
    inter = profile[third:L - third]
    if not inter:
        return None
    arr = np.asarray(inter, dtype=float)
    mid = arr[len(arr) // 3: 2 * len(arr) // 3]
    return np.array([
        float(arr.min()),
        float(int(np.argmin(arr)) / L),
        float(mid.mean() if len(mid) else arr.mean()),
    ])


def load_cell(resolved_path):
    """Return (X interior features, y MK3-label, D endpoint drops, tasks).
    `resolved_path` is a full path already chosen by resolve()."""
    rows, corrupt = read_jsonl(resolved_path)
    X, y, D, tasks = [], [], [], []
    for r in rows:
        task = r["task"]
        if task != ANCHOR and task not in DIL:
            continue   # tasks outside the paper's AUC pool are not scored.
        prof = r["head_agreement_per_layer"]
        feats = interior_features(prof)
        if feats is None:
            continue
        X.append(feats)
        y.append(int(task == ANCHOR))
        D.append(endpoint_drop(prof))
        tasks.append(task)
    if not X:
        raise SystemExit(f"no MK3/dilution rows in {profile_path}")
    return np.asarray(X), np.asarray(y, float), np.asarray(D), tasks, corrupt


# --- pure-numpy logistic regression + AUC ---------------------------------
def zfit(X):
    mu, sd = X.mean(0), X.std(0)
    return mu, np.where(sd < 1e-9, 1.0, sd)


def zapply(X, mu, sd):
    return (X - mu) / sd


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def logreg_fit(X, y, iters=100, l2=1e-3):
    n, d = X.shape
    Xb = np.column_stack([np.ones(n), X])
    w = np.zeros(d + 1)
    for _ in range(iters):
        p = sigmoid(Xb @ w)
        W = p * (1 - p)
        grad = Xb.T @ (p - y) + l2 * np.concatenate([[0.0], w[1:]])
        H = Xb.T @ (Xb * W[:, None]) + l2 * np.eye(d + 1)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        w_new = w - step
        if np.max(np.abs(w_new - w)) < 1e-9:
            return w_new
        w = w_new
    return w


def logreg_score(X, w):
    return sigmoid(np.column_stack([np.ones(X.shape[0]), X]) @ w)


def auc(scores, labels):
    """Rank-sum (Mann-Whitney) AUC, ties averaged; pure numpy."""
    labels = np.asarray(labels)
    pos, neg = labels == 1, labels == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return (ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def auc_bruteforce(scores, labels):
    labels = np.asarray(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return wins / (len(pos) * len(neg))


def main():
    lines = []
    emit = lines.append
    emit("# E12: learned probe over the per-layer profile a_l (Q2 / W4)\n")
    emit("Label = the paper's per-input AUC label (heldout_ablation.py): NIAH-MK3 "
         "vs the dilution-prone pool (vt, fwe, qa_1, niah_multivalue). Features = "
         "interior profile summaries (min, argmin depth, mid-band mean) that the "
         "endpoint statistic D discards. Fit on {}, held-out on {}. Pure numpy.\n"
         .format(FIT_CELL[0], EVAL_CELL[0]))

    Xf, yf, Df, tf, cf = load_cell(resolve(FIT_CELL[1], FIT_CELL[2]))
    Xe, ye, De, te, ce = load_cell(resolve(EVAL_CELL[1], EVAL_CELL[2]))
    emit(f"- Fit {FIT_CELL[0]}: N={len(yf)} ({int(yf.sum())} MK3 / {int((ye==0).sum())} dil); corrupt {cf}.")
    emit(f"- Eval {EVAL_CELL[0]}: N={len(ye)} ({int(ye.sum())} MK3 / {int((ye==0).sum())} dil); corrupt {ce}.\n")

    # --- baseline reproduction guard: recomputed drop_D AUC ~ paper's ---
    # higher D => less MK3-like, so score capacity/MK3-ness by -D.
    fit_D_auc = auc(-Df, yf)
    eval_D_auc = auc(-De, ye)
    for label, got, want in [(FIT_CELL[0], fit_D_auc, FIT_CELL[3]),
                             (EVAL_CELL[0], eval_D_auc, EVAL_CELL[3])]:
        if not math.isnan(got) and abs(got - want) > AUC_GUARD_TOL:
            raise SystemExit(
                f"GUARD FAIL: recomputed drop_D AUC on {label} = {got:.3f}, "
                f"paper reports ~{want:.3f} (tol {AUC_GUARD_TOL}). Label/feature "
                f"construction does not match heldout_ablation.py.")

    # z-scored D baseline on the eval cell's own unlabeled pilot.
    Dz = (De - De.mean()) / (De.std() if De.std() > 1e-9 else 1.0)
    zD_auc = auc(-Dz, ye)

    # --- the probe: fit interior features on the fit cell, evaluate held-out ---
    muf, sdf = zfit(Xf)
    mue, sde = zfit(Xe)           # eval cell standardized on its own pilot (unlabeled)
    w = logreg_fit(zapply(Xf, muf, sdf), yf)
    probe_fit_auc = auc(logreg_score(zapply(Xf, muf, sdf), w), yf)
    probe_eval_auc = auc(logreg_score(zapply(Xe, mue, sde), w), ye)

    emit("## Held-out AUC on {} (NIAH-MK3 vs dilution pool)\n".format(EVAL_CELL[0]))
    emit("| predictor | held-out AUC |")
    emit("|---|---:|")
    emit(f"| raw D | {eval_D_auc:.3f} |")
    emit(f"| z-scored D (paper's current best) | {zD_auc:.3f} |")
    emit(f"| **learned profile probe** | **{probe_eval_auc:.3f}** |")
    emit("")
    emit(f"- guard: recomputed drop_D AUC = {fit_D_auc:.3f} (fit, paper 1.000), "
         f"{eval_D_auc:.3f} (eval, paper 0.741 for raw D) — within tol.")
    emit(f"- note: z-scoring a single scalar is monotonic, so z-scored-D AUC "
         f"({zD_auc:.3f}) equals raw-D AUC by construction; the paper's ~0.80 "
         f"refers to z-scored threshold transfer, not per-input AUC on this cell.")
    emit(f"- probe in-sample AUC on {FIT_CELL[0]}: {probe_fit_auc:.3f} "
         f"(D already separates at {fit_D_auc:.3f} here, so no in-sample headroom).")
    emit(f"- probe weights [intercept, min, argmin_depth, mid_band]: {np.round(w,3).tolist()}")
    emit("")

    margin = probe_eval_auc - zD_auc
    if margin >= 0.05:
        v = (f"CLOSES THE GAP: probe {probe_eval_auc:.3f} >= z-scored D "
             f"{zD_auc:.3f} + 0.05 (margin {margin:+.3f}). The interior profile "
             f"carries per-input signal the endpoint contrast misses on Llama, "
             f"answering the paper's open problem.")
    elif margin > 0:
        v = (f"PARTIAL: probe {probe_eval_auc:.3f} beats z-scored D {zD_auc:.3f} "
             f"by {margin:+.3f}, under the +0.05 target. Interior structure helps "
             f"but is not the whole gap.")
    else:
        v = (f"NO IMPROVEMENT: probe {probe_eval_auc:.3f} <= z-scored D "
             f"{zD_auc:.3f}. The endpoint statistic is not the bottleneck on "
             f"Llama; the failure lives elsewhere.")
    emit("## Verdict\n")
    emit(v + "\n")

    out = out_path("E12_profile_probe.md")
    try:
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
    except OSError as e:
        raise SystemExit(f"cannot write {out}: {e}")
    print("\n".join(lines))
    print(f"\nwrote {out}")

    check = auc_bruteforce(logreg_score(zapply(Xe, mue, sde), w), ye)
    if not (math.isnan(check) or abs(check - probe_eval_auc) < 1e-9):
        raise SystemExit(f"CHECK FAIL: AUC {probe_eval_auc} vs brute-force {check}")
    print("CHECK: PASS")


if __name__ == "__main__":
    main()