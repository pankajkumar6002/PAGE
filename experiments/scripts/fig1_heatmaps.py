"""Figure 1, panel 3 heatmaps, drawn from measured attention.

The hand-drawn panel had the dilution-prone case backwards: its "Early layers"
map showed heads differing (scattered) and its "Late layers" map showed heads
agreeing. Every cell we measure shows the opposite ordering, which is forced by
the sign of D = a_early - a_late being positive throughout.

What the maps must show, per task:

  dilution-prone   early: rows look alike (heads share a broad token set)
                   late : rows diverge   (heads specialise)  -> steep drop
  capacity-bound   early: rows already alike, over a narrow set
                   late : still alike                        -> shallow drop

These are rendered from real prefill attention, so the picture cannot drift
from the statistic again. Each map is [heads x tokens]: one row per head, the
mean attention over the observation window, restricted to the union of the
heads' top-k tokens so the structure is visible at print size.
"""
import argparse
import os

import torch

from paths import out_path


def build_prompt(tok, ex):
    user = ex["context"] + "\n\n" + ex["question"]
    if ex.get("answer_prefix"):
        user += "\n" + ex["answer_prefix"]
    return tok.apply_chat_template(
        [{"role": "system", "content": "Answer the question concisely."},
         {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--config", default="4096")
    ap.add_argument("--obs", type=int, default=32)
    ap.add_argument("--top_k", type=int, default=32)
    ap.add_argument("--n_show", type=int, default=48, help="tokens per map")
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ds = load_dataset("simonjegou/ruler", a.config, split="test")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="eager",
        device_map={"": 0}).eval()
    L = model.config.num_hidden_layers
    third = L // 3

    panels = {}
    agree = {}
    for task, label in [("vt", "dilution-prone"),
                        ("niah_multikey_3", "capacity-bound")]:
        ex = ds.filter(lambda r: r["task"] == task)[0]
        ids = tok(build_prompt(tok, ex), return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False, output_attentions=True)
        maps = {}
        for name, layers in (("early", range(third)),
                             ("late", range(L - third, L))):
            acc = None
            for li in layers:
                A = out.attentions[li][0][:, -a.obs:, :].float().mean(1)  # [H,T]
                acc = A if acc is None else acc + A
            acc = acc / len(list(layers))
            # D is the Jaccard of per-head top-k SETS, so show set membership,
            # not attention magnitude. Magnitude maps are swamped by the
            # attention sink and say nothing about agreement.
            acc[:, :4] = 0.0                      # drop sink tokens
            top = acc.topk(a.top_k, dim=-1).indices
            H, T = acc.shape
            member = torch.zeros(H, T)
            member.scatter_(1, top.cpu(), 1.0)
            # order columns by how many heads select them: a token chosen by
            # every head becomes a solid band, one chosen by a single head a
            # lone mark, so agreement is read off directly
            freq = member.sum(0)
            cols = freq.argsort(descending=True)[:a.n_show]
            maps[name] = member[:, cols]
            # sanity: the picture must exhibit the D it claims to illustrate
            full = torch.zeros(H, T); full.scatter_(1, top.cpu(), 1.0)
            sets = [set(torch.nonzero(full[h]).flatten().tolist()) for h in range(H)]
            import itertools
            js = [len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j]))
                  for i, j in itertools.combinations(range(H), 2)]
            agree[(label, name)] = sum(js) / len(js)
        panels[label] = maps
        del out
        torch.cuda.empty_cache()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.4))
    for r, (label, maps) in enumerate(panels.items()):
        for c, name in enumerate(("early", "late")):
            M = maps[name]
            ax = axes[r][c]
            ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{name} layers", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{label}\nheads", fontsize=8)
            ax.set_xlabel("tokens", fontsize=7)
    fig.suptitle("Rows are heads; a mark means the token is in that head's top-$k$.\n"
                 "Solid columns = heads agree. Dilution-prone: agreed early, split late\n"
                 "(large $D$). Capacity-bound: agreed throughout (small $D$).", fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path("fig1_heatmaps.pdf"))
    fig.savefig(out_path("fig1_heatmaps.png"), dpi=200)
    print("\nagreement in the rendered example (mean pairwise Jaccard):")
    for lab in ("dilution-prone", "capacity-bound"):
        e, l = agree[(lab, "early")], agree[(lab, "late")]
        print(f"  {lab:16s} early {e:.3f}  late {l:.3f}  D = {e-l:+.3f}")
    dd = agree[("dilution-prone","early")]-agree[("dilution-prone","late")]
    cc = agree[("capacity-bound","early")]-agree[("capacity-bound","late")]
    ok = dd > 0 and cc > 0 and dd > cc
    print("  CHECK:", "example matches the claim (both fall, dilution falls more)"
          if ok else "EXAMPLE CONTRADICTS THE CAPTION - pick another input")
    print("wrote", out_path("fig1_heatmaps.pdf"))


if __name__ == "__main__":
    main()
