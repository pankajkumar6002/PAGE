#!/usr/bin/env python
"""B8: PAGE on realistic near-tie retrieval. Pre-registered in preregistration/revision/b8_neartie_prereg.md.

Datasets (raw files in experiments/data/revision/b8_raw/, fetched by experiments/data/revision/fetch.sh and
hashed in experiments/data/revision/b8_raw/SHA256SUMS):
  nq30      NQ-open multi-document QA with 30 retrieved Wikipedia passages (Liu et al., TACL 2024,
            "Lost in the Middle"). 350 questions from a seed-0 permutation of the 2,655, gold passage
            position cycling through 0, 4, 9, 14, 19, 24, 29. Prompt: the authors' qa.prompt, verbatim.
            Score: the authors' best_subspan_em, verbatim.
  babilong  BABILong qa2 and qa3 at 4k and 8k (Kuratov et al., NeurIPS 2024 D&B), 100 questions per file.
            Prompt: the authors' DEFAULT_PROMPTS and DEFAULT_TEMPLATE, verbatim. Score: compare_answers.

Prefill, D, SnapKV masks and decoding come from longbench_gating.py, unchanged, as in B2 and B7.
Only the plain arm is decoded at each budget. The gated arm follows from the gate identity.
A question is written only when all budgets are done. On a restart, complete questions are kept.
"""
import argparse
import gzip
import json
import os
import string
import sys
import time

import numpy as np
import regex
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("PAGE_SRC", os.path.dirname(HERE))
sys.path.insert(0, SRC)
import longbench_gating as lg  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

RAW = os.environ.get("PAGE_B8_RAW", os.path.join(HERE, "..", "..", "data", "revision", "b8_raw"))
NQ_POS = [0, 4, 9, 14, 19, 24, 29]
NQ_N = 350

# ---- Liu et al. metrics.py, verbatim -------------------------------------------------------------
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def best_subspan_em(prediction, ground_truths):
    normalized_prediction = normalize_answer(prediction)
    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0


# ---- BABILong prompts.py and metrics.py, verbatim (qa2, qa3 only) ---------------------------------
TASK_TEMPLATE = '{instruction}\n\n{examples}\n\n{post_prompt}'
USER_TEMPLATE = '<context>\n{context}\n</context>\n\nQuestion: {question}'
DEFAULT_TEMPLATE = f'{TASK_TEMPLATE}\n\n{USER_TEMPLATE}'
_QA23_INSTR = ('I give you context with the facts about locations and actions of different persons '
               'hidden in some random text and a question.'
               'You need to answer the question based only on the information from the facts.\n'
               'If a person got an item in the first location and travelled to the second location '
               'the item is also in the second location. '
               'If a person dropped an item in the first location and moved to the second location '
               'the item remains in the first location.')
BABI_PROMPTS = {
    'qa2': {
        'instruction': _QA23_INSTR,
        'examples':
            '<example>\n'
            'Charlie went to the kitchen. Charlie got a bottle. Charlie moved to the balcony. '
            'Where is the bottle?\n'
            'Answer: The bottle is in the balcony.\n'
            '</example>\n'
            '<example>\n'
            'Alan moved to the garage. Alan got a screw driver. Alan moved to the kitchen. Where '
            'is the screw driver?\n'
            'Answer: The screw driver is in the kitchen.\n'
            '</example>',
        'post_prompt':
            'Always return your answer in the following format: The ’item’ is in ’location’. '
            'Do not write anything else after that.'
    },
    'qa3': {
        'instruction': _QA23_INSTR.replace('a question.You', 'a question. You'),
        'examples':
            '<example>\n'
            'John journeyed to the bedroom. Mary grabbed the apple. Mary went back to the bathroom. '
            'Daniel journeyed to the bedroom. Daniel moved to the garden. Mary travelled to the kitchen. '
            'Where was the apple before the kitchen?\n'
            'Answer: Before the kitchen the apple was in the bathroom.\n'
            '</example>\n'
            '<example>\n'
            'John went back to the bedroom. John went back to the garden. John went back to the kitchen. '
            'Sandra took the football. Sandra travelled to the garden. Sandra journeyed to the bedroom. '
            'Where was the football before the bedroom?\n'
            'Answer: Before the bedroom the football was in the garden.\n'
            '</example>',
        'post_prompt':
            'Always return your answer in the following format: '
            'Before the $location_1$ the $item$ was in the $location_2$. Do not write anything else after that.'
    },
}
BABI_LABELS = ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office']


def get_formatted_input(context, question, examples, instruction, post_prompt, template=DEFAULT_TEMPLATE):
    formatted_input = template.format(instruction=instruction, examples=examples, post_prompt=post_prompt,
                                      context=context.strip(), question=question)
    return formatted_input.strip()


def preprocess_output(output):
    output = output.lower()
    output = output.split('.')[0]
    output = output.split('<context>')[0]
    output = output.split('<example>')[0]
    output = output.split('Question')[0]
    return output


def compare_answers(target, output, question, task_labels):
    output = preprocess_output(output)
    target = target.lower()
    task_labels = {label.lower() for label in task_labels}
    labels_in_question = {label for label in task_labels if label in question.lower()}
    labels_in_output = {label for label in task_labels if label in output}
    labels_in_output = labels_in_output - labels_in_question
    if ',' in target and len(target) > 3:
        subtargets = target.split(',')
        num_subtargets = len(subtargets)
        if all([t in labels_in_output for t in subtargets]) and len(labels_in_output) == num_subtargets:
            return True
    else:
        if target in labels_in_output and len(labels_in_output) == 1:
            return True
    return False


# ---- loaders ---------------------------------------------------------------------------------------
def load_nq():
    files = {p: [json.loads(l) for l in gzip.open(os.path.join(RAW, f"nq30_gold_at_{p}.jsonl.gz"), "rt")] for p in NQ_POS}
    n = len(files[0])
    assert all(len(v) == n for v in files.values())
    assert all(files[p][i]["question"] == files[0][i]["question"] for p in NQ_POS for i in range(n))
    order = np.random.RandomState(0).permutation(n)[:NQ_N]
    tmpl = open(os.path.join(RAW, "qa.prompt")).read().rstrip("\n")
    rows = []
    for k, i in enumerate(order):
        pos = NQ_POS[k % len(NQ_POS)]
        r = files[pos][int(i)]
        assert [c["isgold"] for c in r["ctxs"]].index(True) == pos
        results = "\n".join(f"Document [{j + 1}](Title: {c['title']}) {c['text']}" for j, c in enumerate(r["ctxs"]))
        user = tmpl.format(search_results=results, question=r["question"])
        rows.append({"task": "nq30", "doc": f"nq{int(i)}", "user": user, "answers": r["answers"], "gold_pos": pos,
                     "question": r["question"], "max_new": 64})
    return rows


def load_babilong():
    rows = []
    for qa in ("qa2", "qa3"):
        for L in ("4k", "8k"):
            data = json.load(open(os.path.join(RAW, f"babilong_{qa}_{L}.json")))
            assert len(data) == 100
            for j, r in enumerate(data):
                p = BABI_PROMPTS[qa]
                user = get_formatted_input(r["input"], r["question"], p["examples"], p["instruction"], p["post_prompt"])
                rows.append({"task": f"babilong_{qa}", "doc": f"{qa}_{L}_{j}", "user": user, "answers": [r["target"]],
                             "length": L, "question": r["question"], "max_new": 32})
    return rows


def score(ex, pred):
    if ex["task"] == "nq30":
        return bool(best_subspan_em(pred, ex["answers"]))
    return bool(compare_answers(ex["answers"][0], pred, ex["question"], BABI_LABELS))


def build_prompt(tok, ex):
    return tok.apply_chat_template([{"role": "user", "content": ex["user"]}], tokenize=False, add_generation_prompt=True)


def complete_ids(path, nb):
    """Question ids with all budgets written; rewrite the log atomically without partial questions."""
    if not os.path.exists(path):
        return set()
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    cnt = {}
    for r in rows:
        cnt[r["id"]] = cnt.get(r["id"], 0) + 1
    done = {i for i, c in cnt.items() if c == nb}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            if r["id"] in done:
                f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["nq30", "babilong"])
    p.add_argument("--model", required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--budgets", default="1.0,0.5,0.25,0.125,0.0625")
    p.add_argument("--attn_impl", default="sdpa", choices=["eager", "sdpa"])
    p.add_argument("--two_pass", action="store_true")
    p.add_argument("--max_context_tokens", type=int, default=16000)
    p.add_argument("--obs_window", type=int, default=32)
    p.add_argument("--n_sink", type=int, default=4)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="dry run: first N questions per task")
    args = p.parse_args()
    if not args.two_pass:
        sys.exit("B8 requires --two_pass, as B2 and B7")

    budgets = [float(b) for b in args.budgets.split(",")]
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation=args.attn_impl).to(device).eval()
    rows = load_nq() if args.dataset == "nq30" else load_babilong()
    if args.limit:
        seen = {}
        keep = []
        for r in rows:
            seen[r["task"]] = seen.get(r["task"], 0) + 1
            if seen[r["task"]] <= args.limit:
                keep.append(r)
        rows = keep

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = complete_ids(args.out, len(budgets))
    fout = open(args.out, "a")
    t0 = time.time()
    n_skip, n_oom = 0, 0
    per_task_i = {}
    for ex_i, ex in enumerate(rows):
        ti = per_task_i.get(ex["task"], 0)
        per_task_i[ex["task"]] = ti + 1
        prompt = build_prompt(tok, ex)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False, truncation=False).input_ids.to(device)
        T = ids.shape[1]
        if T > args.max_context_tokens:
            n_skip += 1
            print(f"skip {ex['task']}/{ex_i} T={T}", flush=True)
            continue
        if ex_i in done:
            continue
        try:
            keys, values, scr, drop, per_layer = lg.prefill_and_score_with_agreement(model, ids, args.obs_window, args.top_k, device, two_pass=True)
            recs = []
            for b in budgets:
                pred, n_kept = lg.evaluate_one(model, tok, ids, keys, values, scr, T, b, ex["max_new"], device, args.obs_window, args.n_sink, do_evict=True)
                rec = {"id": ex_i, "task_index": ti, "task": ex["task"], "doc": ex["doc"], "budget": b, "T": T, "drop": float(drop),
                       "n_kept_plain": n_kept, "answers": ex["answers"], "pred_plain": pred[:2000], "correct_plain": score(ex, pred),
                       "max_new": ex["max_new"]}
                for k in ("gold_pos", "length"):
                    if k in ex:
                        rec[k] = ex[k]
                if b == 1.0:
                    rec["per_layer"] = per_layer
                recs.append(rec)
        except torch.cuda.OutOfMemoryError:
            n_oom += 1
            torch.cuda.empty_cache()
            print(f"oom {ex['task']}/{ex_i} T={T}", flush=True)
            continue
        except RuntimeError as e:
            if "out of memory" in str(e) or "CUBLAS_STATUS_ALLOC_FAILED" in str(e):
                n_oom += 1
                torch.cuda.empty_cache()
                print(f"oom {ex['task']}/{ex_i} T={T}", flush=True)
                continue
            raise
        for rec in recs:
            fout.write(json.dumps(rec) + "\n")
        fout.flush()
        del keys, values, scr
        torch.cuda.empty_cache()
        if (ex_i + 1) % 10 == 0:
            print(f"[{ex_i + 1}/{len(rows)}] {ex['task']} T={T} D={drop:.4f} {time.time() - t0:.0f}s skip={n_skip} oom={n_oom}", flush=True)
    fout.close()
    print(f"b8 done dataset={args.dataset} rows={len(rows)} skipped={n_skip} oom={n_oom}", flush=True)


if __name__ == "__main__":
    main()
