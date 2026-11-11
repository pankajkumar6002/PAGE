#!/usr/bin/env python
"""B7: PAGE on L-Eval (An et al., ACL 2024), closed-ended tasks. Pre-registered in preregistration/revision/b7_leval_prereg.md.

Data: L-Eval's jsonl files from the Hub repo L4NLP/LEval at commit 43b9dbf06c239360b183c76cb5d90ebe4f5e3070, read as data
(experiments/data/revision/leval/SHA256SUMS, fetched by experiments/data/revision/fetch.sh). The dataset's loading script
is not run.

Arms:
  leval   L-Eval's own prompts: the system prompts of Baselines/LEval_config.py get_sys_prompt and the message formats of
          Baselines/llama2-chat-test.py (GitHub OpenLMLab/LEval at cd34b050), put through each model's chat template.
  quote   tpo, quality and coursera only: quote-then-answer, as B2b, so that eviction harm can show on multiple choice.

Scoring reimplements L-Eval's exam scoring (Evaluation/auto_eval.py process_* and em.py exact_match_score, same commit).
A question counts as correct when its L-Eval score is 1. The score itself (0, 0.25 or 1) is logged too.

Only the plain SnapKV arm is decoded at each budget. The gated arm follows from the gate identity. Prefill, scoring,
masks and decoding are longbench_gating.py's, unchanged, exactly as in B2.
"""
import argparse
import hashlib
import json
import os
import re
import string
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("PAGE_SRC", os.path.dirname(HERE))
sys.path.insert(0, SRC)
import longbench_gating as lg  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

LEVAL_DIR = os.environ.get("PAGE_LEVAL_DIR", os.path.join(HERE, "..", "..", "data", "revision", "leval", "Exam"))
TASKS = ["tpo", "quality", "coursera", "topic_retrieval_longchat", "gsm100", "sci_fi"]  # codeU: no input fits the cap
OPTION_TASKS = ("tpo", "quality", "coursera")

# L-Eval's get_sys_prompt, verbatim (including its typos), for the tasks used here.
SYS_LEVAL = {
    "gsm100": "Given several question answer pairs, you need to follow a similar format to answer the last question. Make sure the response is end with The answer is _ . ",
    "topic_retrieval_longchat": "Below is a record of our previous conversation on many different topics. You are the ASSISTANT, and I am the USER. At the beginning of each topic, the USER will say 'I would like to discuss the topic of <TOPIC>'. Memorize each <TOPIC>. At the end of the record, I will ask you to retrieve the first/second/third topic names. Now the record start. ",
    "coursera": "Now you are given a very long document. Please follow the instruction based on this document. For multi-choice questions, there could be a sinlge correct option or multiple correct options. Please only provide the letter corresponding to the answer (like A or AB) when answering. ",
    "sci_fi": "Now you are given a scientific fiction. I will ask you some questions and the answer should be \"True\" or \"False\". Notice that you should answer the question based on the evidence in the document instead of your background knowledge.",
    "tpo": "Now you are given a very long document. Please follow the instruction based on this document. For multi-choice questions, there is only a sinlge correct option. Please only provide the letter corresponding to the answer (like A or B) when answering. For other questions, please directly give the concise and accurate answer. ",
}
SYS_LEVAL["quality"] = SYS_LEVAL["tpo"]

# Quote arm (ours). The B2b tail for single-answer tasks, and the same with several letters allowed for coursera.
SYS_QUOTE = {
    "tpo": "Now you are given a very long document. Please follow the instruction based on this document. For multi-choice questions, there is only a single correct option.",
    "coursera": "Now you are given a very long document. Please follow the instruction based on this document. For multi-choice questions, there could be a single correct option or multiple correct options.",
}
SYS_QUOTE["quality"] = SYS_QUOTE["tpo"]
TAIL_QUOTE = ("First quote the sentence from the text that best supports the answer. Then give the letter "
              "of the correct option on a final line in the form Answer: X. Keep it under 60 words.")  # B2b, verbatim
TAIL_QUOTE_MULTI = ("First quote the sentence from the text that best supports the answer. Then give the letters "
                    "of all correct options on a final line in the form Answer: X, for example Answer: B or Answer: ACD. Keep it under 60 words.")

MAX_NEW_LEVAL = {"tpo": 32, "quality": 32, "coursera": 32, "sci_fi": 32, "topic_retrieval_longchat": 64, "gsm100": 512}  # gsm100: L-Eval's 512 (Amendment 1)
MAX_NEW_QUOTE = 128
CHAT_KW = {}  # extra chat-template arguments; --no_thinking sets enable_thinking=False for Qwen3, as in B3 (Amendment 3)


def build_prompt(tok, task, document, inst, arm):
    """Returns the full prompt string. L-Eval puts "\\nAnswer:" after [/INST] for its exam prompts; here "Answer:" follows
    the chat template's generation prompt. Documents are concatenated, not str.format-ed, so braces in them are safe."""
    suffix = ""
    if arm == "leval":
        system = SYS_LEVAL[task]
        if task in ("gsm100", "topic_retrieval_longchat"):
            user = document + "\n\n" + inst
        else:
            user = "Document is as follows. " + document + " \nQuestion: " + inst + ".  Please directly give the answer without any additional output or explanation "
            suffix = "Answer:"
    else:
        system = SYS_QUOTE[task]
        tail = TAIL_QUOTE_MULTI if task == "coursera" else TAIL_QUOTE
        user = "Document is as follows. " + document + " \nQuestion: " + inst + "\n" + tail
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **CHAT_KW) + suffix


# ---- L-Eval scoring, reimplemented from Evaluation/auto_eval.py and em.py ----

PUNCT = set(string.punctuation)


def normalize_answer(s):
    """em.py: remove punctuation, then the lowercase articles a/an/the, then extra whitespace. No lowercasing."""
    s = "".join(ch for ch in s if ch not in PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match_score(prediction, ground_truth):
    """em.py exact_match_score: 1, 0.25 (option letters a subset of the gold letters) or 0."""
    flag = any(ch not in "ABCD" for ch in normalize_answer(ground_truth))
    if not flag:
        if normalize_answer(prediction) == normalize_answer(ground_truth):
            return 1
        if set(normalize_answer(prediction)).issubset(set(normalize_answer(ground_truth))):
            return 0.25
        return 0
    try:
        return int(float(prediction) == float(ground_truth))
    except ValueError:
        return 1 if ground_truth.lower().replace(" ", "") in prediction.lower().replace(" ", "") else 0


def process_output_mc(response, coursera):
    """auto_eval.py process_output_mc, same branches."""
    if len(response.strip()) == 0:
        return "None"
    if response in "ABCD":
        return response
    if not coursera:
        for ch in response:
            if ch in "ABCD":
                return ch
    cleaned = ""
    rest = response
    for ch in response:
        if ch in "ABCD":
            cleaned += ch
            rest = rest[1:]
        else:
            break
    if len(cleaned) > 1:
        return "".join(sorted(set(cleaned)))
    rest = rest.split("Question")[0]
    options = re.findall(r"\s*[A-Z](?=[\s.)])", rest)
    cleaned += " ".join(options).strip()
    cleaned = re.sub(r"[^A-D]", "", cleaned)
    cleaned = "".join(sorted(set(cleaned)))
    if len(cleaned) < 1:
        has_answer = False
        for ch in rest:
            if ch in "ABCD":
                cleaned += ch
                has_answer = True
            elif has_answer:
                break
    if len(cleaned) < 1:
        cleaned = "A"
    return cleaned


def process_gt_mc(response):
    cleaned = re.sub(r"[^A-D]", "", response.split()[0])
    return cleaned if cleaned else "A"


def process_math(response):
    m = re.search(r"The answer is (\S+)", response)
    if not m:
        ret, found = "", False
        for s in response.split("\n\n")[0].split(" ")[::-1]:
            if any(c.isdigit() for c in s):
                ret, found = s, True
                break
    else:
        ret = m.group(1)
    out = ""
    for c in ret:
        if c.isdigit():
            out += c
        if c == ".":
            break
    return out


def loyalty_gt(gt):
    return gt.lower().split("[fact:")[0].strip()


def loyalty_pred(pred):
    """process_output_judge's first half: the first word containing true or false."""
    for w in pred.lower().split():
        if "true" in w:
            return "true"
        if "false" in w:
            return "false"
    return "<error>"


def parse_quote(pred, coursera):
    """Quote arm: the last "Answer: X". Single-answer tasks use B2b's pattern, verbatim. Coursera reads the rest of that
    line token by token ("AC", "A, C", "(A) and (C)", "B." all parse) and stops at the first token that is not a letter
    group, so "Answer: All of the above" is a parse failure, not A."""
    if coursera:
        idx = pred.rfind("Answer:")
        if idx < 0:
            return None
        line = pred[idx + len("Answer:"):].strip().split("\n")[0]
        letters = ""
        for t in re.split(r"[\s,;/&]+", line):
            if t == "" or t.lower() == "and":
                continue
            m = re.fullmatch(r"\(?([A-D]+)\)?[.:]?", t)
            if not m:
                break
            letters += m.group(1)
        return "".join(sorted(set(letters))) or None
    # B2b's pattern plus a guard so that a word ("Answer: Based on ...") is not read as a letter (Amendment 1)
    found = re.findall(r"Answer:\s*\(?([A-D])(?![A-Za-z])\)?", pred)
    return found[-1] if found else None


def score(task, pred, gt, arm):
    """Returns (L-Eval score, processed prediction, processed gold, parse_fail)."""
    if task in OPTION_TASKS:
        g = process_gt_mc(gt)
        if arm == "quote":
            p = parse_quote(pred, task == "coursera")
            if p is None:
                return 0, None, g, True
        else:
            p = process_output_mc(pred, task == "coursera")
            # Scored exactly as L-Eval scores it, including its fallback answer "A" when the output has no A-D letter.
            # parse_fail flags those outputs so they are reported (Amendment 1).
            return exact_match_score(p, g), p, g, len(pred.strip()) == 0 or not any(ch in "ABCD" for ch in pred)
        return exact_match_score(p, g), p, g, False
    if task == "gsm100":
        g, p = process_math(gt), process_math(pred)
        return exact_match_score(p, g), p, g, p == ""
    if task == "sci_fi":
        g, p = loyalty_gt(gt), loyalty_pred(pred)
        return exact_match_score(p, g), p, g, p == "<error>"
    if task == "topic_retrieval_longchat":
        return exact_match_score(pred, gt), pred, gt, False
    raise ValueError(task)


def load_rows(args):
    rows = []
    tasks = args.tasks.split(",")
    for task in tasks:
        with open(os.path.join(args.leval_dir, task + ".jsonl")) as f:
            docs = [json.loads(line) for line in f if line.strip()]
        for di, d in enumerate(docs):
            doc = hashlib.sha1(d["input"].encode()).hexdigest()[:12]
            for qi, (inst, out) in enumerate(zip(d["instructions"], d["outputs"])):
                rows.append({"task": task, "doc_index": di, "q_index": qi, "doc": doc, "document": d["input"], "inst": inst, "gold": out})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--arm", required=True, choices=["leval", "quote"])
    p.add_argument("--tasks", default=None)
    p.add_argument("--leval_dir", default=LEVAL_DIR)
    p.add_argument("--budgets", default="1.0,0.5,0.25,0.125,0.0625")
    p.add_argument("--attn_impl", default="sdpa", choices=["eager", "sdpa"])
    p.add_argument("--two_pass", action="store_true")
    p.add_argument("--max_context_tokens", type=int, default=16000)
    p.add_argument("--obs_window", type=int, default=32)
    p.add_argument("--n_sink", type=int, default=4)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="first N questions only (dry runs)")
    p.add_argument("--longest_first", action="store_true", help="dry runs only: longest documents first, to measure peak memory")
    p.add_argument("--per_task_limit", type=int, default=0, help="dry runs only: first N questions of each task (Amendment 3)")
    p.add_argument("--no_thinking", action="store_true", help="enable_thinking=False in the chat template, for Qwen3 (Amendment 3)")
    args = p.parse_args()
    if args.no_thinking:
        CHAT_KW["enable_thinking"] = False
    if args.tasks is None:
        args.tasks = ",".join(TASKS if args.arm == "leval" else OPTION_TASKS)
    if args.arm == "quote":
        assert all(t in OPTION_TASKS for t in args.tasks.split(",")), "the quote arm covers tpo, quality and coursera only"
    if not args.two_pass:
        print("warning: one-pass prefill keeps every layer's full attention; long inputs will not fit", flush=True)

    budgets = [float(b) for b in args.budgets.split(",")]
    assert budgets[0] == 1.0
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation=args.attn_impl).to(device).eval()
    rows = load_rows(args)
    if args.longest_first:
        rows.sort(key=lambda r: -len(r["document"]))
    if args.per_task_limit:
        kept, count = [], {}
        for r in rows:
            count[r["task"]] = count.get(r["task"], 0) + 1
            if count[r["task"]] <= args.per_task_limit:
                kept.append(r)
        rows = kept
    if args.limit:
        rows = rows[: args.limit]
    print(f"b7 model {args.model} arm {args.arm} tasks {args.tasks} questions {len(rows)} budgets {budgets} "
          f"eos {tok.eos_token_id} cap {args.max_context_tokens}", flush=True)
    print("prompt head " + repr(build_prompt(tok, rows[0]["task"], "<DOC>", rows[0]["inst"], args.arm)[:600]), flush=True)

    # Resume: keep only questions with every budget logged, and drop partial ones so they are redone.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        kept_lines, by, bad = [], {}, 0
        with open(args.out) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:  # a line cut off when the job was stopped (Amendment 1)
                    bad += 1
                    continue
                by.setdefault((r["task"], r["id"]), []).append((r, line if line.endswith("\n") else line + "\n"))
        for key, rs in by.items():
            if sorted(r["budget"] for r, _ in rs) == sorted(budgets):
                done.add(key)
                kept_lines += [ln for _, ln in rs]
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            f.writelines(kept_lines)
        os.replace(tmp, args.out)  # atomic
        print(f"resume: {len(done)} questions complete, {len(by) - len(done)} partial ones and {bad} unreadable lines dropped", flush=True)
    skipped_path = args.out + ".skipped"
    open(skipped_path, "w").close()  # skips are recomputed on every (re)start

    fout = open(args.out, "a")
    t0 = time.time()
    n_skip = n_oom = n_new = 0
    per_task_i = {}
    for ex_i, ex in enumerate(rows):
        task = ex["task"]
        ti = per_task_i.get(task, 0)
        per_task_i[task] = ti + 1
        if (task, ex_i) in done:
            continue
        prompt = build_prompt(tok, task, ex["document"], ex["inst"], args.arm)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False, truncation=False).input_ids.to(device)
        T = ids.shape[1]
        if T > args.max_context_tokens:
            n_skip += 1
            with open(skipped_path, "a") as fs:
                fs.write(json.dumps({"id": ex_i, "task": task, "doc": ex["doc"], "T": T, "reason": "over cap"}) + "\n")
            continue
        max_new = MAX_NEW_QUOTE if args.arm == "quote" else MAX_NEW_LEVAL[task]
        recs = []
        try:
            keys, values, scr, drop, per_layer = lg.prefill_and_score_with_agreement(model, ids, args.obs_window, args.top_k, device, two_pass=args.two_pass)
            for b in budgets:
                pred, n_kept = lg.evaluate_one(model, tok, ids, keys, values, scr, T, b, max_new, device, args.obs_window, args.n_sink, do_evict=True)
                sc, pp, gp, pf = score(task, pred, ex["gold"], args.arm)
                rec = {"id": ex_i, "task_index": ti, "task": task, "doc": ex["doc"], "doc_index": ex["doc_index"], "q_index": ex["q_index"],
                       "arm": args.arm, "budget": b, "T": T, "drop": float(drop), "n_kept_plain": n_kept, "gold": ex["gold"], "gold_proc": gp,
                       "pred_plain": pred[:4000], "pred_proc": pp, "leval_score": sc, "correct_plain": bool(sc == 1), "parse_fail": bool(pf),
                       "max_new": max_new}
                if b == 1.0:
                    rec["per_layer"] = per_layer
                recs.append(rec)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:  # prefill or any budget: the whole question is skipped and recorded
            if not isinstance(e, torch.cuda.OutOfMemoryError) and not any(s in str(e) for s in ("out of memory", "CUBLAS_STATUS_ALLOC_FAILED")):
                raise
            keys = values = scr = None
            torch.cuda.empty_cache()
            n_oom += 1
            with open(skipped_path, "a") as fs:
                fs.write(json.dumps({"id": ex_i, "task": task, "doc": ex["doc"], "T": T, "reason": "oom"}) + "\n")
            continue
        for rec in recs:  # write a question only once every budget is done, so a stopped run leaves no partial question
            fout.write(json.dumps(rec) + "\n")
        fout.flush()
        n_new += 1
        del keys, values, scr
        torch.cuda.empty_cache()
        if n_new % 10 == 0:
            print(f"[{ex_i + 1}/{len(rows)}] {task} T={T} D={drop:+.4f} {time.time() - t0:.0f}s skip={n_skip} oom={n_oom}", flush=True)
    fout.close()
    print(f"peak memory {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB", flush=True)
    print(f"b7 done questions={len(rows)} new={n_new} resumed={len(done)} skipped={n_skip} oom={n_oom}", flush=True)


if __name__ == "__main__":
    main()
