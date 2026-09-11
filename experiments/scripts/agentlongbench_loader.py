"""E11 agentic slice: load AgentLongBench (ign1s/AgentLongBench, MIT) into the
row shape longbench_gating.py consumes, so the same gate/eviction runner scores
agentic tool-call traces.

VERIFIED dataset shape (streamed, not assumed):
  * One HF config ("default"), WebDataset-style: each shard is a `jsonl` bytes
    blob keyed `benchmark/{setting}/{ctxlen}/{category}/{task}`, e.g.
    `benchmark/ki-c/32k/tool_response/count_frequency_tool`.
  * Context buckets: 32k (smallest), 64k, 128k, 256k, 512k, 1M, 2M, 4M.
  * Inner record fields: id, sample_id, round, question_type, answer,
    tool_result, messages, answer_index, QA_type, enough_token, question.
    `messages` holds the agent/tool-call trace = the long context.

BOUNDARY CAVEAT (frozen in E11 prereg): even the 32k bucket is 2x PAGE's 16K
fitting max. longbench_gating.py skips inputs above --max_context_tokens
(default 24000), so by default these rows are SKIPPED. Raising the cap admits
them but takes them OUT of the fitting range; such results are reported
gate-closure-only, never pooled into the in-range f. This loader defaults to the
32k bucket and emits a clear warning about the boundary.

METRIC CAVEAT (verified by inspecting the data, not assumed): AgentLongBench
answers are short integers ("0", "1", "8", ...). longbench_gating.is_correct is
a substring match, so `correct_plain`/`correct_gated` on these rows are
essentially uninformative (a verbose agent output almost always contains a
single digit). Therefore only the GATE-CLOSURE fields (`drop`, `gate_open`) are
trustworthy on AgentLongBench; the accuracy fields must NOT be read as an
accuracy signal. The E11 prereg already restricts the agentic slice to
gate-closure / D-ordering, consistent with this. Downstream analysis must honor
that (do not compute f or Delta from these rows).

This module is import-only for longbench_gating (it monkeypatches load_dataset),
mirroring longbench_capkv_seed2.py's pattern. Run via run_e11_gaps.sh.
"""
import io
import json
import sys

SETTING = "ki-c"     # knowledge-intensive, concise responses (shortest traces).
CTXLEN = "32k"       # smallest bucket; closest to PAGE's boundary.


def _blob_to_records(blob):
    """A shard's jsonl bytes blob -> list of dict records, skipping bad lines."""
    recs = []
    text = blob.decode("utf-8", errors="replace") if isinstance(blob, (bytes, bytearray)) else blob
    for line in io.StringIO(text):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return recs


def _messages_to_context(messages):
    """Serialize the agent/tool-call trace into a single context string."""
    if isinstance(messages, str):
        return messages
    parts = []
    try:
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role", "")
                content = m.get("content", "")
                parts.append(f"{role}: {content}" if role else str(content))
            else:
                parts.append(str(m))
    except TypeError:
        return str(messages)
    return "\n".join(parts)


def load_agentlongbench(setting=SETTING, ctxlen=CTXLEN, tasks=None, max_examples=50):
    """Return rows in longbench_gating's expected shape: each has `context`,
    `question`, `answers` (list), `_task`, and optional `answer_prefix`.

    Raises SystemExit with a clear message on any load failure (offline, gated
    repo, changed schema) rather than a bare traceback."""
    try:
        from datasets import load_dataset
    except Exception as e:   # pragma: no cover
        raise SystemExit(f"datasets not importable: {e}")

    try:
        ds = load_dataset("ign1s/AgentLongBench", split="train", streaming=True)
    except Exception as e:
        raise SystemExit(
            f"cannot load ign1s/AgentLongBench (offline or gated?): "
            f"{type(e).__name__}: {e}")

    prefix = f"benchmark/{setting}/{ctxlen}/"
    out = []
    per_task = {}
    try:
        for shard in ds:
            key = shard.get("__key__", "")
            if not key.startswith(prefix):
                continue
            task = key[len(prefix):].replace("/", "_")
            if tasks and task not in tasks:
                continue
            for rec in _blob_to_records(shard.get("jsonl", b"")):
                if per_task.get(task, 0) >= max_examples:
                    break
                ctx = _messages_to_context(rec.get("messages", rec.get("tool_result", "")))
                q = rec.get("question", "")
                ans = rec.get("answer", "")
                out.append({
                    "context": ctx,
                    "question": q,
                    "answers": [str(ans)] if not isinstance(ans, list) else [str(a) for a in ans],
                    "_task": f"alb_{task}",
                    "answer_prefix": "",
                })
                per_task[task] = per_task.get(task, 0) + 1
    except Exception as e:
        raise SystemExit(f"error iterating AgentLongBench shards: {type(e).__name__}: {e}")

    if not out:
        raise SystemExit(
            f"no AgentLongBench rows matched setting={setting} ctxlen={ctxlen} "
            f"tasks={tasks}. Check the bucket name against the streamed keys.")
    sys.stderr.write(
        f"[agentlongbench] loaded {len(out)} rows from {setting}/{ctxlen} "
        f"across {len(per_task)} tasks. BOUNDARY: {ctxlen} is beyond PAGE's 16K "
        f"fitting range; longbench_gating will SKIP any input above "
        f"--max_context_tokens unless raised, and raised results are "
        f"out-of-fitting-range (gate-closure only, not pooled).\n")
    return out


if __name__ == "__main__":
    # Smoke test: print the shape of the first few rows without running a model.
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--setting", default=SETTING)
    p.add_argument("--ctxlen", default=CTXLEN)
    p.add_argument("--max_examples", type=int, default=3)
    a = p.parse_args()
    rows = load_agentlongbench(setting=a.setting, ctxlen=a.ctxlen,
                               max_examples=a.max_examples)
    for r in rows[:3]:
        print(json.dumps({k: (v[:80] if isinstance(v, str) else v)
                          for k, v in r.items()}, ensure_ascii=False)[:300])