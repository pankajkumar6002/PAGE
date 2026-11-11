#!/usr/bin/env python
"""B3: released gated_eviction.py on a newer model, with thinking off in the chat template.

Pre-registered in preregistration/revision/b3_b4_models_prereg.md. All arguments pass through.
"""
import os
import sys
from pathlib import Path

SRC = os.environ.get("PAGE_SRC", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, SRC)
import gated_eviction as ge  # noqa: E402


def build_prompt(tokenizer, ex):
    user_msg = ex["context"] + "\n\n" + ex["question"]
    if ex.get("answer_prefix"):
        user_msg = user_msg + "\n" + ex["answer_prefix"]
    messages = [
        {"role": "system", "content": "Answer the question concisely."},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


ge.build_prompt = build_prompt
ge.main()
