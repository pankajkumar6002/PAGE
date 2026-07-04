import torch
import numpy as np
import random

from datasets import load_dataset, Dataset
from functools import partial

prompt_template = "Read the context and answer the question.\n\nContext: {context}\n\nQuestion: {question}"
answer_template = "Answer: {answer}"

def apply_chat_template(
    example,
    tokenizer,
) -> dict[str, str]:
    prompt = prompt_template.format(
        context=example["extended_context"],
        question=example["question"],
    )
    answer = answer_template.format(
        answer=example["answer"],
    )
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


def tokenize_fn(tokenizer, examples):
    outputs = tokenizer(
        examples["text"],
        add_special_tokens=False,
        truncation=True,
        return_tensors="pt",
        padding=True,
        # padding="max_length",
    )
    return {"input_ids": outputs["input_ids"][0]}


def is_valid(ex):
    for col in ("extended_context", "question", "answer"):
        v = ex.get(col, None)
        if v is None:
            return False
        # optional: also treat empty strings/lists/dicts as "null"
        if isinstance(v, str) and not v.strip():
            return False
        if isinstance(v, (list, dict)) and len(v) == 0:
            return False
    return True


def load_buddhi_dataset(dataset_name, tokenizer, training_max_length=None, max_samples=None):
    dataset = load_dataset("aiplanet/buddhi-dataset", "gpt4", split="train")
    # filter out examples with empty 'extended_context', or 'question', or 'answer' fields
    dataset = dataset.filter(is_valid, num_proc=8)  # adjust num_proc to your CPU
    if max_samples is not None and max_samples > 0:
        dataset = dataset.select(range(max_samples))

    dataset = dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
    )

    dataset = dataset.map(
        partial(tokenize_fn, tokenizer),
        batched=False,
        num_proc=8,
    )

    dataset = dataset.filter(
        lambda x: len(x["input_ids"]) < training_max_length and len(x["input_ids"]) > 4096,
        num_proc=8,
    )

    return dataset

__all__ = [
    "load_buddhi_dataset",
]
