import torch
import numpy as np
import random

from datasets import load_dataset, Dataset, concatenate_datasets
from functools import partial

prompt_template = "Read the following text and summarize it.\n\nText: {context}"
answer_template = "Answer: {answer}"

def apply_chat_template(
    example,
    tokenizer,
) -> dict[str, str]:
    prompt = prompt_template.format(context=example["text"])
    summary = random.choice(example["summary"])["text"]
    answer = answer_template.format(answer=summary)
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
        # truncation=True,
        return_tensors="pt",
        padding=True,
        # padding="max_length",
    )
    return {"input_ids": outputs["input_ids"][0]}


def load_booksum_dataset(dataset_name, tokenizer, training_max_length=None, max_samples=None):
    book_dataset = load_dataset("ubaada/booksum-complete-cleaned", "books", split="train")
    # chapter_dataset = load_dataset("ubaada/booksum-complete-cleaned", "chapters", split="train")
    # dataset = concatenate_datasets([book_dataset, chapter_dataset])
    dataset = book_dataset
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
        lambda x: len(x["input_ids"]) <= training_max_length and len(x["input_ids"]) > 4096,
        num_proc=8,
    )

    return dataset

__all__ = [
    "load_booksum_dataset",
]
