import torch
import numpy as np
import random

from datasets import load_dataset, Dataset
from functools import partial


def apply_chat_template(
    example,
    tokenizer,
) -> dict[str, str]:
    text = tokenizer.apply_chat_template(
        example["messages"],
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


def load_ultrachat_dataset(dataset_name, tokenizer, training_max_length=None):
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

    dataset = dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns="messages",  # renamed to "text"
    )

    dataset = dataset.filter(
        lambda x: len(x["text"]) > 512,
        num_proc=8,
    )

    dataset = dataset.map(
        partial(tokenize_fn, tokenizer),
        batched=False,
        num_proc=8,
    )

    # if training_max_length is not None:
    #     dataset = build_optimized_chunks(
    #         dataset=dataset, 
    #         chunk_size=training_max_length,
    #         tokenizer=tokenizer,
    #     )

    return dataset

__all__ = [
    "load_ultrachat_dataset",
]
