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
        example["conversations"],
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


def load_synth_long_dataset(dataset_name, tokenizer, training_max_length=None, max_samples=None):
    dataset = load_dataset("cerebras/Synth-Long-SFT32K", split="train_convqa_raft+train_convqa_raft_syntactic+train_narrativeqa_aug_32k+train_rag_tge_raft")
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

    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col != "input_ids"]
    )

    dataset = dataset.filter(
        lambda x: len(x["input_ids"]) < training_max_length and len(x["input_ids"]) > 4096,
        num_proc=8,
    )

    return dataset

__all__ = [
    "load_synth_long_dataset",
]
