import torch
import numpy as np
import random

from datasets import load_dataset, Dataset
from functools import partial

prompt_template = "You are given a math problem.\n\nProblem: {question}\n\n You need to solve the problem step by step. First, you need to provide the chain-of-thought, then provide the final answer.\n\n Provide the final answer in the format: Final answer:  \\boxed{{}}"


def apply_chat_template(
    example,
    tokenizer,
) -> dict[str, str]:
    prompt = prompt_template.format(question=example["problem"])
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": example["messages"][1]["content"]},
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


def build_optimized_chunks(dataset, chunk_size, tokenizer):
    samples = []
    for example in dataset:
        input_ids = torch.as_tensor(example["input_ids"])
        seq_len = input_ids.size(0)
        samples.append((input_ids, seq_len))
    
    samples.sort(key=lambda x: -x[1])  
    
    chunks = [] 
    
    for input_ids, seq_len in samples:
        best_idx = -1
        min_remainder = chunk_size + 1  
        

        for i, chunk in enumerate(chunks):
            total = chunk['current_length'] + seq_len
            if total <= chunk_size:
                remainder = chunk_size - total
                if remainder < min_remainder:
                    min_remainder = remainder
                    best_idx = i
        
        if best_idx != -1:
            chunks[best_idx]['input_ids'].append(input_ids)
            start_idx = chunks[best_idx]['current_length']
            end_idx = start_idx + seq_len
            chunks[best_idx]['indices'].append([start_idx, end_idx])
            chunks[best_idx]['current_length'] += seq_len
        else:
            chunks.append({
                'input_ids': [input_ids],
                'indices': [[0, seq_len]],
                'current_length': seq_len
            })
    
    processed_chunks = []
    random.shuffle(chunks)
    for chunk in chunks:
        full_chunk = torch.cat(chunk['input_ids'])
        if full_chunk.size(0) < chunk_size:
            padding_length = chunk_size - full_chunk.size(0)
            padding = torch.full((padding_length,), fill_value=tokenizer.eos_token_id, dtype=torch.long)
            full_chunk = torch.cat([full_chunk, padding], dim=0)
            attention_mask = torch.cat([torch.ones(full_chunk.size(0) - padding_length, dtype=torch.long), torch.zeros(padding_length, dtype=torch.long)], dim=0)
        else:
            attention_mask = torch.ones(full_chunk.size(0), dtype=torch.long)
        processed_chunks.append({
            "input_ids": full_chunk.numpy(),
            "indices": np.array(chunk['indices']),
            "length": full_chunk.size(0),
            "domain": "math",
            "attention_mask": attention_mask.numpy(),
        })
    
    return Dataset.from_dict({
        "input_ids": [chunk["input_ids"] for chunk in processed_chunks],
        "indices": [chunk["indices"] for chunk in processed_chunks],
        "length": [chunk["length"] for chunk in processed_chunks],
        "domain": [chunk["domain"] for chunk in processed_chunks],
        "attention_mask": [chunk["attention_mask"] for chunk in processed_chunks],
    })

def load_openr1_math_dataset(dataset_name, tokenizer, training_max_length=None, max_samples=None):
    dataset = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")
    if max_samples is not None and max_samples > 0:
        dataset = dataset.select(range(max_samples))

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
    "load_openr1_math_dataset",
]
