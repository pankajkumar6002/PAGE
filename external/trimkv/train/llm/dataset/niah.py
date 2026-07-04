
import numpy as np

from datasets import load_dataset, Dataset, load_from_disk
from functools import partial
from tqdm import tqdm 
import os
import json 
import re
from nltk.tokenize import sent_tokenize
import numpy as np
import random
import wonderwords

PROMPT_TEMPLATE = "<|im_start|>user\nSome special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards. \n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nThe special magic {type_needle_v} for {query} mentioned in the provided text are{ground_truth}<|im_end|>"
NEEDLE_TEMPLATE = "One of the special magic {type_needle_v} for {key} is: {value}."
ANSWER_PREFIX = " The special magic {type_needle_v} for {query} mentioned in the provided text are"
DEPTHS = np.round(np.linspace(2, 98, num=100, endpoint=True, dtype=float), 3).tolist()
nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
# verbs = wonderwords.random_word._get_words_from_text_file("verblist.txt")
words = [f"{adj}-{noun}" for adj in adjs for noun in nouns]
WORDS = list(set(words))
TYPE_NEEDLE_V = "words"
   

def read_json(data_path: str):
    data = []
    with open(data_path, 'r', encoding='utf-8') as f: 
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            data.append(item)
    return data

def sandwich_needles_into_essay(record, sentences):
    all_needles = record['all_needles']
    insert_positions = sorted([int(len(sentences) * depth / 100) for depth in record['sampled_depths']])
    insert_positions = [0] + insert_positions + [len(sentences)]
    
    context = []
    for i in range(len(insert_positions) - 1):
        context.extend(sentences[insert_positions[i]:insert_positions[i+1]])
        if i < len(all_needles):
            context.append(all_needles[i])
    context = " ".join(context)
    return context

input_essay = {}
def fill_record_with_context(record, tokenized_essay, max_seq_length, tokenizer):

    if "inputs" not in record:
        record["inputs"] = {}
        
    remaining_tokens = max_seq_length - record['contextless_input_length'] - 128 # for safety
    # print(f"Remaining tokens: {remaining_tokens}, input")
    if remaining_tokens <= 0:
        base_context = " ".join(neddle for neddle in record['needle']) 
        input_text = PROMPT_TEMPLATE.format(
            type_needle_v=TYPE_NEEDLE_V,
            context=base_context,
            query=record['query'],
            ground_truth=record['response']
        )

    else:
        will_tokenize = True
        for k in input_essay.keys():
            if abs(k - remaining_tokens) <= 50:
                sentences = input_essay[k]
                will_tokenize = False 
                break
        if will_tokenize:
            decoded_essay = tokenizer.decode(tokenized_essay[:remaining_tokens])
            sentences = sent_tokenize(decoded_essay)[:-1]
            input_essay[remaining_tokens] = sentences

        
        context = sandwich_needles_into_essay(record, sentences)
        input_text = PROMPT_TEMPLATE.format(
            type_needle_v=TYPE_NEEDLE_V,
            context=context,
            query=record['query'],
            ground_truth=record['response']
        )
        
    record["inputs"][max_seq_length] = {
        "input": input_text,
        "input_length": len(tokenizer(input_text)['input_ids']),
    }
    return record 

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


def load_synthetic_niah_dataset(dataset_name, tokenizer, training_max_length=None, max_samples=None):
    # dataset = load_dataset("cerebras/Synth-Long-SFT32K", split="train_convqa_raft+train_convqa_raft_syntactic+train_narrativeqa_aug_32k+train_rag_tge_raft")
    # records = read_json("/datas/wama/new_folder/reclone3/trimkv-dev/experiments/ruler/data/niah/500samples_k24words_v3uuids_q3train.json")
    # dataset = load_dataset("JunHill/Qwen3_Niah_v1")
    dataset = load_dataset("JunHill/niah_k8v4q4")['train']
    essay_path = "../../experiments/ruler/data/essay/PaulGrahamEssays.json"
    essay = json.load(open(essay_path))['text']
    essay = re.sub(r'\s+', " ", essay)
    tokenized_essay = tokenizer(essay)['input_ids']
    records = dataset.to_list()
    # records = random.sample(records, 3500)
    data = []
    for record in tqdm(records):
        for seq_len in [0, 4000, 8000,16000, 32000, 64000, 128000, 198000]:
            if seq_len >= training_max_length:
                continue
            record = fill_record_with_context(record, tokenized_essay, max_seq_length=seq_len, tokenizer=tokenizer)
            if seq_len in record["inputs"]:
                data.append({'text': record["inputs"][seq_len]["input"]})
    
    if len(data) == 0:
        raise ValueError()
    print(len(data))
    dataset = Dataset.from_list(data)
    print(f" dataset size: {len(dataset)}")
    if max_samples is not None and max_samples > 0:
        dataset = dataset.select(range(max_samples))


    dataset = dataset.map(
        partial(tokenize_fn, tokenizer),
        batched=False,
        num_proc=8,
    )

    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col != "input_ids"]
    )

    dataset = dataset.filter(
        lambda x: len(x["input_ids"]) < training_max_length and len(x["input_ids"]) > 512,
        num_proc=8,
    )
    print(f"Filtered niah dataset dataset size: {len(dataset)}")

    return dataset

__all__ = [
    "load_synthetic_niah_dataset",
]