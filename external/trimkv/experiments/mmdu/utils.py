import os 
from huggingface_hub import hf_hub_download
import zipfile 
import json 
from dataclasses import dataclass, field, fields
from typing import Union
import torch 
import numpy as np 
import random

def load_dataset(config):
    hf_hub_download(repo_id="laolao77/MMDU", filename="benchmark.json", local_dir=config.dataset_dir, repo_type="dataset")
    hf_hub_download(repo_id="laolao77/MMDU", filename="mmdu_pics.zip", local_dir=config.dataset_dir, repo_type="dataset")
    with zipfile.ZipFile(os.path.join(config.dataset_dir, "mmdu_pics.zip"), 'r') as zip_ref:
        zip_ref.extractall(config.dataset_dir)
    
    with open(os.path.join(config.dataset_dir, "benchmark.json"), 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print("Number of samples in benchmarks:", len(dataset))

    for item in dataset:
        img_paths = item["image"]
        new_img_paths = []
        for img_path in img_paths:
            new_img_paths.append(os.path.join(config.dataset_dir, "mmdu_pics", os.path.basename(img_path)))
        item["image"] = new_img_paths
    return dataset

def find_gap_between_turns(processor):
    # create a sample conversation with multiple turns
    sample_conversation = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"},
        {"role": "user", "content": "Can you tell me a joke?"},
    ]
    full_prompt = processor.apply_chat_template(sample_conversation, tokenizer=False, add_generation_prompt=True)
    last_turn = processor.apply_chat_template([sample_conversation[-1]], tokenizer=False, add_generation_prompt=True)

    start_idx = full_prompt.rfind(sample_conversation[-2]["content"]) + len(sample_conversation[-2]["content"])
    if full_prompt.endswith(last_turn):
        gap = full_prompt[start_idx: len(full_prompt) - len(last_turn)]
        # print so that it print \n instead of new line
        print("Identified gap between turns: ", repr(gap))
        return gap
    else:
        raise ValueError("Could not find the gap between turns.")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

