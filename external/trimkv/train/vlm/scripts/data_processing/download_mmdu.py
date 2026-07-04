import os 
from huggingface_hub import hf_hub_download
import zipfile 
import json 
from dataclasses import dataclass, field, fields
from typing import Union
import torch 
import numpy as np 
import random
from dotenv import load_dotenv


label_map = {
    "user": "human",
    "assistant": "gpt",
}
def load_dataset(dataset_dir):
    hf_hub_download(repo_id="laolao77/MMDU", filename="mmdu-45k.json", local_dir=dataset_dir, repo_type="dataset")
    hf_hub_download(repo_id="laolao77/MMDU", filename="mmdu-45k_pics.zip", local_dir=dataset_dir, repo_type="dataset")
    with zipfile.ZipFile(os.path.join(dataset_dir, "mmdu-45k_pics.zip"), 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    
    with open(os.path.join(dataset_dir, "mmdu-45k.json"), 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print("Number of samples in benchmarks:", len(dataset))

    for item in dataset:
        img_paths = item["image"]
        new_img_paths = []
        for img_path in img_paths:
            new_img_paths.append(os.path.join("mmdu-45k_pics", os.path.basename(img_path)))
            # new_img_paths.append(os.path.join(dataset_dir, "mmdu_pics", os.path.basename(img_path)))
        item["image"] = new_img_paths

        conversation = item["conversations"]
        new_conversation = []
        for turn in conversation:

            new_turn = {
                "from": label_map[turn["from"]],
                "value": turn["value"].replace("<ImageHere>", "<image>")
            }
            new_conversation.append(new_turn)
        item["conversations"] = new_conversation.copy()
    return dataset


if __name__ == "__main__":

    load_dotenv()
    dataset_dir = os.path.join(os.getenv("DATASET_DIR"), "mmdu")
    
    # config = Config()
    dataset = load_dataset(dataset_dir=dataset_dir)
    with open(os.path.join(dataset_dir, "mmdu_processed.json"), 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    
    with open(os.path.join(dataset_dir, "mmdu_processed_mini.json"), 'w', encoding='utf-8') as f:
        json.dump(dataset[:5], f, indent=4, ensure_ascii=False)
    # breakpoint()