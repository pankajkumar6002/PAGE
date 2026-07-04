import os
import json
from datasets import load_dataset
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True 

data = load_dataset("lmms-lab/LLaVA-NeXT-Data", split="train", cache_dir="data/llava_next_data")
image_folder = "data/llava_next_data/images"
os.makedirs(image_folder, exist_ok=True)

def process(example):
    result = {"id": example["id"], "conversations": example["conversations"], "image_path": ""}
    img = example.get("image", None)
    if img is not None:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        fname = f"{example['id']}.jpg"
        image_path = os.path.join(image_folder, fname)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        img.save(image_path)
        result["image_path"] = fname
    return result

processed = data.map(
    process,
    num_proc=128,  
    desc="Processing data",
    remove_columns=data.column_names
)

processed = processed.rename_column("image_path", "image")
processed = list(processed)
for rec in processed:
    if rec.get("image") == "":
        del rec["image"]

with open("data/llava_next_data/annotations.json", "w", encoding="utf-8") as f:
    json.dump(processed, f, indent=4, ensure_ascii=False)