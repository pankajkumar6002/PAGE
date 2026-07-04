import os
import json
import base64
from io import BytesIO

from datasets import load_dataset
from PIL import Image, ImageFile

from dotenv import load_dotenv
load_dotenv()

DATASET_DIR = os.getenv("DATASET_DIR", "./data")
BASE_DIR = os.path.join(DATASET_DIR, "Processed-R1-Onevision")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
ANNOTATION_PATH = os.path.join(BASE_DIR, "R1-Onevision_annotation.json")
ImageFile.LOAD_TRUNCATED_IMAGES = True

def ensure_dirs():
    os.makedirs(IMAGES_DIR, exist_ok=True)

def decode_base64_to_pil(s: str) -> Image.Image:
    # Handles both plain base64 and data URLs by splitting on the first comma
    img_bytes = base64.b64decode(s.split(",", 1)[-1], validate=False)
    img = Image.open(BytesIO(img_bytes))
    return img.convert("RGB")


def normalize(conv):
    assert conv[0]["from"] == "human", "First message must be from human"
    if "<image>" not in conv[0]["value"]:
        conv[0]["value"] = "<image>\n" + conv[0]["value"]
    return conv

def main():
    ensure_dirs()

    print("Loading R1-Onevision dataset...")
    data = load_dataset("Fancy-MLLM/R1-Onevision", split="train")
    print(f"Saving to {BASE_DIR}...")
    # data = load_dataset("/storage2/hiu/secrets/datasets/R1-Onevision", split="train")
    # data = data.select(range(5000, 10000))  # For testing
    out = []
    for ex in data:
        img = decode_base64_to_pil(ex["image"])
        fname = f"{ex['id']}.jpg"
        img.save(os.path.join(IMAGES_DIR, fname), format="JPEG", quality=95)

        out.append({
            "id": ex["id"],
            "conversations": normalize(ex["conversations"]),
            "image": os.path.join("images", fname), 
        })

    with open(ANNOTATION_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4, ensure_ascii=False)

    print(f"Saved annotations to: {ANNOTATION_PATH}")
    print(f"Saved images to: {IMAGES_DIR}")
    print(f"Total records: {len(out)}, total images saved: {len(out)}")

if __name__ == "__main__":
    main()
