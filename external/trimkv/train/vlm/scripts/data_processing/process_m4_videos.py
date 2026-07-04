import os
import json
import argparse
from tqdm import tqdm

def main(json_file: str, video_path: str):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        file_list = [item['video'] for item in data]

    name_to_path = {}
    mp4_name = set()
    for root, _, files in os.walk(video_path):
        for f in files:
            if f.endswith(".mp4"):
                rel_path = os.path.join(root, f)
                base = os.path.splitext(f)[0]
                mp4_name.add(base)
                name_to_path[base] = rel_path.replace(video_path+"/", "")

    file_list = list(set(file_list))
    found, missing = [], []

    for f in tqdm(file_list):
        if f in mp4_name:
            found.append(f)
        else:
            missing.append(f)

    print(f"✅ Found {len(found)} files")
    print(f"❌ Missing {len(missing)} files")
    new_data = []
    for item in data:
        if item["video"] in found:
            for conv in item.get("conversations", []):
                if "<image>" in conv["value"]:
                    conv["value"] = conv["value"].replace("<image>", "<video>")
            item["video"] = name_to_path[item["video"]]
            new_data.append(item)

    new_file = json_file.replace(".json", "_fixed.json")
    with open(new_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)
    print(f"Saved to {new_file}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fix video paths in JSON.")
    ap.add_argument("--json-file", required=True, help="Path to JSON file (e.g., m4_instruct_video.json)")
    ap.add_argument("--video-path", required=True, help="Root folder containing .mp4 files (e.g., raw_videos)")
    args = ap.parse_args()
    main(args.json_file, args.video_path)