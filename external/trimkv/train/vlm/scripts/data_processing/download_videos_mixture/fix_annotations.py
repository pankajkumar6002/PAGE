import json 
import os
import argparse

parser = argparse.ArgumentParser(description="Filter records by existing image/video paths and write *_fixed.json")
parser.add_argument(
    "-j", "--json-path",
    required=True,
    help="Path to the input annotations JSON",
)
parser.add_argument(
    "-rim", "--replace-image-with-video",
    action="store_true",
)
parser.add_argument(
    "-dm", "--drop-metadata",
    action="store_true",
)

args = parser.parse_args()

input_json_path = args.json_path

with open(input_json_path, "r") as f:
    annotations = json.load(f)


def _path_or_paths_exist(value, root_dir: str) -> bool:
    if isinstance(value, str):
        return os.path.exists(os.path.join(root_dir, value))
    if isinstance(value, list):
        return all(os.path.exists(os.path.join(root_dir, p)) for p in value)
    return False

def fix_record(record: dict) -> dict:
    conversations = record["conversations"]
    for conv in conversations:
        if "<image>" in conv["value"]:
            conv["value"] = conv["value"].replace("<image>", "<video>")
    record["conversations"] = conversations
    return record

def has_existing_media_paths(record: dict, root_dir: str) -> bool:
    has_media_key = False
    all_exist = True
    is_text_only = False
    if "image" in record:
        has_media_key = True
        all_exist = all_exist and _path_or_paths_exist(record["image"], root_dir)
    if "video" in record:
        has_media_key = True
        all_exist = all_exist and _path_or_paths_exist(record["video"], root_dir)
    if "image" not in record and "video" not in record:
        is_text_only = True
    return (has_media_key and all_exist) or is_text_only



filtered_records = []
missing_records = []
total_records = len(annotations)
kept_count = 0
skipped_count = 0
annotations_dir = os.path.dirname(input_json_path)

for record in annotations:
    if args.replace_image_with_video:
        record = fix_record(record)
    if args.drop_metadata and "metadata" in record:
        record.pop("metadata")
    if has_existing_media_paths(record, annotations_dir):
        filtered_records.append(record)
        kept_count += 1
    else:
        missing_records.append(record)
        skipped_count += 1

base, ext = os.path.splitext(input_json_path)
kept_json_path = f"{base}_fixed{ext}"
missing_json_path = f"{base}_missing_{skipped_count}{ext}"

if kept_count == 0:
    raise ValueError("No records kept, something is wrong! check the path, json needs to be in images folder! Please call Hieu to fix")

with open(kept_json_path, "w", encoding="utf-8") as f:
    json.dump(filtered_records, f, indent=4, ensure_ascii=False)

with open(missing_json_path, "w", encoding="utf-8") as f:
    json.dump(missing_records, f, indent=4, ensure_ascii=False)

print(f"Input JSON: {input_json_path}")
print(f"Kept JSON: {kept_json_path}")
print(f"Missing JSON: {missing_json_path}")
print(f"Total records: {total_records}, kept: {kept_count}, skipped: {skipped_count}")
