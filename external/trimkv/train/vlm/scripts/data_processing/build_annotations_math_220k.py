import argparse
import json
import os
import random
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace
from tqdm import tqdm
from functools import partial

from dotenv import load_dotenv
from dataset.configs import get_dataset_configs
from dataset import data_processor
from transformers import AutoProcessor
from transformers.models.qwen3_vl import Qwen3VLProcessor  # noqa: F401 (available via AutoProcessor)
from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor  # noqa: F401 (available via AutoProcessor)

from dataset.qwen_utils import (
    preprocess_qwen_visual,
    update_processor_pixels,
)
from dataset.llava_utils import (
    preprocess_llava_visual,
)

# ----------------------------- Environment & Globals -----------------------------

load_dotenv()

# Keep CPU-threaded libs from over-subscribing when we also use multi-process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# In multi-process, tokenizer threads often hurt more than help.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_MAP = {
    "qwen2_5vl": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen3vl": "Qwen/Qwen3-VL-8B-Thinking",
    "llava1_5": "llava-hf/llava-1.5-7b-hf",

}

# Per-process globals (initialized once in each worker)
_worker_processor = None
_worker_data_path = None


# ----------------------------- I/O Helpers -----------------------------

def read_jsonl(path: str):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def write_json(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_jsonl(items, path: str):
    """Write list of dicts to JSONL file."""
    with open(path, "w") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _jsonl_line_offsets(path: str):
    """
    Return (offsets, file_size) where offsets[i] is the starting byte of line i.
    """
    offsets = []
    pos = 0
    with open(path, "rb") as f:
        for line in f:
            offsets.append(pos)
            pos += len(line)
    return offsets, pos


# ----------------------------- Core per-sample logic -----------------------------

def compute_sample_seq_length(preprocess_fn, sample) -> int:
    """
    Compute sequence length (tokens) for a single annotation sample.
    """
    data = preprocess_fn([sample])

    input_ids = data["input_ids"]
    return len(input_ids[0]) if isinstance(input_ids, list) else int(input_ids.shape[1])


# ----------------------------- Utilities for nested annotations -----------------------------

def _attach_data_path(annotations, data_path):
    """
    Ensure every sample dict has data_path set, even inside nested lists.
    """
    for ann in annotations:
        if isinstance(ann, list):
            for sub in ann:
                sub["data_path"] = data_path
        else:
            ann["data_path"] = data_path


def _flatten_annotations(annotations):
    """
    Flattens annotations into a list of sample dicts while preserving references
    so we can assign back `seqlen` after computation.
    """
    flat = []
    for ann in annotations:
        if isinstance(ann, list):
            flat.extend(ann)
        else:
            flat.append(ann)
    return flat


def _split_indices(n: int, k: int):
    """
    Split [0, n) into k contiguous ranges as evenly as possible.
    Returns a list of (start, end) tuples where end is exclusive.
    """
    if k <= 0:
        return [(0, n)]
    k = min(k, max(1, n))  # don't create more chunks than items
    base, rem = divmod(n, k)
    ranges = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        end = start + size
        if start < end:
            ranges.append((start, end))
        start = end
    return ranges


# ----------------------------- Worker init & workers -----------------------------

def _init_worker(model_key: str, args_dict: dict, data_path: str):
    """
    Initializer that runs ONCE per worker process to build and cache the processor.
    Also sets conservative thread env to avoid oversubscription.
    """
    import os as _os
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    _os.environ.setdefault("MKL_NUM_THREADS", "1")
    _os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    global _worker_processor, _worker_data_path
    _worker_data_path = data_path
    _worker_processor = AutoProcessor.from_pretrained(MODEL_MAP[model_key], add_eos_token=True,)
    # If you need pixel/vision-specific updates:
    # from types import SimpleNamespace
    # _worker_processor = data_processor.update_processor_pixels(_worker_processor, SimpleNamespace(**args_dict))

    with open("chat_template/templates.json", "r") as f:
        chat_template = json.load(f)

    if model_key == 'llava1_5':
        _worker_processor.tokenizer.chat_template = chat_template["llava-hf/llava-1.5-7b-hf"]
        print("Using modified chat template for Llava 1.5")

    preprocess_fn = None
    if model_key.startswith("qwen"):
        # update_processor_pixels(
        #     _worker_processor,
        #     SimpleNamespace(**args_dict),
        # )
        preprocess_fn = partial(
            preprocess_qwen_visual,
            processor=_worker_processor,
        )
    elif args_dict["model"].startswith("llava"):
        preprocess_fn = partial(
            preprocess_llava_visual,
            processor=_worker_processor,
        )
    else:
        raise ValueError(f"Unknown model key: {model_key}")
    _worker_processor = preprocess_fn



def _worker_compute_jsonl_range(jsonl_path: str, byte_start: int, byte_end: int):
    """
    Worker for JSONL input: reads [byte_start, byte_end) directly from disk, computes seqlens,
    and returns a list of ints. Avoids pickling big dicts.
    """
    import json as _json
    global _worker_processor, _worker_data_path

    out = []
    with open(jsonl_path, "rb") as f:
        f.seek(byte_start)
        while f.tell() < byte_end:
            line = f.readline()
            if not line:
                break
            try:
                sample = _json.loads(line)
                sample["data_path"] = _worker_data_path
                slen = compute_sample_seq_length(_worker_processor, sample)
                out.append(int(slen))
            except Exception:
                out.append(-1)
    return out


# ----------------------------- Single-process path (optional) -----------------------------

def compute_seqlen_single(cfg, model_key, args):
    """
    Single-process baseline. Useful for debugging and for cases where multiproc overhead dominates.
    """
    ann_path = cfg["annotation_path"]
    file_format = ann_path.split(".")[-1].lower()

    # Load annotations fully
    if file_format == "jsonl":
        annotations = read_jsonl(ann_path)
    else:
        annotations = json.load(open(ann_path, "r"))

    sampling_rate = cfg.get("sampling_rate", 1.0)
    if sampling_rate < 1.0:
        print(f"Total annotations before sampling: {len(_flatten_annotations(annotations)) if file_format!='jsonl' else len(annotations)}")
        if file_format == "jsonl":
            k = int(len(annotations) * sampling_rate)
            annotations = random.sample(annotations, k)
        else:
            flat = _flatten_annotations(annotations)
            k = int(len(flat) * sampling_rate)
            annotations = random.sample(flat, k)
        print(f"Sampled down to {len(annotations)} annotations.")

    _attach_data_path(annotations, cfg["data_path"])
    flat_samples = _flatten_annotations(annotations)

    _init_worker(model_key, vars(args), cfg["data_path"])
    global _worker_processor
    processor = _worker_processor

    failed = 0
    for sample in tqdm(flat_samples, desc="Computing seqlen", dynamic_ncols=True):
        # try:
        slen = compute_sample_seq_length(processor, sample)
        sample["seqlen"] = int(slen)
        # except Exception as e:
        #     sample["seqlen"] = -1
        #     failed += 1

    seqlens = [s["seqlen"] for s in flat_samples if "seqlen" in s]
    if seqlens:
        print(
            f"Max seqlen: {max(seqlens)}, "
            f"Min seqlen: {min(seqlens)}, "
            f"Avg seqlen: {sum(seqlens)/len(seqlens):.2f} "
            f"(failed: {failed})"
        )
    else:
        print("Warning: all samples failed to compute seqlen.")
    return annotations


# ----------------------------- Parallel driver (ALWAYS JSONL) -----------------------------

def compute_seqlen_parallel(cfg, model_key, args):
    """
    Parallel computation that ALWAYS operates on a JSONL file via byte ranges.
    If the source is a JSON array or sampling is requested, we first materialize a
    temporary JSONL containing exactly the items to process (and in the right order).
    """
    ann_path = cfg["annotation_path"]
    file_format = ann_path.split(".")[-1].lower()
    sampling_rate = cfg.get("sampling_rate", 1.0)

    temp_jsonl_path = None
    annotations_out = None  # The list we'll attach seqlens to and return.

    if file_format == "jsonl" and sampling_rate >= 1.0:
        # Fast path: use the original JSONL directly.
        work_jsonl_path = ann_path
        # We'll load the full annotations AFTER computing seqlens to attach results.
        load_after = True
    else:
        # We need to create a working JSONL (either source is .json OR sampling < 1.0).
        load_after = False
        tmp = tempfile.NamedTemporaryFile(prefix="seqlen_work_", suffix=".jsonl", delete=False)
        temp_jsonl_path = tmp.name
        tmp.close()

        if file_format == "json":
            # Load JSON, optional sampling at item level, then write to JSONL.
            annotations = json.load(open(ann_path, "r"))

            if sampling_rate < 1.0:
                flat = _flatten_annotations(annotations)
                print(f"Total annotations before sampling: {len(flat)}")
                k = max(1, int(len(flat) * sampling_rate))
                sampled = random.sample(flat, k)
                print(f"Sampled down to {len(sampled)} annotations.")
                _write_jsonl(sampled, temp_jsonl_path)
                # This sampled list becomes the output list (flat)
                annotations_out = sampled
            else:
                # No sampling: keep original nested structure for output,
                # but write a flattened JSONL for fast parallel processing.
                flat = _flatten_annotations(annotations)
                _write_jsonl(flat, temp_jsonl_path)
                annotations_out = annotations  # preserve original structure

        else:
            # Source is JSONL but sampling is requested.
            assert file_format == "jsonl" and sampling_rate < 1.0
            # Count lines and sample indices
            offsets, _ = _jsonl_line_offsets(ann_path)
            n = len(offsets)
            print(f"Total annotations before sampling: {n}")
            k = max(1, int(n * sampling_rate))
            sel = sorted(random.sample(range(n), k))
            print(f"Sampled down to {k} annotations.")

            selected_items = []
            with open(ann_path, "r") as fin, open(temp_jsonl_path, "w") as fout:
                for i, line in enumerate(fin):
                    if not sel:
                        break
                    if i == sel[0]:
                        fout.write(line)
                        selected_items.append(json.loads(line))
                        sel.pop(0)

            # This sampled list becomes the output list (flat)
            annotations_out = selected_items

        work_jsonl_path = temp_jsonl_path

    # Build byte ranges on the working JSONL
    offsets, file_size = _jsonl_line_offsets(work_jsonl_path)
    n = len(offsets)
    if n == 0:
        print("No annotations found.")
        if temp_jsonl_path and os.path.exists(temp_jsonl_path):
            os.remove(temp_jsonl_path)
        return [] if annotations_out is None else annotations_out

    num_workers = max(1, int(args.num_workers))
    index_ranges = _split_indices(n, num_workers)
    jobs = []
    for (i0, i1) in index_ranges:
        if i0 >= i1:
            continue
        b0 = offsets[i0]
        b1 = offsets[i1] if i1 < n else file_size
        jobs.append((i0, i1, b0, b1))

    init_args = (model_key, vars(args), cfg["data_path"])
    seqlens_all = [None] * n

    with ProcessPoolExecutor(
        max_workers=len(jobs),
        initializer=_init_worker,
        initargs=init_args
    ) as ex:
        pending = {}
        for (i0, i1, b0, b1) in jobs:
            fut = ex.submit(_worker_compute_jsonl_range, work_jsonl_path, b0, b1)
            pending[fut] = (i0, i1)

        with tqdm(total=n, desc="Computing seqlen", dynamic_ncols=True) as pbar:
            for fut in as_completed(pending):
                i0, i1 = pending[fut]
                try:
                    results = fut.result()
                except Exception:
                    results = [-1] * (i1 - i0)
                for idx, slen in zip(range(i0, i1), results):
                    seqlens_all[idx] = int(slen)
                pbar.update(i1 - i0)

    # Prepare output annotations and attach seqlens
    if load_after:
        # We used the original JSONL directly; now load it to attach results.
        annotations = read_jsonl(ann_path)
        _attach_data_path(annotations, cfg["data_path"])
        for i, ann in enumerate(annotations):
            ann["seqlen"] = int(seqlens_all[i]) if seqlens_all[i] is not None else -1
        annotations_out = annotations
    else:
        # We already have annotations_out (either sampled JSON/JSONL or original JSON structure).
        # Map seqlens back in order.
        if isinstance(annotations_out, list) and annotations_out and isinstance(annotations_out[0], list):
            # Nested structure preserved: assign sequentially
            idx = 0
            for group in annotations_out:
                for item in group:
                    item["data_path"] = cfg["data_path"]
                    item["seqlen"] = int(seqlens_all[idx]) if seqlens_all[idx] is not None else -1
                    idx += 1
        else:
            # Flat list
            _attach_data_path(annotations_out, cfg["data_path"])
            for i, ann in enumerate(_flatten_annotations(annotations_out)):
                ann["seqlen"] = int(seqlens_all[i]) if seqlens_all[i] is not None else -1

    # Stats
    ok = [s for s in seqlens_all if s is not None and s != -1]
    if ok:
        print(
            f"Max seqlen: {max(ok)}, "
            f"Min seqlen: {min(ok)}, "
            f"Avg seqlen: {sum(ok)/len(ok):.2f} "
            f"(failed: {n - len(ok)})"
        )
    else:
        print("Warning: all samples failed to compute seqlen.")

    # Cleanup temp JSONL if created
    if temp_jsonl_path and os.path.exists(temp_jsonl_path):
        try:
            os.remove(temp_jsonl_path)
        except OSError:
            pass

    return annotations_out

def build_annotation_math_220k(data_config):
    from datasets import load_dataset
    # Load and preprocess the dataset
    dataset = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")
    annotations = []
    
    for item in dataset:
        annotation = {
            "id": item["uuid"],
            "conversations": [
                {
                    "from": "human",
                    "value": item["problem"]
                },
                {
                    "from": "gpt",
                    "value": item["messages"][1]["content"]
                }
            ]

        }
        annotations.append(annotation)
    
    output_path = data_config['annotation_path']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_json(output_path, annotations)
    print(f"Wrote: {output_path}")

# ----------------------------- CLI -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute token sequence lengths for annotations.")
    parser.add_argument("--model", type=str, default="qwen3vl", help="Name of the model to use.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of CPU workers to use.")
    parser.add_argument("--mode", type=str, choices=["single", "parallel"], default="single",
                        help="Use single-process or parallel mode.")
    parser.add_argument("--min_pixels", type=int, default=28 * 28 * 32, help="Minimum pixels for image resizing.")
    parser.add_argument("--max_pixels", type=int, default=2048 * 28 * 28, help="Maximum pixels for image resizing.")
    parser.add_argument("--video_min_frames", type=int, default=4, help="Minimum frames for video sampling.")
    parser.add_argument("--video_max_frames", type=int, default=8, help="Maximum frames for video sampling.")
    parser.add_argument("--video_min_pixels", type=int, default=4 * 32 * 28 * 28, help="Minimum frame pixels for video resizing.")
    parser.add_argument("--video_max_pixels", type=int, default=4 * 2048 * 28 * 28, help="Maximum frame pixels for video resizing.")
    parser.add_argument("--video_fps", type=float, default=2, help="FPS for video sampling.")
    args = parser.parse_args()

    dataset_dir = os.getenv("DATASET_DIR", ".")
    cfg = get_dataset_configs(["math_220k"], dataset_dir=dataset_dir)[0]
    build_annotation_math_220k(cfg)
    
    if args.mode == "single":
        annotations = compute_seqlen_single(cfg, args.model, args)
    else:
        annotations = compute_seqlen_parallel(cfg, args.model, args)

    # Write alongside the original, preserving extension (.json or .jsonl)
    stem, ext = os.path.splitext(cfg["annotation_path"])
    out_path = f"{stem}_{args.model}_seqlen{ext}"
    write_json(out_path, annotations)
    print(f"Wrote: {out_path}")
    
    # output_path = cfg['annotation_path']
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # write_json(output_path, annotations)
    # print(f"Wrote: {output_path}")
