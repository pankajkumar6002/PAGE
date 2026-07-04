#!/usr/bin/env python3
import os
import json
import argparse
from collections import defaultdict
from longproc.longproc_data import load_longproc_data


def evaluate_jsonl(save_path: str, output_dir: str, name: str, dataset_dir: str = "data/"):
    """
    Evaluate a single .jsonl file using the provided logic and append results to summary.txt.
    """
    dataset, eval_func = None, None
    with open(save_path, "r") as f:
        eval_metrics = defaultdict(list)
        metric_names = set()
        for line in f.readlines():
            if not line.strip():
                continue
            example = json.loads(line)
            eid = example.get("question_id", "unknown_id")
            dataset_name = "_".join(eid.split("_")[:-1])
            question_id = int(eid.split("_")[-1])
            if dataset is None or eval_func is None:
                dataset, eval_func = load_longproc_data(dataset_name, dataset_dir)

            prediction = example.get("prediction", "")
            metrics, additional_info = eval_func(prediction, dataset[question_id])
            # Expecting {"question_id": ..., "metrics": {...}}
            eval_metrics[example["question_id"]].append(metrics)
            metric_names.update(metrics.keys())

        if len(eval_metrics) > 0:
            print(f"\n=== {name} ===")
            print(f"Average metrics over {len(eval_metrics)} examples:")
            print(f"Num samples per example: {[len(v) for v in eval_metrics.values()]}")

            # for each example, average over samples
            eval_metrics_mean = {}
            for k, v in eval_metrics.items():
                if not v:
                    continue
                mean_metrics = {}
                for metric in metric_names:
                    # Assumes every sample has the metric (matches the original snippet).
                    mean_metrics[metric] = sum([m[metric] for m in v]) / len(v)
                eval_metrics_mean[k] = mean_metrics

            avg_metrics = {
                "runname": name,
                "num_samples": len(eval_metrics),
            }
            for metric in metric_names:
                avg_metrics[metric] = (
                    sum([m[metric] for m in eval_metrics_mean.values()]) / len(eval_metrics_mean)
                )

            os.makedirs(output_dir, exist_ok=True)
            summary_path = os.path.join(output_dir, "summary.txt")
            with open(summary_path, "a") as out_f:
                print(json.dumps(avg_metrics, indent=4), flush=True)
                out_f.write(json.dumps(avg_metrics) + "\n")


def find_jsonl_files(root_dir: str):
    """Yield absolute paths to all .jsonl files under root_dir (recursively)."""
    for base, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(".jsonl"):
                yield os.path.join(base, fn)


def main():
    parser = argparse.ArgumentParser(
        description="Recursively evaluate all .jsonl files and append results to summary.txt."
    )
    parser.add_argument("folder", help="Root folder to search for .jsonl files.")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="Directory where summary.txt will be written (default: current directory).",
    )
    args = parser.parse_args()

    jsonl_paths = list(find_jsonl_files(args.folder))
    if not jsonl_paths:
        print("No .jsonl files found.")
        return

    for path in sorted(jsonl_paths):
        # Derive a readable run name from the path (relative to the root folder).
        try:
            rel = os.path.relpath(path, args.folder)
        except Exception:
            rel = os.path.basename(path)
        name = os.path.splitext(rel)[0]  # strip .jsonl
        evaluate_jsonl(save_path=path, output_dir=args.output_dir, name=name)


if __name__ == "__main__":
    main()
