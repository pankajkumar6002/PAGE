import json
import csv
from pathlib import Path
import argparse

OUTPUT_CSV = "all_results.csv"


def parse_run_name(run_name: str):
    """
    Parse run_name with one of the following forms:

      1) <method_name>-<chunk_size>-<budget_size>b-<max_seq_len>l-<somethingelse>
      2) <method_name>-<chunk_size>-<budget_size>b-<buffer_size>bf-<max_seq_len>l-<somethingelse>

    Example:
      "locret-3072-6000b-131072l-1nspl_42s"
      "locret-3072-6000b-512bf-131072l-1nspl_42s"
    """
    # Drop any suffix after an underscore (e.g., seed)
    core = run_name.split("_", 1)[0]

    parts = core.split("-")
    if len(parts) < 4:
        raise ValueError(f"Unexpected run_name format: {run_name}")

    method = parts[0]

    # chunk size
    try:
        chunk_size = int(parts[1])
    except ValueError:
        raise ValueError(f"Cannot parse chunk_size in run_name: {run_name}")

    # budget_size with trailing 'b'
    budget_part = parts[2]
    if not budget_part.endswith("b"):
        raise ValueError(f"Cannot find 'b' suffix for budget_size in run_name: {run_name}")
    try:
        budget_size = int(budget_part[:-1])
    except ValueError:
        raise ValueError(f"Cannot parse budget_size in run_name: {run_name}")

    buffer_size = -1  # default if no buffer_size present

    # Check if we have a buffer_size part (ends with 'bf')
    # Format with buffer: parts[3] = "<buffer_size>bf", parts[4] = "<max_seq_len>l"
    # Format without buffer: parts[3] = "<max_seq_len>l"
    if len(parts) >= 5 and parts[3].endswith("bf"):
        buffer_part = parts[3]
        try:
            buffer_size = int(buffer_part[:-2])  # strip "bf"
        except ValueError:
            raise ValueError(f"Cannot parse buffer_size in run_name: {run_name}")

        max_seq_part_index = 4
    else:
        # no buffer_size, stick with default -1
        max_seq_part_index = 3

    # We don't actually need max_seq_len for the CSV,
    # but we can sanity-check its format if we want.
    if len(parts) <= max_seq_part_index:
        raise ValueError(f"Missing max_seq_len part in run_name: {run_name}")
    max_seq_part = parts[max_seq_part_index]
    if not max_seq_part.endswith("l"):
        raise ValueError(f"Cannot find 'l' suffix for max_seq_len in run_name: {run_name}")
    # If you ever need it:
    # max_seq_len = int(max_seq_part[:-1])

    return method, chunk_size, budget_size, buffer_size


def gather_results(root: Path):
    rows = []

    # Walk through <root>/<model_name>/<dataset_name>/summary.txt
    for summary_path in root.rglob("summary.txt"):
        try:
            rel_parts = summary_path.relative_to(root).parts
        except ValueError:
            # Not under root, skip
            continue

        if len(rel_parts) < 3:
            # Expect at least model_name/dataset_name/summary.txt
            continue

        model_name = rel_parts[0]
        dataset_name = rel_parts[1]

        with summary_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: could not parse JSON in {summary_path}: {line}")
                    continue

                run_name = record.get("run_name")
                acc = record.get("acc")

                if run_name is None or acc is None:
                    print(f"Warning: missing run_name or acc in {summary_path}: {record}")
                    continue

                if 'greedy' not in run_name:
                    continue

                try:
                    method, chunk_size, budget_size, buffer_size = parse_run_name(run_name)
                except ValueError as e:
                    print(f"Warning: {e}")
                    continue

                rows.append({
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "run_name": run_name,
                    "method": method,
                    "budget_size": budget_size,
                    "buffer_size": buffer_size,
                    "chunk_size": chunk_size,
                    "acc": acc,
                })

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Gather experiment results from summary.txt files into a CSV."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="results",
        help="Root directory containing <model_name>/<dataset_name>/summary.txt (default: results)",
    )
    args = parser.parse_args()

    root_dir = Path(args.dir)

    if not root_dir.exists():
        raise SystemExit(f"Error: directory '{root_dir}' does not exist.")

    rows = gather_results(root_dir)

    # Optional: sort by dataset_name, then method, then model_name
    rows.sort(key=lambda r: (r["dataset_name"], r["method"], r["model_name"]))

    fieldnames = [
        "model_name",
        "dataset_name",
        "run_name",
        "method",
        "budget_size",
        "buffer_size",
        "chunk_size",
        "acc",
    ]

    output_path = root_dir / OUTPUT_CSV
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()

