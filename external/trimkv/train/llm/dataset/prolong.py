import os
import torch

from streaming import StreamingDataset

from .utils import SafeStream

DATANAME_MAP = {
    "prolong_64k": "long-context-65536",
    "prolong_512k": "long-context-524288",
    "prolong_ultrachat": "prolong-ultrachat-64K",
}


def load_prolong_64k_dataset(dataset_name, dataset_path, tokenizer, training_max_length=None, proportion=1.0):
    domains = (
        "thestackv1_concat_by_repo-65536@0.3",
        "book-65536@0.3",
        "fineweb-edu@0.1",
        "fineweb-2023-50@0.1",
        "stackexchange@0.04",
        "dolmawiki@0.04",
        "tuluv2@0.03",
        "arxiv@0.03",
        "openwebmath@0.03",
        "textbooks@0.03",
    )
    sum_proportion = sum(float(domain.split("@")[-1]) if "@" in domain else 1.0 for domain in domains)
    streams = []
    dataset_path = os.path.join(dataset_path, DATANAME_MAP[dataset_name])
    paths = [os.path.join(dataset_path, domain) for domain in domains]

    for path in paths:
        path, p = path.split("@", 1)
        p = float(p) / sum_proportion * proportion
        print(f"Loading dataset from {path} with proportion {p}")
        streams.append(SafeStream(remote=path, local=path, proportion=p))

    return streams

def load_prolong_512k_dataset(dataset_name, dataset_path, tokenizer, training_max_length=None, proportion=1.0):
    domains=(
        "thestackv1_concat_by_repo-524288@0.15",
        "thestackv1_concat_by_repo-65536@0.15",
        "book-524288@0.05",
        "book-65536@0.25",
        "fineweb-edu@0.1",
        "fineweb-2023-50@0.1",
        "stackexchange@0.04",
        "dolmawiki@0.04",
        "tuluv2@0.03",
        "arxiv@0.03",
        "openwebmath@0.03",
        "textbooks@0.03",
    )
    sum_proportion = sum(float(domain.split("@")[-1]) if "@" in domain else 1.0 for domain in domains)
    streams = []
    dataset_path = os.path.join(dataset_path, DATANAME_MAP[dataset_name])
    paths = [os.path.join(dataset_path, domain) for domain in domains]

    for path in paths:
        path, p = path.split("@", 1)
        p = float(p) / sum_proportion * proportion
        print(f"Loading dataset from {path} with proportion {p}")
        streams.append(SafeStream(remote=path, local=path, proportion=p))

    return streams

def load_prolong_ultrachat_dataset(dataset_name, dataset_path, tokenizer, training_max_length=None, proportion=1.0):
    dataset_path = os.path.join(dataset_path, DATANAME_MAP[dataset_name])
    stream = SafeStream(remote=dataset_path, local=dataset_path, proportion=proportion)

    return [stream]


__all__ = [
    "load_prolong_64k_dataset",
    "load_prolong_512k_dataset",
    "load_prolong_ultrachat_dataset",
]
