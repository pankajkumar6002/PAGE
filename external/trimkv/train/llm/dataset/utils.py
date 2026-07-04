# Source: https://github.com/princeton-nlp/ProLong/blob/main/training/dataset.py

import os
import torch

from streaming import StreamingDataset, Stream
from dataclasses import dataclass
import logging

from itertools import islice
import itertools

import transformers
from typing import Dict, Any, List, Tuple
from collections.abc import Sequence
from collections.abc import Iterator
from torch.utils.data import Dataset
from collections import deque
import numpy as np


IGNORE_INDEX = -100


class _Fenwick:
    def __init__(self, n: int):
        self.n = n
        self.bit = [0]*(n+1)
    def add(self, i: int, delta: int) -> None:
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i
    def sum(self, i: int) -> int:
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s
    def range_sum(self, l: int, r: int) -> int:
        if r < l: return 0
        return self.sum(r) - self.sum(l-1)
    def find_by_prefix(self, target: int) -> int:
        idx, bitmask = 0, 1 << (self.n.bit_length()-1)
        while bitmask:
            t = idx + bitmask
            if t <= self.n and self.bit[t] < target:
                target -= self.bit[t]
                idx = t
            bitmask >>= 1
        return idx + 1

def binpack_bfd(items: List[int], capacity: int) -> List[List[int]]:
    """
    Best-Fit Decreasing bin packing (integer sizes), returning original indices.

    Args:
        items: positive integers (each <= capacity)
        capacity: positive integer

    Returns:
        bins_idx: List of bins; each bin is a list of 0-based original indices.
    """
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if any(x <= 0 or int(x) != x for x in items):
        raise ValueError("All item sizes must be positive integers.")
    if np.max(items) > capacity:
        raise ValueError(f"Found item larger than capacity. capacity={capacity}, max_item={np.max(items)}")

    # sort by size desc (stable -> preserves index order among ties)
    seq = sorted([(int(w), i) for i, w in enumerate(items)], key=lambda t: -t[0])

    buckets = [deque() for _ in range(capacity+1)]  # bins by exact residual r
    ft = _Fenwick(capacity)                          # counts of bins per residual r (1..capacity)

    bins_idx: List[List[int]] = []
    residuals: List[int] = []

    for w, i in seq:
        if ft.range_sum(w, capacity) == 0:
            # open a new bin
            bins_idx.append([i])
            r = capacity - w
            residuals.append(r)
            if r > 0:
                buckets[r].append(len(bins_idx)-1)
                ft.add(r, +1)
        else:
            # tightest fit: smallest residual r >= w that exists
            target = ft.sum(w-1) + 1
            r = ft.find_by_prefix(target)
            b = buckets[r].pop()
            ft.add(r, -1)

            bins_idx[b].append(i)
            new_r = r - w
            residuals[b] = new_r
            if new_r > 0:
                buckets[new_r].append(b)
                ft.add(new_r, +1)

    return bins_idx


class SafeStream(Stream):
    """Safe if multiple processes try to decompress the same shard."""

    def _decompress_shard_part(self, zip_info, zip_filename, raw_filename, compression):
        unique_extension = "." + str(os.getenv("SLURM_JOB_ID", "local")) + "-" + str(os.getpid())
        super()._decompress_shard_part(zip_info, zip_filename, raw_filename + unique_extension, compression)
        os.rename(raw_filename + unique_extension, raw_filename)


class SortByLengthDataset(StreamingDataset):
    def __init__(
        self,
        *args,
        sort_by_length_size=1,
        single_seq: bool = False,
        per_device_max_tokens: int = 4294967296,
        apply_instruct_masks: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sort_by_length_size = sort_by_length_size
        self.single_seq = single_seq
        self.per_device_max_tokens = per_device_max_tokens
        self.apply_instruct_masks = apply_instruct_masks

    def _negative_item_cost(self, item):
        if "indices" in item:
            return -sum(
                (end - start)**2 for start, end in item["indices"]
            )
        elif "length" in item:
            return -item["length"]**2
        else:
            return -len(item["input_ids"])**2

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self.sort_by_length_size <= 1:
            yield from super().__iter__()
        else:
            iterator = super().__iter__()
            while True:
                block = list(islice(iterator, self.sort_by_length_size))
                if not block:
                    return

                yield from sorted(block, key=self._negative_item_cost)

class DataCollator:
    def __init__(
        self, 
        tokenizer,
        single_seg: bool = False,
        per_device_max_tokens: int = 4294967296,
        apply_instruct_masks: bool = False,
    ):
        self.tokenizer = tokenizer
        self.single_seg = single_seg
        self.per_device_max_tokens = per_device_max_tokens
        self.apply_instruct_masks = apply_instruct_masks

    @torch.no_grad()
    def __call__(self, features):
        input_ids = []
        labels = []
        seq_lengths = []

        available_tokens = self.per_device_max_tokens
        for item in features:
            apply_instruct_masks = self.apply_instruct_masks and ("mask" in item)
            indices = item["indices"] if "indices" in item else [(0, len(item["input_ids"]))]
            if self.single_seq:
                indices = [(0, len(item["input_ids"]))]

            label_seq = torch.tensor(item["input_ids"], dtype=torch.long)

            for a, b in indices:
                b = a + min(b - a, available_tokens)
                if b - a > 1:
                    input_seq = torch.tensor(item["input_ids"][a:b], dtype=torch.long)
                    input_ids.append(input_seq)

                    _label = label_seq[a:b]
                    _label[0] = -100 # Don't predict the first token
                    if apply_instruct_masks:
                        # Read the `mask` field and set the corresponding labels to -100
                        mask = torch.tensor(item["mask"][a:b], dtype=torch.long)
                        _label[mask == 0] = -100
                    labels.append(_label)

                    seq_lengths.append(b - a)
                    available_tokens -= b - a
                elif available_tokens <= 0:
                    assert available_tokens == 0, "Available tokens should be non-negative"
                    break

        input_ids = torch.concat(input_ids, dim=0)
        labels = torch.concat(labels, dim=0)
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)

        return dict(input_ids=input_ids,
                    attention_mask=None,
                    labels=labels,
                    seq_lengths=seq_lengths)


class PackedDataset(Dataset):
    def __init__(self, dataset: Dataset, max_length: int):
        self.dataset = dataset
        self.max_length = max_length

        print("Packing dataset into fixed-length sequences...")
        self.bins = binpack_bfd(
            [len(item["input_ids"]) for item in self.dataset],
            self.max_length,
        )
        print(f"Packed dataset into {len(self.bins)} bins.")
        print(f"Average bin utilization: {np.mean([sum(len(self.dataset[i]["input_ids"]) for i in bin_indices)/self.max_length for bin_indices in self.bins]):.4f}")

    def __get_single_item(self, index: int) -> Dict[str, Any]:
        instance = self.dataset[index]
        input_ids = instance["input_ids"]
        labels = instance["input_ids"].copy()
        labels[-1] = -100  # Do not predict the last token
        return dict(
            input_ids=torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
            labels=torch.tensor(labels, dtype=torch.long).unsqueeze(0),
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        bin_indices = self.bins[index]
        input_ids_list = []
        labels_list = []
        position_ids_list = []
        
        for i in bin_indices:
            item = self.__get_single_item(i)
            input_ids_list.append(item["input_ids"])
            labels_list.append(item["labels"])
            position_ids_list.append(torch.arange(item["input_ids"].shape[1], dtype=torch.long).unsqueeze(0))

        return dict(
            input_ids=torch.cat(input_ids_list, dim=1),
            labels=torch.cat(labels_list, dim=1),
            position_ids=torch.cat(position_ids_list, dim=1),
        )

    def __len__(self):
        return len(self.bins)


@dataclass
class FlattenedDataCollatorForLanguageModeling(object):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        
        max_length = self.tokenizer.model_max_length
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=1)

        length = input_ids.shape[1]

        if max_length is not None and length < max_length:
                # Pad to the max_length
                pad_length = max_length - length
                input_ids = torch.nn.functional.pad(
                    input_ids, (0, pad_length), "constant", self.tokenizer.pad_token_id
                )
                labels = torch.nn.functional.pad(
                    labels, (0, pad_length), "constant", IGNORE_INDEX
                )
                position_ids = torch.nn.functional.pad(
                    position_ids, (0, pad_length), "constant", 0
                )

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            position_ids=position_ids,
        )
        return batch
