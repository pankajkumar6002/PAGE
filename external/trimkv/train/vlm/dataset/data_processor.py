import json
import os
import random
import logging
import re
import time
import itertools
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple, Any
from tqdm import tqdm
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import transformers

from .configs import get_dataset_configs
from .qwen_utils import (
    get_rope_index_2,
    get_rope_index_25,
    get_rope_index_3,
    preprocess_qwen_visual,
    update_processor_pixels,
)
from .llava_utils import (
    get_llava_rope_index,
    preprocess_llava_visual,
    llava_apply_chat_template,
)

from .utils import binpack_bfd

local_rank = None
IGNORE_INDEX = -100

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _make_abs_paths(base: Path, files: str) -> str:
    return f"{(base / files).resolve()}"


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, processor, data_args):
        super(LazySupervisedDataset, self).__init__()

        self.dataset_name = data_args.dataset_use
        dataset_names = self.dataset_name.split(",")
        dataset_configs = get_dataset_configs(dataset_names, data_args.dataset_dir)

        rank0_print(f"Loading datasets: {dataset_configs}")

        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
            self.preprocess_fn = preprocess_qwen_visual
            # update_processor_pixels(processor, data_args)
        elif data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
            self.preprocess_fn = preprocess_qwen_visual
            # update_processor_pixels(processor, data_args)
        elif data_args.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
            self.preprocess_fn = preprocess_qwen_visual
            # update_processor_pixels(processor, data_args)
        elif data_args.model_type == "llava1.5":
            self.get_rope_index = get_llava_rope_index
            self.preprocess_fn = preprocess_llava_visual
            processor.tokenizer.apply_chat_template = llava_apply_chat_template.__get__(processor.tokenizer)
        else:
            raise ValueError(f"model_type: {data_args.model_type} not supported")

        self.processor = processor
        self.tokenizer = processor.tokenizer

        list_data_dict = []

        for data, dataname in zip(dataset_configs, dataset_names):
            file_format = data["annotation_path"].split(".")[-1]
            path = data["annotation_path"].replace(f".{file_format}", f"_{data_args.model_type.replace('.', '_')}_seqlen.{file_format}")
            if not os.path.exists(path):
                print(f"Precomputed data file {path} not found, using original annotation file. We will need to compute seqlen on the fly, which is slower.")
                path = data["annotation_path"]

            if file_format == "jsonl":
                annotations = read_jsonl(path)
            else:
                annotations = json.load(open(path, "r"))
                
            for ann in annotations:
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = data["data_path"]
                else:
                    ann["data_path"] = data["data_path"]

            print(f"[{dataname}] - Before filtering, {len(annotations)} samples.")

            annotations = self._filter_data(annotations, processor, min_length=data.get("min_length", 128))
            
            print(f"[{dataname}] - After filtering, {len(annotations)} samples remain.")
            print(f"[{dataname}] - Average seqlen: {np.mean([ann['seqlen'] for ann in annotations])}")
            print(f"[{dataname}] - Max seqlen: {np.max([ann['seqlen'] for ann in annotations])}")
            print(f"[{dataname}] - Min seqlen: {np.min([ann['seqlen'] for ann in annotations])}")

            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                rank0_print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")

            print(f"[{dataname}] - Using {len(annotations)} samples after sampling.")

            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", None)
        self.list_data_dict = list_data_dict

        if data_args.data_packing:
            packed_file = os.path.join(data_args.dataset_dir, f"packed_{data_args.dataset_use.replace(',', '_')}.json")
            self.packed_chunks = self._build_packed_index(data_args.data_packing_shuffle)
            # if data_args.repacking or (not os.path.exists(packed_file)):
            #     json.dump(self.packed_chunks, open(packed_file, "w"))
            # else:
            #     self.packed_chunks = json.load(open(packed_file, "r"))
            #     print(f"Load packed chunks from {packed_file}, num_chunks: {len(self.packed_chunks)}")

            self.item_fn = self._get_packed_item
            self._calc_dataset_size = lambda: len(self.packed_chunks)
        else:
            self.item_fn = self._get_item
            self._calc_dataset_size = lambda: len(self.list_data_dict)


    def _filter_data(self, list_data_dict, processor, min_length=128):
        filtered = []
        def good_enough(sample):
            seqlen = sample["seqlen"]
            if (seqlen >= processor.tokenizer.model_max_length) or (seqlen <= min_length):
                return False
            return True

        for i, sample in enumerate(tqdm(list_data_dict)):
            if "seqlen" in sample:
                seqlen = sample["seqlen"]
            else:
                seqlen = self._compute_sample_seq_length(sample)
                sample['seqlen'] = seqlen
            seqlen += 21 # buffer for thinking tokens
            
            if not good_enough(sample):
                continue
            filtered.append(sample)

        return filtered
    
    def _compute_sample_seq_length(self, sample) -> int:
        data = self.preprocess_fn(
            [sample],
            self.processor,
        )

        input_ids = data["input_ids"]
        # print(self.tokenizer.batch_decode(input_ids, skip_special_tokens=False))
        # # raise ValueError("Debugging packed item")
        # breakpoint()
        return len(input_ids[0]) if isinstance(input_ids, list) else input_ids.shape[1] 

    def _build_packed_index(self, shuffle: bool = True, repacking: bool = False):
        max_len = self.tokenizer.model_max_length
        lengths = self.pre_calculated_length.tolist()

        packed_indices = binpack_bfd(lengths, max_len)
        num_padded = sum([max_len - sum([lengths[idx] for idx in group]) for group in packed_indices])
        print(f"Packing {len(self.list_data_dict)} samples into {len(packed_indices)} chunks")
        print(f"Total padding tokens: {num_padded}, avg padding per chunk: {num_padded/len(packed_indices):.2f}")
        return packed_indices

    def __len__(self):
        return self._calc_dataset_size()

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "seqlen" in self.list_data_dict[0]:
            # to ensure we account for thinking tokens from reasoning models
            length_list = [sample["seqlen"] + 21 for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            length_list = [self._compute_sample_seq_length(sample) + 21 for sample in self.list_data_dict]
            return np.array(length_list)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            # try:
                # start_time = time.time()
                sample = self.item_fn(i)
                # end_time = time.time()
                # print(f"[Try #{attempt_idx}] Successfully fetched sample {i} in {end_time - start_time:.2f}s")
                return sample
            # except Exception as e:
            #     # sleep 1s in case it is a cloud disk issue
            #     print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                # time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                print(f"[Try other #{attempt_idx}] Try other samples instead of {i}")
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                sample = self.item_fn(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sources = self.list_data_dict[i]
            if isinstance(sources, dict):
                sources = [sources]
            sample = self.item_fn(sources)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, sample_idx) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[sample_idx]
        if isinstance(sources, dict):
            sources = [sources]

        data_dict = self.preprocess_fn(
            sources,
            self.processor,
        )

        seq_len = data_dict["input_ids"][0].size(0)
        # assert seq_len == sources[0]['seqlen'], f"Sequence length mismatch: computed {seq_len}, precomputed {sources[0]['seqlen']}, source: {sources[0]}"

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size
                / self.processor.video_processor.fps
            ] * len(video_grid_thw)
        else:
            video_grid_thw = None
            second_per_grid_ts = None

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]

        # text = self.processor.tokenizer.decode(
        #     data_dict["input_ids"][0], skip_special_tokens=False
        # )

        # labels = data_dict["labels"][0]
        # labels = [
        #     tid if tid != -100 else self.processor.tokenizer.pad_token_id
        #     for tid in labels
        # ]
        # label = self.processor.tokenizer.decode(labels, skip_special_tokens=False)

        return data_dict

    def _get_packed_item(self, chunk_idx) -> Dict[str, torch.Tensor]:
        sample_indices = self.packed_chunks[chunk_idx]

        data_list = []
        new_data_dict = {}
        for sample_idx in sample_indices:
            data_list.append(self._get_item(sample_idx))

        max_length = self.tokenizer.model_max_length
        while sum(d["input_ids"].shape[1] for d in data_list) > max_length:
            # print(len(data_list))
            # for idx, sample_idx in enumerate(sample_indices):
            #     source = self.list_data_dict[sample_idx]
            #     print(source)
            #     print(f"Sample {idx, sample_idx} seqlen: {source['seqlen']}, {data_list[idx]['input_ids'].shape[1]}", flush=True)
            print(f"Warning: packed chunk exceeded max length ({max_length}), dropping last sample. {[d['input_ids'].shape[1] for d in data_list]}")
            print(sum(d["input_ids"].shape[1] for d in data_list))
            print(max_length)
            print(sample_indices)
            print([self.list_data_dict[sample_idx] for sample_idx in sample_indices])
            for i, d in enumerate(data_list):
                print(d["input_ids"].shape[1], self.list_data_dict[sample_indices[i]]['seqlen'])
            raise ValueError
            data_list.pop()

        input_ids = torch.cat([d["input_ids"] for d in data_list], dim=1)
        # print(self.tokenizer.batch_decode(input_ids, skip_special_tokens=False))
        
        labels = torch.cat([d["labels"] for d in data_list], dim=1)
        if "position_ids" in data_list[0] and data_list[0]["position_ids"] is not None:
            position_ids = torch.cat([d["position_ids"] for d in data_list], dim=2)
        else:
            position_ids = None

        attention_mask = [
            d["attention_mask"][0] for d in data_list if "attention_mask" in d
        ]
        new_data_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "attention_mask": attention_mask if attention_mask else None,
        }

        if any("pixel_values" in d for d in data_list):
            new_data_dict.update(
                {
                    "pixel_values": torch.cat(
                        [
                            d["pixel_values"]
                            for d in data_list
                            if "pixel_values" in d
                        ],
                        dim=0,
                    ),
                }
            )
        if any("image_grid_thw" in d for d in data_list):
            new_data_dict.update(
                {
                    "image_grid_thw": torch.cat(
                        [
                            d["image_grid_thw"]
                            for d in data_list
                            if "image_grid_thw" in d
                        ],
                        dim=0,
                    ),
                }
            )

        if any("pixel_values_videos" in d for d in data_list):
            new_data_dict.update(
                {
                    "pixel_values_videos": torch.cat(
                        [
                            d["pixel_values_videos"]
                            for d in data_list
                            if "pixel_values_videos" in d
                        ],
                        dim=0,
                    ),
                }
            )

        if any("video_grid_thw" in d for d in data_list):
            new_data_dict.update(
                {
                    "video_grid_thw": torch.cat(
                        [
                            d["video_grid_thw"]
                            for d in data_list
                            if "video_grid_thw" in d
                        ],
                        dim=0,
                    ),
                }
            )

        return new_data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        max_length = self.tokenizer.model_max_length
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        doc_pos_ids = torch.cat([torch.arange(L, dtype=torch.int32) for L in attention_mask])
        if all(p is not None for p in position_ids):
            position_ids = torch.cat(position_ids, dim=2) # 3, 1, s
            position_ids = torch.cat([doc_pos_ids.unsqueeze(0).unsqueeze(0), position_ids], dim=0)
        else:
            position_ids = doc_pos_ids.unsqueeze(0)

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
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )

        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            batch["pixel_values"] = concat_images

        if all("image_grid_thw" in instance for instance in instances):
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
            batch["image_grid_thw"] = grid_thw

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            batch["pixel_values_videos"] = concat_videos

        if all("video_grid_thw" in instance for instance in instances):
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
            batch["video_grid_thw"] = video_grid_thw

        return batch


def make_supervised_data_module(processor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(processor, data_args=data_args)
    if data_args.data_flatten or data_args.data_packing:
        data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
