from functools import partial
from dataclasses import dataclass, field
from typing import Optional

from .math import *
from .ultrachat import *
from .synth_long import *
from .buddhi import *
from .booksum import *
from .long_alpaca import *
from .niah import *
from .utils import PackedDataset, FlattenedDataCollatorForLanguageModeling
# from .utils import SortByLengthDataset


DATASET_LOADER = {
    "openr1_math": load_openr1_math_dataset,
    "ultrachat": load_ultrachat_dataset,
    "synth_long": load_synth_long_dataset,
    "long_alpaca": load_long_alpaca_dataset,
    "buddhi": load_buddhi_dataset,
    "booksum": load_booksum_dataset,
    "niah": load_synthetic_niah_dataset,
}


def load_dataset(training_args, tokenizer):

    dataset_names = training_args.dataset_name.split(",")
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name in DATASET_LOADER:
            dataset = DATASET_LOADER[dataset_name](
                dataset_name,
                tokenizer,
                training_args.training_max_length,
                max_samples=training_args.max_samples,
            )
            print(f"Loaded {dataset_name} dataset with {len(dataset)} samples.")
            datasets.append(dataset)
        else:
            raise ValueError(f"Dataset {dataset_name} is not supported.")

    if len(datasets) == 1:
        return datasets[0]
    else:
        from datasets import concatenate_datasets
        concat_dataset = concatenate_datasets(datasets)
        print(f"Concatenated dataset has {len(concat_dataset)} samples.")
        return concat_dataset
