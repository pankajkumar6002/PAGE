# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, List, Tuple
from dataclasses import dataclass, field
import torch
from compute_scores import compute_scores
from datasets import load_dataset
from eval_utils import (
    DATA_NAME_TO_MAX_NEW_TOKENS,
    GreedySearch,
    create_multiturn_prompt,
    create_scdq_prompt,
    dump_jsonl,
    get_compressed_examples,
    get_ground_truth,
    load_data,
)
from torch import Tensor
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    MambaForCausalLM,
    Qwen2ForCausalLM,
)
from transformers.cache_utils import SinkCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils.import_utils import _is_package_available

from load_model import load_model

import random


@dataclass
class Config:
    # experiment parameters
    task: str = field(default="scbench_summary", metadata={"help": "Task name"})
    output_dir: str = field(default="./results", metadata={"help": "Output directory"})
    n_samples: int = field(default=1, metadata={"help": "Number of samples"})
    num_inspect: int = field(default=2, metadata={"help": "Number of inspected samples"})
    seed: int = field(default=42, metadata={"help": "Random seed"})
    eval_batch_size: int = field(default=1, metadata={"help": "Batch size for evaluation"})
    max_return_sequences: int = field(default=-1, metadata={"help": "Max return sequences"})
    name_suffix: str = field(default="", metadata={"help": "Name suffix of the run"})
    evaluate: bool = field(default=True, metadata={"help": "Whether to evaluate the results"})
    rewrite: bool = field(default=False, metadata={"help": "Whether to rewrite existing results"})
    # data parameters
    data_dir: str = field(default="./data/scbench", metadata={"help": "Data directory"})
    max_input_length: int = field(default=200000, metadata={"help": "Max input length"})
    max_turns: int = field(default=5, metadata={"help": "Max number of turns to consider"})
    # evaluation parameters
    num_eval_examples: int = field(default=-1, metadata={"help": "Number of evaluation examples"})
    verbose: bool = field(default=False, metadata={"help": "Whether to print verbose logs"})
    eval_mode: str = field(default="multiturn", metadata={"help": "Evaluation mode (multiturn or scdq)"})
    # inference parameters
    max_new_tokens: int = field(default=500, metadata={"help": "Max new tokens to generate"})
    top_k: int = field(default=-1, metadata={"help": "Top k for generation"})
    start_example_id: int = field(default=0, metadata={"help": "Start example ID for evaluation"})
    use_chat_template: bool = field(default=True, metadata={"help": "Whether to use chat template"})

    # generation parameters
    temperature: float = field(default=0.6, metadata={"help": "Temperature"})
    top_p: float = field(default=0.95, metadata={"help": "Top p"})
    do_sample: bool = field(default=False, metadata={"help": "Whether to do sampling"})
    # model parameters 
    method: str = field(default="fullkv", metadata={"help": "KV method"})
    model_path: str = field(default="Qwen/Qwen3-1.7B", metadata={"help": "Model"})
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Attention implementation"})
    max_model_len: int = field(default=200000, metadata={"help": "Max model length"})
    download_from: str = field(default='wandb', metadata={"help": "Where to download the model from, 'local' or 'wandb' or 'huggingface'"})

    # general parameters for compression
    kv_budget: int = field(default=None, metadata={"help": "KV budget for compression"})
    update_kv: bool = field(default=True, metadata={"help": "Whether to update KV"})
    compress_strategy: str = field(default="alpha", metadata={"help": "Compression strategy"})
    buffer_size: int = field(default=16, metadata={"help": "Buffer size for compression"})
    lookahead_steps: int = field(default=1, metadata={"help": "Number of lookahead steps for scoring tokens in trimkv"})

    # for RKV compression
    window_size: int = field(default=8, metadata={"help": "Window size for compression"})
    mix_lambda: float = field(default=0.1, metadata={"help": "Mix lambda for compression"})
    retain_ratio: float = field(default=0.2, metadata={"help": "Retain ratio for compression"})
    retain_direction: str = field(default="last", metadata={"help": "Retain direction for compression"})
    divide_method: str = field(default="step_length", metadata={"help": "Method to divide input"})
    divide_length: int = field(default=16, metadata={"help": "Length to divide input"})
    compression_content: str = field(default="all", metadata={"help": "Content to compress"})

    # for streamingllm
    first_tokens: int = field(default=128, metadata={"help": "First tokens for compression"})

    def update_from_dict(self, args):
        for k, v in args.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown argument: {k}")
            setattr(self, k, v)
        return self

    @property
    def model_name(self):
        model_path = self.model_path.strip('/')
        model_name = os.path.basename(model_path).replace("/", "_").replace("-", "_").replace(",", "_").replace(":", "_")
        return model_name

    @property
    def run_name(self):
        name = f"{self.method}{self.name_suffix}-{self.kv_budget}b-{self.max_model_len}l"

        if self.n_samples is not None:
            name += f"-{self.n_samples}nspl"

        name += f"_{self.seed}s"
        if not self.do_sample:
            name += "-greedy"

        return name

# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
def truncate_input(input: list, max_length: int, manner="middle"):
    if max_length < 0:
        return input
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens or max_tokens < 0
    return tokens


def get_pred(
    model,
    tokenizer,
    eg,
    data_name,
    max_new_tokens,
    max_input_length: int,
    eval_mode: str = "multiturn",
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    if eval_mode == "scdq":
        encoded_eg = create_scdq_prompt(
            eg,
            data_name=data_name,
            tok=tokenizer,
            use_chat_template=True,
            use_vllm=False,
        )
    elif eval_mode == "multiturn":
        # multi-turn mode
        encoded_eg = create_multiturn_prompt(
            eg,
            data_name=data_name,
            tok=tokenizer,
            use_chat_template=True,
            use_vllm=False,
            disable_golden_context=True, # always disable golden context in multi-turn mode
        )
    else:
        raise ValueError(f"Unknown eval_mode: {eval_mode}")

    context = truncate_by_tokens(
        encoded_eg["prompts"][0], model.tokenizer, max_input_length
    )
    encoded_eg["prompts"][0] = context

    if eval_mode == "scdq":
        # scdq mode has no action for disable_golden_context
        outputs = model.test_scdq(encoded_eg, max_length=max_new_tokens)
    else:
        # multi-turn mode test
        outputs = model.test(
            encoded_eg,
            max_length=max_new_tokens,
        )

    print("Chunked generation:", json.dumps(outputs, indent=2, ensure_ascii=False))
    return outputs


def main(**kwargs):
    config = Config()
    config = config.update_from_dict(kwargs)
    # check_benchmark_availability(args.data_dir)
    model_name = config.model_path
    real_model_name = model_name.replace(',', '_').strip("/").split("/")[-1]
    data_name = config.task
    scdq_mode = config.eval_mode == "scdq"

    if "," in data_name:
        data_names = data_name.split(",")
    else:
        data_names = [data_name]

    model, tokenizer, cache_creator = load_model(config)
    model = GreedySearch(model, tokenizer, cache_creator)

    result_dir = Path(
        config.output_dir,
        f"{real_model_name}/{config.method}_{config.eval_mode}",
    )
    result_dir.mkdir(exist_ok=True, parents=True)

    results = {}
    for data_name in data_names:
        max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
        if 'qwen3' in model_name.lower() and isinstance(max_new_tokens, int):
            max_new_tokens = max_new_tokens + 256 # Qwen3 tends to generate more tokens due to the difference in the tokenizer

        if isinstance(max_new_tokens, dict):
            assert (
                max(max_new_tokens.values()) <= config.max_model_len
            ), "max_new_tokens must be less than max_model_len"
        elif max_new_tokens >= config.max_model_len:
            max_new_tokens = 500

        # Data
        output_path = (
            result_dir / f"prediction_{data_name}_{config.run_name}.jsonl"
        )

        examples = load_dataset("microsoft/SCBench", data_name, split="test")

        max_turn_size = len(examples[0]["multi_turns"])
        if config.max_turns > 0 and config.max_turns < max_turn_size:
            examples = [
                {**eg, "multi_turns": eg["multi_turns"][: config.max_turns]}
                for eg in examples
            ]
            max_turn_size = config.max_turns

        if config.num_eval_examples != -1:
            num_eval_examples = min(config.num_eval_examples, len(examples))
            examples = examples[:num_eval_examples]

        preds = []
        print(f"==== Evaluation {data_name}====")
        print(f"# examples: {len(examples)}")
        print(f"Num eval examples: {config.num_eval_examples}")
        print(f"Verbose: {config.verbose}")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"Num of turns: {max_turn_size}")

        done = set()
        if os.path.exists(output_path) and not config.rewrite:
            print(f"Output file {output_path} exists. Loading from file.")
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    tmp = json.loads(line)
                    done.add(int(tmp["id"]))
                    preds.append(tmp)
            # examples = examples[len(preds):]
            compute_scores(
                output_path, data_name, real_model_name, config.max_model_len, scdq_mode
            )

        for i, eg in tqdm(enumerate(examples)):
            if i < config.start_example_id or i in done:
                continue
            if data_name in [
                "scbench_summary_with_needles",
                "scbench_repoqa_and_kv",
            ]:
                max_input_length = config.max_model_len - (
                    sum(list(max_new_tokens.values())) * max_turn_size // 2
                )
            else:
                max_input_length = config.max_model_len - max_new_tokens * max_turn_size
            if scdq_mode:
                max_input_length -= 1000

            pred = get_pred(
                model,
                tokenizer=tokenizer,
                eg=eg,
                data_name=data_name,
                max_new_tokens=max_new_tokens,
                max_input_length=max_input_length,
                eval_mode=config.eval_mode,
            )
            # a list of ground truth answers for each turn
            gts = get_ground_truth(eg, data_name)
            for turn_idx, (ans, gt, turn) in enumerate(
                zip(pred["answers"], gts, eg["multi_turns"])
            ):
                case = {
                    "id": i,
                    "turn_idx": turn_idx,
                    "prediction": ans,
                    "ground_truth": gt,
                }
                if "task" in pred:
                    case["task"] = pred["task"][turn_idx]
                if data_name == "scbench_repoqa":
                    case["lang"] = eg["lang"]
                    case["repo"] = eg["repo"]
                    case["func_name"] = turn["name"]
                if data_name == "scbench_repoqa_and_kv":
                    case["lang"] = eg["lang"]
                    case["repo"] = eg["repo"]
                    if turn["task"] == "scbench_repoqa":
                        case["func_name"] = turn["name"]
                if data_name == "scbench_kv_compressible":
                    case["task"] = eg["task"]
                preds.append(case)
            dump_jsonl(preds, output_path)
            torch.cuda.empty_cache()
            done.add(i)

        score = compute_scores(
            output_path,
            data_name,
            real_model_name,
            max_seq_length=config.max_model_len,
            scdq_mode=scdq_mode,
            method=config.method,
            kv_budget=config.kv_budget,
        )
        results[data_name] = score
        with open(os.path.join(config.output_dir, "summary_results.jsonl"), "a", encoding="utf-8") as f:
            f.write(f"{model_name},{data_name},{config.max_model_len},{config.method},{config.kv_budget},{score}\n")

    print("==== Results ====")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    import fire
    fire.Fire(main)
