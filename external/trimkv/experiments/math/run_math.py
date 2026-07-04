import json
import fire
import os
import random
import argparse
import torch
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from collections import defaultdict
from load_model import load_model
from transformers import DynamicCache
from generation_utils import batch_exist_generate
from utils import estimate_max_batch_size
from trimkv.cache_utils import TrimKVCache
from dotenv import load_dotenv
import eval_math


dataset2key = {
    "gsm8k": ["question", "answer"],
    "aime24": ["question", "answer"],
    "math": ["problem", "answer"],
}

dataset2max_length = {
    "gsm8k": 16384,
    "aime24": 32768,
    "math": 16384,
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


prompt_template = "You are given a math problem.\n\nProblem: {question}\n\n You need to solve the problem step by step. First, you need to provide the chain-of-thought, then provide the final answer.\n\n Provide the final answer in the format: Final answer:  \\boxed{{}}"


def apply_chat_template(prompts, tokenizer):
    chat_prompts = []

    for prompt in prompts:
        messages = [
            {"role": "user", "content": prompt}
        ]
        chat_messages = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # enable_thinking=True,
        )
        chat_prompts.append(chat_messages)
    return chat_prompts


@dataclass
class Config:
    # experiment parameters
    dataset: str = field(default="aime24", metadata={"help": "Dataset to use for evaluation"})
    dataset_dir: str = field(default="./data", metadata={"help": "Path to dataset"})
    output_dir: str = field(default="./results", metadata={"help": "Output directory"})
    n_samples: int = field(default=1, metadata={"help": "Number of samples"})
    num_inspect: int = field(default=2, metadata={"help": "Number of inspected samples"})
    seed: int = field(default=42, metadata={"help": "Random seed"})
    eval_batch_size: int = field(default=1, metadata={"help": "Batch size for evaluation"})
    max_return_sequences: int = field(default=-1, metadata={"help": "Max return sequences"})
    resume: bool = field(default=True, metadata={"help": "Whether to resume from previous run"})
    name_suffix: str = field(default="", metadata={"help": "Name suffix of the run"})
    evaluate: bool = field(default=True, metadata={"help": "Whether to evaluate the results"})
    start_idx: int = field(default=0, metadata={"help": "Start index for evaluation"})
    end_idx: int = field(default=-1, metadata={"help": "End index for evaluation"})

    # generation parameters
    temperature: float = field(default=0.6, metadata={"help": "Temperature"})
    top_p: float = field(default=0.95, metadata={"help": "Top p"})
    do_sample: bool = field(default=True, metadata={"help": "Whether to do sampling"})
    # model parameters 
    method: str = field(default="fullkv", metadata={"help": "KV method"})
    model_path: str = field(default="Qwen/Qwen3-1.7B", metadata={"help": "Model"})
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Attention implementation"})
    max_model_len: int = field(default=None, metadata={"help": "Max model length"})
    download_from: str = field(default='local', metadata={"help": "Where to download the model from, 'local' or 'wandb' or 'huggingface'"})

    # general parameters for compression
    kv_budget: int = field(default=None, metadata={"help": "KV budget for compression"})
    update_kv: bool = field(default=True, metadata={"help": "Whether to update KV"})
    compress_strategy: str = field(default="alpha", metadata={"help": "Compression strategy"})
    buffer_size: int = field(default=128, metadata={"help": "Buffer size for compression"})
    lookahead_steps: int = field(default=1, metadata={"help": "Number of lookahead steps for scoring tokens in trimkv"})

    # for RKV compression
    window_size: int = field(default=8, metadata={"help": "Window size for compression"})
    mix_lambda: float = field(default=0.1, metadata={"help": "Mix lambda for compression"})
    retain_ratio: float = field(default=0.2, metadata={"help": "Retain ratio for compression"})
    retain_direction: str = field(default="last", metadata={"help": "Retain direction for compression"})
    divide_method: str = field(default="step_length", metadata={"help": "Method to divide input"})
    divide_length: int = field(default=128, metadata={"help": "Length to divide input"})
    compression_content: str = field(default="all", metadata={"help": "Content to compress"})

    # for streamingllm
    first_tokens: int = field(default=4, metadata={"help": "First tokens for compression"})

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


# write a generation function that can handle some exceptions, restarting the generation if it fails, trying up to 3 times
def generation_with_retries(model, cache_creator, config, **generate_kwargs):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            past_key_values = cache_creator(model, max_model_len=config.max_model_len)
            generate_kwargs['past_key_values'] = past_key_values
            output = model.generate(**generate_kwargs)
            return output
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"Out of memory error on attempt {attempt + 1}. Retrying...")
                torch.cuda.empty_cache()
            else:
                raise e
    raise RuntimeError("Generation failed after multiple attempts due to out of memory errors.")


def evaluate(save_path, dataset):
    args = argparse.Namespace()
    args.exp_name = "math_eval"
    args.prompt_type = "cot"
    args.output_dir = os.path.dirname(save_path)
    args.stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "\n题目："]
    args.dataset = dataset
    args.base_dir = os.path.dirname(save_path)
    args.input_path = save_path
    args.num_workers = 4
    eval_math.main(data_name=dataset, args=args, json_file=save_path)



def main(**kwargs):
    config = Config()
    config.__dict__.update(kwargs)
    set_seed(config.seed)

    if config.max_model_len is None:
        config.max_model_len = dataset2max_length[config.dataset]

    if config.kv_budget is None or config.method.lower() == "fullkv":
        config.kv_budget = config.max_model_len

    output_dir = os.path.join(config.output_dir, f"{config.dataset}/{config.model_name}")
    save_path = os.path.join(output_dir, f"{config.run_name}_{config.start_idx}.jsonl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Saving results to {save_path}")

    done = defaultdict(int)
    if config.resume and os.path.exists(save_path):
        print(f"Resuming from {save_path}")
        # get what is already in the file
        with open(save_path, "r") as f:
            # get number of rollouts already done for each sample
            for line in f.readlines():
                example = json.loads(line)
                sample_idx = example.get("sample_idx", -1)
                done[sample_idx] += 1
        fout = open(save_path, "a")
    else:
        fout = open(save_path, "w")

    prompts = []
    test_data = []

    dataset_path = os.path.join(config.dataset_dir, f"{config.dataset}.jsonl")
    with open(dataset_path) as f:
        for index, line in enumerate(f):
            example = json.loads(line)
            question_key = dataset2key[config.dataset][0]

            question = example[question_key]
            example["question"] = question
            prompt = prompt_template.format(**example)

            example["prompt"] = prompt
            example["index"] = index
            prompts.append(prompt)
            test_data.append(example)

    model, tokenizer, cache_creator = load_model(config)

    if config.max_return_sequences <= 0:
        # Phi need to switch between short and long rope, which recomputes the input, so we need to set a minimum sequence length
        if 'seerattn' in config.method:
            min_seq_len = dataset2max_length[config.dataset]
        elif 'phi' in config.model_path:
            min_seq_len = 4096
        else:
            min_seq_len = 512
        est = estimate_max_batch_size(model, tokenizer, cache_creator, config.kv_budget, min_seq_len=min_seq_len)
        print(f"Estimating max batch size: {est}")
        config.max_return_sequences = est["estimated_max_batch_size"]
        print(f"Setting max return sequences to {config.max_return_sequences}")

    assert config.eval_batch_size == 1, "Batch size must be 1 for evaluation"

    while True:
        all_done = True
        start_idx = config.start_idx
        end_idx = config.end_idx if config.end_idx >= 0 else len(prompts)
        for i in tqdm(range(start_idx, len(prompts), config.eval_batch_size)):
            if done[i] >= config.n_samples:
                continue

            all_done = False
            batch_prompts = prompts[i: i + config.eval_batch_size]
            batch_prompts = apply_chat_template(batch_prompts, tokenizer)
            tokenized_prompts = tokenizer(
                batch_prompts,
                padding="longest",
                return_tensors="pt",
                add_special_tokens=True,
            ).to("cuda")

            prefill_length = tokenized_prompts["attention_mask"].sum(dim=1).tolist()[0]
            print(f"Sample {i}, prompt length: {prefill_length}, done: {done[i]}/{config.n_samples}")

            n_return_sequences = min(config.max_return_sequences, config.n_samples - done[i])

            output = generation_with_retries(
                model,
                cache_creator,
                config,
                **tokenized_prompts,
                max_length=config.max_model_len,
                do_sample=config.do_sample,
                num_beams=1,
                temperature=config.temperature,
                top_p=config.top_p,
                num_return_sequences=n_return_sequences,
            )

            batch_token_stats = []
            for j in range(output.size(0)):
                total_tokens = int((output[j] != tokenizer.pad_token_id).sum().item())

                prefill = prefill_length
                output_tokens = total_tokens - prefill

                batch_token_stats.append(
                    {
                        "sample_idx": i,
                        "prefill_tokens": prefill,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                    }
                )

            batch_outputs = tokenizer.batch_decode(
                [output[j][prefill_length:] for j in range(output.size(0))],
                skip_special_tokens=True,
            )

            torch.cuda.empty_cache()

            for j in range(len(batch_outputs)):
                sample_idx = batch_token_stats[j]["sample_idx"]
                out = {
                    "prompt": batch_prompts[0],
                    "output": batch_outputs[j],
                    **batch_token_stats[j],
                    **test_data[sample_idx],
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
                done[i] += 1

        if all_done:
            print("All done!")
            break

    fout.close()

    if config.evaluate:
        evaluate(save_path, config.dataset)


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(main)
