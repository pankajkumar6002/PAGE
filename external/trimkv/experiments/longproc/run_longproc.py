import os
import numpy as np
import json
import random
import fire
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from longproc.longproc_data import load_longproc_data
from openai import OpenAI
from tqdm import tqdm

import torch
from trimkv.cache_utils import TrimKVCache
from load_model import load_model
from dotenv import load_dotenv


# allows some buffer to accommodate variations in token usage for different tokenizers
dataset2max_length = {
    "0.5k": 1536,
    "2k": 4096,
    "8k": 12288,
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    # experiment parameters
    dataset: str = field(default="countdown_0.5k", metadata={"help": "Dataset to use for evaluation"})
    dataset_dir: str = field(default="./data", metadata={"help": "Path to dataset"})
    output_dir: str = field(default="./results", metadata={"help": "Output directory"})
    n_samples: int = field(default=1, metadata={"help": "Number of samples"})
    seed: int = field(default=42, metadata={"help": "Random seed"})
    max_return_sequences: int = field(default=1, metadata={"help": "Max return sequences"})
    resume: bool = field(default=True, metadata={"help": "Whether to resume from previous run"})
    gen_length: int = field(default=None, metadata={"help": "Generation length"})

    retriever_type: str = field(default="orig-session", metadata={"help": "Retriever type (orig-session, orig-turn)"})
    topk_context: int = field(default=1000, metadata={"help": "Number of top-k context"})
    history_format: str = field(default="json", metadata={"help": "History format (json or nl)"})
    useronly: bool = field(default=True, metadata={"help": "Whether to use user only history (true or false)"})
    cot: bool = field(default=True, metadata={"help": "Whether to use chain-of-thought (true or false)"})
    con: bool = field(default=False, metadata={"help": "Reading method (con or non-con)"})
    merge_key_expansion_into_value: str = field(default="none", metadata={"help": "How to merge key expansion into value (merge, replace, none)"}) # for retrieval-augmented models

    # generation parameters
    temperature: float = field(default=0.6, metadata={"help": "Temperature"})
    top_p: float = field(default=0.95, metadata={"help": "Top p"})
    do_sample: bool = field(default=True, metadata={"help": "Whether to do sampling"})
    # model parameters
    method: str = field(default="fullkv", metadata={"help": "KV method (fullkv, trimkv, rkv, snapkv, streamingllm)"})
    model_path: str = field(default="Qwen/Qwen3-4B-Instruct-2507", metadata={"help": "Model"})
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Attention implementation"})
    max_model_len: int = field(default=131072, metadata={"help": "Max model length"})
    download_from: str = field(default='local', metadata={"help": "Where to download the model from, 'local' or 'wandb' or 'huggingface'"})

    # general parameters for compression
    kv_budget: int = field(default=None, metadata={"help": "KV budget for compression"})
    update_kv: bool = field(default=True, metadata={"help": "Whether to update KV"})
    buffer_size: int = field(default=128, metadata={"help": "Buffer size for compression"})
    compress_strategy: str = field(default="alpha", metadata={"help": "Compression strategy"})

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
    def split(self):
        return self.dataset.split("_")[-1]

    @property
    def model_name(self):
        model_path = self.model_path.strip('/')
        model_name = os.path.basename(model_path).replace("/", "_").replace("-", "_").replace(",", "_").replace(":", "_")
        return model_name

    @property
    def max_gen_length(self):
        if self.gen_length is not None:
            return self.gen_length
        return dataset2max_length[self.split]

    @property
    def run_name(self):
        name = f"{self.method}-{self.kv_budget}b-{self.max_gen_length}l"

        if self.n_samples is not None:
            name += f"-{self.n_samples}nspl"

        name += f"_{self.seed}s"
        if not self.do_sample:
            name += "-greedy"

        return name


def apply_chat_template(prompts, tokenizer, **kwargs):
    chat_prompts = []

    for prompt in prompts:
        messages = [
            {"role": "user", "content": prompt}
        ]
        chat_messages = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
        chat_prompts.append(chat_messages)
    return chat_prompts


def main(**kwargs):
    config = Config()
    config.__dict__.update(kwargs)

    random.seed(config.seed)
    torch.manual_seed(config.seed)

    if config.kv_budget is None:
        config.kv_budget = config.max_model_len

    output_dir = os.path.join(config.output_dir, f"{config.dataset}/{config.model_name}")
    name = config.run_name

    save_path = os.path.join(output_dir, f"{config.run_name}.jsonl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Saving results to {save_path}")

    dataset, eval_func = load_longproc_data(config.dataset, config.dataset_dir)
    print(f"Loaded dataset: {config.dataset} with {len(dataset)} samples")

    model, tokenizer = load_model(config)

    done = defaultdict(int)

    if config.resume and os.path.exists(save_path):
        print(f"Resuming from {save_path}")
        # get what is already in the file
        with open(save_path, "r") as f:
            # get number of rollouts already done for each sample
            for line in f.readlines():
                example = json.loads(line)
                sample_idx = example.get("question_id", -1)
                done[sample_idx] += 1
        fout = open(save_path, "a")
    else:
        fout = open(save_path, "w")

    n_samples = config.n_samples if config.n_samples != 8 else 2

    for i, d in tqdm(list(enumerate(dataset))):
        qid = f"{config.dataset}_{i}"
        print(f"Processing sample {i} with question_id {qid}, done {done[qid]} / {n_samples}")

        if done[qid] >= n_samples:
            continue

        prompt = apply_chat_template([d["input_prompt"]], tokenizer)
        tokenized_prompt = tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=True,
        ).to("cuda")

        prompt_length = tokenized_prompt.input_ids.shape[1]
        max_length = prompt_length + config.max_gen_length

        while done[qid] < n_samples:
            n_return_sequences = min(config.max_return_sequences, n_samples - done[i])
            if config.method == 'trimkv':
                past_key_values = TrimKVCache(
                    memory_size=config.kv_budget,
                    buffer_size=config.buffer_size,
                    device="cuda",
                )
            else:
                past_key_values = None

            try:
                prediction = model.generate(
                    **tokenized_prompt,
                    max_length=max_length,
                    do_sample=config.do_sample,
                    num_beams=1,
                    past_key_values=past_key_values,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    num_return_sequences=n_return_sequences,
                )[0, prompt_length:]
                prediction = tokenizer.decode(
                    prediction, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
            except Exception as e:
                print(f"Error during generation: {e}")
                prediction = ""

            metrics, additional_info = eval_func(prediction, d)

            out_dict = {
                "question_id": qid,
                "prediction": prediction,
                "metrics": metrics,
                "additional_info": additional_info,
            }

            # print(json.dumps(out_dict), flush=True)
            fout.write(json.dumps(out_dict, ensure_ascii=False) + "\n")
            fout.flush()
            done[qid] += 1

    fout.close()

    # final results
    with open(save_path, "r") as f:
        eval_metrics = defaultdict(list)
        metric_names = set()
        for line in f.readlines():
            example = json.loads(line)
            eval_metrics[example["question_id"]].append(example["metrics"])
            metric_names.update(example["metrics"].keys())

        if len(eval_metrics) > 0:
            print(f"Average metrics over {len(eval_metrics)} examples:")
            print(f"Num samples per example: {[len(v) for v in eval_metrics.values()]}")
            # for each example, average over samples
            eval_metrics_mean = {}
            for k, v in eval_metrics.items():
                mean_metrics = {}
                for metric in metric_names:
                    mean_metrics[metric] = sum([m[metric] for m in v]) / len(v)
                eval_metrics_mean[k] = mean_metrics

            avg_metrics = {
                "runname": name,
                "num_samples": len(eval_metrics),
            }
            for metric in metric_names:
                avg_metrics[metric] = sum([m[metric] for m in eval_metrics_mean.values()]) / len(eval_metrics_mean)

            with open(os.path.join(output_dir, "summary.txt"), "a") as f:
                print(json.dumps(avg_metrics, indent=4), flush=True)
                f.write(json.dumps(avg_metrics) + '\n')


if __name__ == '__main__':
    load_dotenv()
    fire.Fire(main)
    # main()
