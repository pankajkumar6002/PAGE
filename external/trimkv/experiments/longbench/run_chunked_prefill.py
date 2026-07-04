import json
import re
import time
import fire
import os
import random
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from dataclasses import dataclass, field
from collections import defaultdict
from load_model import load_model
from dotenv import load_dotenv

from transformers import Phi3ForCausalLM, AutoTokenizer, Qwen3ForCausalLM


from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}



@dataclass
class Config:
    # experiment parameters
    dataset: str = field(default="hotpotqa", metadata={"help": "Dataset name"})
    output_dir: str = field(default="./results", metadata={"help": "Output directory"})
    n_samples: int = field(default=1, metadata={"help": "Number of samples"})
    num_inspect: int = field(default=8, metadata={"help": "Number of inspected samples"})
    seed: int = field(default=42, metadata={"help": "Random seed"})
    eval_batch_size: int = field(default=1, metadata={"help": "Batch size for evaluation"})
    max_return_sequences: int = field(default=1, metadata={"help": "Max return sequences"})
    resume: bool = field(default=True, metadata={"help": "Whether to resume from previous run"})
    gen_length: int = field(default=None, metadata={"help": "Generation length"})

    # generation parameters
    temperature: float = field(default=0.6, metadata={"help": "Temperature"})
    top_p: float = field(default=0.95, metadata={"help": "Top p"})
    do_sample: bool = field(default=False, metadata={"help": "Whether to do sampling"})

    # model parameters 
    method: str = field(default="fullkv", metadata={"help": "KV method (fullkv, rnsa, rkv, snapkv, streamingllm)"})
    model_path: str = field(default="microsoft/Phi-3-mini-128k-instruct", metadata={"help": "Model"})
    model_type: str = field(default="phi3-mini-128k", metadata={"help": "Model type"})
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Attention implementation"})
    max_model_len: int = field(default=131072, metadata={"help": "Max model length"})
    download_from: str = field(default='local', metadata={"help": "Where to download the model from, 'local' or 'wandb' or 'huggingface'"})

    # general parameters for compression
    kv_budget: int = field(default=None, metadata={"help": "KV budget for compression"})
    update_kv: bool = field(default=True, metadata={"help": "Whether to update KV"})
    compress_strategy: str = field(default="alpha", metadata={"help": "Compression strategy"})
    buffer_size: int = field(default=0, metadata={"help": "Buffer size for compression"})

    # following LocRet
    chunk_size: int = field(default=3072, metadata={"help": "Chunk size for chunked prefill"})
    stabilizers: int = field(default=2500, metadata={"help": "Number of stabilizers for LocRet"})

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
        if self.method == 'locret':
            name = f"{self.method}-{self.chunk_size}-{self.kv_budget}b-{self.stabilizers}bf-{self.max_model_len}l"
        else:
            name = f"{self.method}-{self.chunk_size}-{self.kv_budget}b-{self.buffer_size}bf-{self.max_model_len}l"

        if self.n_samples is not None:
            name += f"-{self.n_samples}nspl"

        name += f"_{self.seed}s"
        if not self.do_sample:
            name += "-greedy"

        return name


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "qwen" in model_name:
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif "phi3" in model_name:
        messages = [
            {"role": "user", "content": prompt}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif "llama3" in model_name:
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def prepare_prompt(item, tokenizer, prompt_format, model_name, max_length=16384, dataset_name="hotpotqa", device='cuda'):
    # following https://github.com/THUDM/LongBench/blob/main/LongBench/pred.py
    prompt = prompt_format.format(**item)
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized_prompt) > max_length:
        half = int(max_length/2)
        prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

    # if dataset_name not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
    prompt = build_chat(tokenizer, prompt, model_name)

    if "chatglm3" in model_name:
        if dataset_name in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        else:
            input = prompt.to(device)
    else:
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
    return input


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]

    # find the last </think> tag and remove everything before it
    think_end_tag = "</think>"
    if think_end_tag in response:
        last_think_end = response.rfind(think_end_tag)
        response = response[last_think_end + len(think_end_tag):].strip()
    return response


def vanilla_generate(
    model,
    tokenizer,
    cache_creator,
    inputs,
    config,
    max_gen,
):
    past_key_values = cache_creator(model)

    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_gen,
        temperature=config.temperature,
        top_p=config.top_p,
        do_sample=config.do_sample,
        past_key_values=past_key_values,
    )

    output_tokens = output[:, inputs["input_ids"].size(1):]
    batch_outputs = []
    for i in range(output_tokens.size(0)):
        decoded_output = tokenizer.decode(output_tokens[i], skip_special_tokens=True)
        batch_outputs.append(decoded_output)

    return batch_outputs


def chunk_prefill_and_generate(
    model,
    tokenizer,
    cache_creator,
    inputs,
    config,
    max_gen,
):
    chunk_size = config.chunk_size
    if hasattr(model.config, "rope_scaling") and isinstance(model.config.rope_scaling, dict):
        rope_type = model.config.rope_scaling.get("rope_type", model.config.rope_scaling.get("type"))
    else:
        rope_type = 'default'

    if 'phi3' in config.model_type:
        # crazy
        stop_tokens = ["<|end|>", "<|endoftext|>"]
        end_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in stop_tokens]
    else:
        if isinstance(model.config.eos_token_id, int):
            model_eos_token_id = [model.config.eos_token_id]
        else:
            model_eos_token_id = model.config.eos_token_id

        end_token_ids = (
            [tokenizer.eos_token_id]
            + model_eos_token_id
        )

    if hasattr(model.config, "original_max_position_embeddings"):
        ompe = model.config.original_max_position_embeddings
    else:
        ompe = model.config.max_position_embeddings

    assert inputs["input_ids"].size(0) == 1, "Batch size must be 1 for chunked prefill generation"
    inputs.pop("attention_mask", None)  # remove attention mask for chunked prefill
    input_ids = inputs.pop("input_ids")

    prefill_length = input_ids.size(1)

    with torch.no_grad():
        past_key_values = cache_creator(model)
        # this is stupid but do it for now
        past_key_values.sliding_window_size = config.buffer_size

        start_idx = 0
        while start_idx < prefill_length:
            end_idx = min(start_idx + chunk_size, prefill_length)
            if rope_type == 'longrope' and end_idx <= ompe + 1:
                # For models with LongRoPE, it will switch between short and long RoPE at ompe + 1
                # To avoid weird behavior, we extend the chunk to ompe + 1 so that the model always uses long RoPE
                end_idx = min(ompe + 1, prefill_length)
                # This only happens for the first chunk

            # # check if it's the last chunk, we set the sliding window size to 128 to mimic LocRet local size
            # if end_idx == prefill_length:
            #     past_key_values.sliding_window_size = 128

            outputs = model(
                input_ids=input_ids[:, start_idx:end_idx],
                cache_position=torch.arange(start_idx, end_idx).to(input_ids.device),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            last_logits = outputs.logits[ :, -1, :]
            next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

            start_idx = end_idx

        input_ids = torch.cat([input_ids, next_token], dim=-1)

        for _generation_step in range(max_gen):
            outputs = model(
                input_ids=input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            if config.do_sample:
                probs = torch.nn.functional.softmax(logits / config.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            if next_token.item() in end_token_ids:
                break

            input_ids = torch.cat([input_ids, next_token], dim=-1)

    output_tokens = input_ids[:, prefill_length:]
    batch_outputs = []
    for i in range(output_tokens.size(0)):
        decoded_output = tokenizer.decode(output_tokens[i], skip_special_tokens=True)
        batch_outputs.append(decoded_output)

    return batch_outputs


def locret_chunk_prefill_and_generate(
    model,
    tokenizer,
    cache_creator,
    inputs,
    config,
    max_gen,
):
    input_ids = inputs.pop("input_ids")

    if 'phi3' in config.model_type:
        # crazy
        stop_tokens = ["<|end|>", "<|endoftext|>"]
        end_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in stop_tokens]
    else:
        if isinstance(model.config.eos_token_id, int):
            model_eos_token_id = [model.config.eos_token_id]
        else:
            model_eos_token_id = model.config.eos_token_id

        end_token_ids = (
            [tokenizer.eos_token_id]
            + model_eos_token_id
        )

    prefill_length = input_ids.size(1)

    output_ids = cache_creator(
        model,
        input_ids,
        eos_token_ids=end_token_ids,
        max_new_tokens=max_gen,
        budget_size=config.kv_budget,
        stabilizers=config.stabilizers,
    )

    batch_outputs = []
    for i in range(output_ids.size(0)):
        decoded_output = tokenizer.decode(output_ids[i, prefill_length:], skip_special_tokens=True)
        batch_outputs.append(decoded_output)

    return batch_outputs


def main(**kwargs):
    config = Config()
    config.__dict__.update(kwargs)
    set_seed(config.seed)

    if config.kv_budget is None or config.method.lower() == "fullkv":
        config.kv_budget = config.max_model_len

    output_dir = os.path.join(config.output_dir, f"{config.model_name}/{config.dataset}/")
    save_path = os.path.join(output_dir, f"{config.run_name}.jsonl")
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
                sample_idx = example.get("_id", -1)
                done[sample_idx] += 1
        fout = open(save_path, "a")
    else:
        fout = open(save_path, "w")



    dataset = load_dataset('THUDM/LongBench', config.dataset, split='test', trust_remote_code=True)

    dataset2prompt = json.load(open("configs/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("configs/dataset2maxlen.json", "r"))
    model2path = json.load(open("configs/model2path.json", "r"))
    model2maxlen = json.load(open("configs/model2maxlen.json", "r"))

    prompt_format = dataset2prompt[config.dataset]
    max_gen = dataset2maxlen[config.dataset]
    max_length = model2maxlen[config.model_type]
    score_func = dataset2metric[config.dataset]

    model, tokenizer, cache_creator = load_model(config)
    model.eval()

    assert config.eval_batch_size == 1, "Batch size must be 1 for evaluation"

    for i in tqdm(range(len(dataset))):
        # Ttruncate the retrieval part of the prompt such that the context length never exceeds
        item = dataset[i]

        while done[item['_id']] < config.n_samples:
            inputs = prepare_prompt(
                item,
                tokenizer,
                prompt_format,
                config.model_type,
                max_length=max_length,
                dataset_name=config.dataset
            )

            start_time = time.time()
            # output = vanilla_generate(
            #     model,
            #     tokenizer,
            #     cache_creator,
            #     inputs,
            #     config,
            #     max_gen,
            # )
            # print("vanilla Output:", output)

            if config.method.lower() != 'locret':
                batch_outputs = chunk_prefill_and_generate(
                    model,
                    tokenizer,
                    cache_creator,
                    inputs,
                    config,
                    max_gen,
                )
            else:
                batch_outputs = locret_chunk_prefill_and_generate(
                    model,
                    tokenizer,
                    cache_creator,
                    inputs,
                    config,
                    max_gen,
                )

            end_time = time.time()
            print(f"Generation time: {end_time - start_time:.2f}s")

            torch.cuda.empty_cache()

            for j in range(len(batch_outputs)):
                response = batch_outputs[j].strip()
                pred = post_process(response, config.model_type)
                score = max([score_func(pred, ans) for ans in item['answers']]) if score_func is not None else -1.0

                out_dict = {
                    '_id': item['_id'],
                    'question': item['input'],
                    'pred': pred,
                    'gold_answer': item['answers'],
                    'all_classes': item['all_classes'],
                    'response': response,
                    'acc': score,
                    'length': item['length'],
                }

                print(json.dumps(out_dict), flush=True)
                fout.write(json.dumps(out_dict, ensure_ascii=False) + "\n")
                fout.flush()
                done[item['_id']] += 1

    fout.close()


    # final stats
    results = defaultdict(list)

    with open(save_path, "r") as f:
        for line in f.readlines():
            example = json.loads(line)
            results['acc'].append(example['acc'])

    result_summary = {}
    result_summary['run_name'] = config.run_name
    for k, v in results.items():
        print(f"{k}: {np.mean(v):.4f} ({len(v)} samples)")
        result_summary[k] = np.mean(v)
    
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "a") as f:
        f.write(json.dumps(result_summary) + "\n")
        f.flush()


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(main)
