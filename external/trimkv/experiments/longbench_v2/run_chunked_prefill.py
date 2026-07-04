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
from transformers import DynamicCache
from dotenv import load_dotenv



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()


def prepare_prompt(item, config):
    context = item['context']
    if config.rag > 0:
        template = template_rag
        retrieved = item["retrieved_context"][:config.rag]
        retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
        context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
    elif config.no_context:
        template = template_no_context
    elif config.cot:
        template = template_0shot_cot
    else:
        template = template_0shot
    prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
    return prompt

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def extract_answer_from_cot(model, tokenizer, cot_response, item, config):
    prompt = template_0shot_cot_ans.replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', cot_response)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda")
    input_length = inputs['input_ids'].size(1)

    output = model.generate(
        **inputs,
        max_length=input_length + 100,
        do_sample=False,
        num_beams=1,
        num_return_sequences=1,
    )
    decoded = tokenizer.batch_decode(output[:, input_length:], skip_special_tokens=True)[0].strip()

    return decoded


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
                # To avoid weird behavior, we extend the first chunk to ompe + 1 so that the model always uses long RoPE
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


@dataclass
class Config:
    # experiment parameters
    dataset: str = field(default="longbench_v2", metadata={"help": "Dataset to use for evaluation"})
    output_dir: str = field(default="./results", metadata={"help": "Output directory"})
    n_samples: int = field(default=1, metadata={"help": "Number of samples"})
    num_inspect: int = field(default=8, metadata={"help": "Number of inspected samples"})
    seed: int = field(default=42, metadata={"help": "Random seed"})
    eval_batch_size: int = field(default=1, metadata={"help": "Batch size for evaluation"})
    max_return_sequences: int = field(default=1, metadata={"help": "Max return sequences"})
    resume: bool = field(default=True, metadata={"help": "Whether to resume from previous run"})
    gen_length: int = field(default=None, metadata={"help": "Generation length"})
    limit: int = field(default=None, metadata={"help": "Limit number of samples to eval"})

    rag: int = field(default=0, metadata={"help": "Number of retrieved chunks to use, 0 means no retrieval"})
    cot: bool = field(default=True, metadata={"help": "Whether to use chain-of-thought prompting"})
    no_context: bool = field(default=False, metadata={"help": "Whether to use no context at all"})

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

def run(config):
    output_dir = os.path.join(config.output_dir, f"{config.dataset}/{config.model_name}")
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


    dataset = load_dataset('THUDM/LongBench-v2', split='train')

    model, tokenizer, cache_creator = load_model(config)

    assert config.eval_batch_size == 1, "Batch size must be 1 for evaluation"

    for i in tqdm(range(len(dataset))):
        print(f"Processing sample {i}", flush=True)
        if config.limit is not None and i >= config.limit:
            print("Reached limit, stopping evaluation")
            break
        # Ttruncate the retrieval part of the prompt such that the context length never exceeds
        item = dataset[i]
        if item['length'] == 'long':
            # only evaluate short and medium samples
            print("Skipping long sample")
            continue

        prompt = prepare_prompt(item, config)
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        chat_messages = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # enable_thinking=True,
        )

        tokenized_prompts = tokenizer(
            [chat_messages],
            padding="longest",
            return_tensors="pt",
            add_special_tokens=True,
        ).to("cuda")

        prefill_length = tokenized_prompts["attention_mask"].sum(dim=1).tolist()[0]
        if prefill_length >= config.max_model_len:
            print(f"Skipping sample {i} due to prompt length {prefill_length} exceeding max model length {config.max_model_len}")
            continue

        print(json.dumps({'_id': item['_id'], 'question': item['question'], 'answer': item['answer'], 'length': item['length']}, indent=4), flush=True)
        print(f"Prompt length: {prefill_length}")

        while done[item['_id']] < config.n_samples:
            n_return_sequences = min(config.max_return_sequences, config.n_samples - done[i])
            max_gen_length = 1024 if config.cot else 128

            start_time = time.time()
            try:
                if config.method.lower() != 'locret':
                    batch_outputs = chunk_prefill_and_generate(
                        model,
                        tokenizer,
                        cache_creator,
                        tokenized_prompts,
                        config,
                        max_gen_length,
                    )
                else:
                    batch_outputs = locret_chunk_prefill_and_generate(
                        model,
                        tokenizer,
                        cache_creator,
                        tokenized_prompts,
                        config,
                        max_gen_length,
                    )

                end_time = time.time()
                print(f"Generation time: {end_time - start_time:.2f}s")

                # catch OOM errors
            except torch.cuda.OutOfMemoryError:
                print("CUDA out of memory, skipping this sample")
                torch.cuda.empty_cache()
                batch_outputs = ["CUDA OOM"] * n_return_sequences

            torch.cuda.empty_cache()

            for j in range(len(batch_outputs)):
                if config.cot:
                    cot_response = batch_outputs[j].strip()
                    # response = extract_answer_from_cot(model, tokenizer, cot_response, item, config) if cot_response != 'CUDA OOM' else "CUDA OOM"
                else:
                    cot_response = batch_outputs[j].strip()
                    # response = batch_outputs[j].strip()

                # answer = extract_answer(response)
                out_dict = {
                    '_id': item['_id'],
                    'question': item['question'],
                    # 'predicted_answer': answer,
                    'gold_answer': item['answer'],
                    # 'response': response,
                    'cot_response': cot_response,
                    # 'acc': int(answer == item['answer']) if answer is not None else 0,
                    'length': item['length'],
                    'difficulty': item['difficulty'],
                }

                print(json.dumps(out_dict), flush=True)
                fout.write(json.dumps(out_dict, ensure_ascii=False) + "\n")
                fout.flush()
                done[item['_id']] += 1

    fout.close()
    return save_path


def evaluate(save_path, config):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # load Qwen3-4B as the evaluator
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507").cuda()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

    examples = []
    with open(save_path, "r") as f:
        for line in f.readlines():
            example = json.loads(line)
            examples.append(example)

    eval_path = save_path.replace(".jsonl", "_eval.jsonl")
    done = set()
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            for line in f.readlines():
                example = json.loads(line)
                done.add(example['_id'])
    fout = open(eval_path, "a")

    dataset = load_dataset('THUDM/LongBench-v2', split='train')
    dataset_dict = {item['_id']: item for item in dataset}

    for example in tqdm(examples):
        if example['_id'] in done:
            continue

        item = dataset_dict[example['_id']]
        cot_response = example['cot_response']
        if config.cot:
            response = extract_answer_from_cot(model, tokenizer, cot_response, item, config)
        else:
            response = cot_response

        answer = extract_answer(response)
        example['predicted_answer'] = answer
        example['acc'] = int(answer == example['gold_answer']) if answer is not None else 0
        fout.write(json.dumps(example, ensure_ascii=False) + "\n")
        fout.flush()
        done.add(example['_id'])

    fout.close()
    return eval_path


def main(**kwargs):
    config = Config()
    config.update_from_dict(kwargs)

    set_seed(config.seed)

    if config.kv_budget is None or config.method.lower() == "fullkv":
        config.kv_budget = config.max_model_len

    save_path = run(config)
    eval_path = evaluate(save_path, config)

    # final stats
    results = defaultdict(list)

    with open(eval_path, "r") as f:
        for line in f.readlines():
            example = json.loads(line)
            results['acc'].append(example['acc'])
            results[f"acc_{example['length']}"].append(example['acc'])
            results[f"acc_{example['difficulty']}"].append(example['acc'])

    result_summary = {}
    result_summary['run_name'] = config.run_name
    for k, v in results.items():
        print(f"{k}: {np.mean(v):.4f} ({len(v)} samples)")
        result_summary[k] = np.mean(v)

    summary_path = os.path.join(config.output_dir, "summary.txt")
    with open(summary_path, "a") as f:
        f.write(json.dumps(result_summary) + "\n")
        f.flush()


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(main)
