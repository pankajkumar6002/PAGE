import json
import time
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
from generation_utils import batch_exist_generate, prepare_prompt
from utils import estimate_max_batch_size
from trimkv.cache_utils import TrimKVCache
from dotenv import load_dotenv



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
    dataset: str = field(default="longmemeval_s", metadata={"help": "Dataset to use for evaluation"})
    dataset_dir: str = field(default="./data", metadata={"help": "Path to dataset"})
    output_dir: str = field(default="./results", metadata={"help": "Output directory"})
    n_samples: int = field(default=1, metadata={"help": "Number of samples"})
    num_inspect: int = field(default=8, metadata={"help": "Number of inspected samples"})
    seed: int = field(default=42, metadata={"help": "Random seed"})
    eval_batch_size: int = field(default=1, metadata={"help": "Batch size for evaluation"})
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
    buffer_size: int = field(default=32, metadata={"help": "Buffer size for compression"})
    compress_strategy: str = field(default="alpha", metadata={"help": "Compression strategy"})
    lookahead_steps: int = field(default=1, metadata={"help": "Number of lookahead steps for scoring tokens in trimkv"})

    # for RKV compression # use default values from MInference for stability (because they tuned for 4096 budget)
    window_size: int = field(default=32, metadata={"help": "Window size for compression"})
    mix_lambda: float = field(default=0.1, metadata={"help": "Mix lambda for compression"})
    retain_ratio: float = field(default=0.2, metadata={"help": "Retain ratio for compression"})
    retain_direction: str = field(default="last", metadata={"help": "Retain direction for compression"})
    divide_method: str = field(default="step_length", metadata={"help": "Method to divide input"})
    divide_length: int = field(default=128, metadata={"help": "Length to divide input"})
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
        name = f"{self.method}-{self.kv_budget}b-{self.max_model_len}l-{self.lookahead_steps}lhs"

        if self.n_samples is not None:
            name += f"-{self.n_samples}nspl"

        name += f"_{self.seed}s"
        if not self.do_sample:
            name += "-greedy"

        return name

def decode(
    model,
    tokenizer,
    input_ids,
    max_length,
    extra_end_token_ids=[],
    past_key_values=None,
):
    if input_ids.dim() == 1:
        input_ids = input_ids[None, :]
    input_ids = input_ids.cuda()
    assert input_ids.size(0) == 1

    if isinstance(model.config.eos_token_id, int):
        model_eos_token_id = [model.config.eos_token_id]
    else:
        model_eos_token_id = model.config.eos_token_id

    end_token_ids = (
        extra_end_token_ids
        + [tokenizer.eos_token_id]
        + model_eos_token_id
    )
    logits = None

    for i in range(max_length):
        if i == 0:  # prefilling
            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    use_cache=True,
                    return_dict=True,
                    past_key_values=past_key_values,
                )
            logits, past_key_values = out.logits, out.past_key_values
        else:  # decoding
            with torch.no_grad():
                out = model(
                    input_ids=input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            logits, past_key_values = out.logits, out.past_key_values

        logits = logits[:, -1, :]
        word = logits.argmax(dim=-1)
        if word.item() in end_token_ids or (i == max_length - 1):
            break

        input_ids = torch.cat(
            (input_ids, word.to(input_ids.device).view(1, 1)), dim=-1
        )

    return input_ids



def prefile_and_generate(
    model,
    tokenizer,
    chat_messages,
    max_length,
    past_key_values=None,
):
    model.eval()
    # first prefile the past key values with the chat messages
    model.is_compressing = True

    for msg in chat_messages[:-1]:
        tokenized_msg = tokenizer(msg, return_tensors="pt").to(model.device)
        print(f"Prefilling {tokenized_msg['input_ids'].shape[1]} tokens")
        with torch.no_grad():
            outputs = model(
                **tokenized_msg,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            print("cache length:", past_key_values.get_cache_length())

    model.is_compressing = None # leave it to the method's default
    tokenized_msg = tokenizer(chat_messages[-1], return_tensors="pt").to(model.device)
    # generate the response
    output = decode(
        model,
        tokenizer,
        tokenized_msg['input_ids'],
        max_length=max_length,
        extra_end_token_ids=[],
        past_key_values=past_key_values,
    )
    prefill_length = tokenized_msg['input_ids'].shape[1]
    output = tokenizer.decode(output[0, prefill_length:], skip_special_tokens=True)
    return output


def main(**kwargs):
    config = Config()
    config.update_from_dict(kwargs)
    set_seed(config.seed)

    if config.kv_budget is None or config.method.lower() == "fullkv":
        config.kv_budget = config.max_model_len

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
                sample_idx = example.get("question_id", -1)
                done[sample_idx] += 1
        fout = open(save_path, "a")
    else:
        fout = open(save_path, "w")


    data_path = os.path.join(config.dataset_dir, f"{config.dataset}.json")
    try:
        in_data = json.load(open(data_path))
    except:
        in_data = [json.loads(line) for line in open(data_path).readlines()]

    model, tokenizer, cache_creator = load_model(config)
    tokenizer_backend = 'huggingface'

    assert config.eval_batch_size == 1, "Batch size must be 1 for evaluation"

    for i in tqdm(range(len(in_data))):
        # Ttruncate the retrieval part of the prompt such that the context length never exceeds
        entry = in_data[i]

        gen_length = config.gen_length
        if gen_length is None:
            gen_length = 500 if not config.cot else 800
        max_retrieval_length = config.max_model_len - gen_length - 1000

        if config.con:
            raise ValueError("Not supported yet")
        else:
            prompt = prepare_prompt(entry, config.retriever_type, config.topk_context, config.useronly=='true',
                                    config.history_format, config.cot=='true', 
                                    tokenizer=tokenizer, tokenizer_backend=tokenizer_backend, max_retrieval_length=max_retrieval_length,
                                    merge_key_expansion_into_value=config.merge_key_expansion_into_value)

        # prompt = prompt[:50000] + prompt[-50000:]
        messages = [
            {"role": "user", "content": prompt}
        ]
        chat_messages = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # enable_thinking=True,
        )

        # sessions = chat_messages.split("### Session ")
        # for i, session in enumerate(sessions):
        #     if i > 0:
        #         sessions[i] = "### Session " + session

        sessions = [chat_messages]
        # for the last session, split by `Current Date:`
        last_session = sessions[-1]
        if "Current Date:" in last_session:
            parts = last_session.split("Current Date:")
            sessions[-1] = parts[0]
            sessions.append("Current Date:" + parts[1])
        print(f"Total sessions: {len(sessions)}")

        print(json.dumps({'question_id': entry['question_id'], 'question': entry['question'], 'answer': entry['answer']}, indent=4), flush=True)

        while done[entry['question_id']] < config.n_samples:
            past_key_values = cache_creator(model, max_model_len=config.max_model_len)

            start_time = time.time()
            output = prefile_and_generate(
                model,
                tokenizer,
                chat_messages=sessions,
                max_length=gen_length,
                past_key_values=past_key_values,
            )
            end_time = time.time()
            print(f"Generation time: {end_time - start_time:.2f}s")

            torch.cuda.empty_cache()

            answer = output.strip()
            out_dict = {'question_id': entry['question_id'], 'hypothesis': answer}

            print(json.dumps(out_dict), flush=True)
            fout.write(json.dumps(out_dict, ensure_ascii=False) + "\n")
            fout.flush()
            done[entry['question_id']] += 1

    fout.close()

if __name__ == "__main__":
    load_dotenv()
    fire.Fire(main)
