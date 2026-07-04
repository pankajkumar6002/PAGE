from tqdm import tqdm
import json 
from load_model import load_model
from dataclasses import dataclass, field, fields
from typing import Union
import torch 
from dotenv import load_dotenv
import fire 
import random 
import numpy as np 
from collections import defaultdict
import os 
from huggingface_hub import hf_hub_download
from qwen_vl_utils import process_vision_info
import re 
import time 
from utils import RED, GREEN, YELLOW, CYAN, RESET, set_seed, load_dataset, find_gap_between_turns
from run_eval_mmdu import evaluate
from transformers import DynamicCache

@dataclass
class Config:
    method: str = field(default="vanilla", metadata={"help": "Compression method to use"})
    model_path: str = field(default="Qwen/Qwen3-VL-8B-Thinking", metadata={"help": "Model name"})
    model_args: str = field(default="", metadata={"help": "For compatibility with cli args"})
    dataset_dir: str = field(default="./data", metadata={"help": "Path to dataset"})
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Attention implementation to use"})
    max_model_len: int = field(default=131072, metadata={"help": "Maximum model length"})
    # max_model_len: int = field(default=131072, metadata={"help": "Maximum model length"})
    batch_size: Union[int, str] = field(default=1, metadata={"help": "Batch size for evaluation"})
    max_batch_size: Union[int, str] = field(default=None, metadata={"help": "Max batch size for evaluation"})
    n_samples: int = field(default=None, metadata={"help": "Number of samples for experiments"})
    do_sample: int = field(default=False, metadata={"help": "Whether to use sampling during generation"})
    rerun: bool = field(default=False, metadata={"help": "Whether to rerun existing samples"})
    disable_thinking: bool = field(default=False, metadata={"help": "Whether to disable <think> tokens during generation"})
    start_idx: int = field(default=0, metadata={"help": "Start index for evaluation"})
    end_idx: int = field(default=None, metadata={"help": "End index for evaluation"})
    
    min_pixels: int = field(default=384 * 28 * 28, metadata={"help": "Minimum pixels for image processor"})
    max_pixels: int = field(default=2048 * 28 * 28, metadata={"help": "Maximum pixels for image processor"})
    video_min_frames: int = field(default=4, metadata={"help": "Minimum frames for video processor"})
    video_max_frames: int = field(default=8, metadata={"help": "Maximum frames for video processor"})
    video_min_pixels: int = field(default=144 * 28 * 28, metadata={"help": "Minimum pixels per frame for video processor"})
    video_max_pixels: int = field(default=576* 28 * 28, metadata={"help": "Maximum pixels per frame for video processor"})
    video_fps: int = field(default=2, metadata={"help": "FPS for video processor"})
    output_dir: str = field(default="./results", metadata={"help": "Output directory"})
    
    # gen kwargs
    max_new_tokens: int = field(default=2048, metadata={"help": "Maximum new tokens to generate"})
    temperature: float = field(default=0, metadata={"help": "Temperature for generation"})
    top_p: float = field(default=None, metadata={"help": "Top-p for generation"})
    num_beams: int = field(default=1, metadata={"help": "Number of beams for generation"})
    
    # for KV compression methods
    kv_budget: int = field(default=512, metadata={"help": "KV budget for compression"})

    # for trimkv method
    download_from: str = field(default="wandb", metadata={"help": "Where to download the model from"})
    buffer_size: int = field(default=32, metadata={"help": "Buffer size for compression"})
    fixed_kv_budget: bool = field(default=True, metadata={"help": "Set to False for a fair comparison with visual token prunning methods. If set to False, the actual KV budget will be determined dynamically based on the text length, which is num_text_tokens + kv_budget."})
    compress_strategy: str = field(default="alpha", metadata={"help": "Compression strategy to use"})
    strategy: str = field(default="fixed_budget", metadata={"help": "Compression strategy to use, [fixed_budget, threshold]"})
    alpha_threshold: float = field(default=0.8, metadata={"help": "Alpha threshold for compression when strategy is set to threshold"})

    # for RKV compression
    window_size: int = field(default=8, metadata={"help": "Window size for compression"})
    mix_lambda: float = field(default=0.1, metadata={"help": "Mix lambda for compression"})
    retain_ratio: float = field(default=0.2, metadata={"help": "Retain ratio for compression"})
    retain_direction: str = field(default="last", metadata={"help": "Retain direction for compression"})
    divide_method: str = field(default="step_length", metadata={"help": "Method to divide input"})
    divide_length: int = field(default=32, metadata={"help": "Length to divide input"})
    compression_content: str = field(default="all", metadata={"help": "Content to compress"})
    
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})
    # for streamingllm
    first_tokens: int = field(default=4, metadata={"help": "First tokens for compression"})

    def update_from_dict(self, args):
        for k, v in args.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown argument: {k}")
            setattr(self, k, v)
        return self
    
    @property
    def run_name(self):
        name = f"{self.method}-{self.kv_budget}b-{self.max_model_len}l-{self.max_new_tokens}t"
        return name
    
    @property
    def model_name(self):
        model_path = self.model_path.strip('/')
        model_name = os.path.basename(model_path).replace("/", "_").replace("-", "_").replace(",", "_").replace(":", "_")
        return model_name
    
    @property
    def model_type(self):

        model_path = self.model_path.lower().replace("-", "_").replace(" ", "_").replace(".", "_")
        if 'qwen3' in model_path:
            return 'qwen3_vl'
        elif 'llava_1_5' in model_path:
            return 'llava_hf'
        elif 'qwen2_5' in model_path:
            return 'qwen2_5_vl'
        else:
            raise ValueError(f"Unknown base model in path: {model_path}")


def decode(
    model,
    processor,
    prepare_inputs_for_generation_fn,
    messages,
    max_length=1024,
    extra_end_token_ids=[],
    past_key_values=None,
    num_images_this_turn=0,
    seen_tokens=0,
    turn_index=0,
    disable_thinking=True,
    turn_gap_str="<im_end>\n",
):
    model.eval()
    if turn_index == 0:
        text_inputs = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text_inputs = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text_inputs = turn_gap_str + text_inputs
    print(text_inputs)

    if disable_thinking:
        if not text_inputs.strip().strip('\n').endswith("<think>"):
            raise ValueError("This is not a reasoning model but you want to disable thinking?")
        text_inputs += "\n</think>\n\n"

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=text_inputs, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    
    if turn_index == 0:
        # This function will prepare past_key_values for the first turn
        inputs = prepare_inputs_for_generation_fn(model, inputs)
    else:
        # For subsequent turns, we use past_key_values from previous turns
        inputs['past_key_values'] = past_key_values

    inputs.pop('attention_mask')
    inputs = inputs.to(model.device)
    input_ids = inputs.pop('input_ids')

    prefill_length = input_ids.shape[1]
    past_key_values = inputs.pop('past_key_values', None)
    end_token_ids = (processor.tokenizer.eos_token_id,) + tuple(extra_end_token_ids)

    with torch.no_grad():
        for i in range(max_length + 1):
            model_kwargs = dict(
                use_cache=True,
                return_dict=True,
                past_key_values=past_key_values,
            )
            # NOTE!! A bit dangerous here, RKV implementation uses DynamicCache but newer version of transformers will return num cached tokens via get_seq_length() instead of all seen tokens. For RKV, we are using an adapted version of DynamicCache that implements get_seq_length()
            # Make sure that get_seq_length() will return seen tokens, not current cache length
            # seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            # Oke, just to be safe, we will track seen_tokens separately and pass it in the function arguments
        
            if i == 0:
                if num_images_this_turn > 0:
                    model_kwargs.update(inputs) # only pass image at first step
                model_kwargs['cache_position'] = torch.arange(seen_tokens, prefill_length + seen_tokens, device=model.device)
                model_kwargs['input_ids'] = input_ids
                seen_tokens += prefill_length
            else:
                model_kwargs['cache_position'] = torch.tensor([seen_tokens], device=model.device)
                model_kwargs['input_ids'] = input_ids[:, -1:]
                seen_tokens += 1

            out = model(**model_kwargs)
            logits, past_key_values = out.logits, out.past_key_values
            logits = logits[:, -1, :]
            word = logits.argmax(dim=-1)

            if word.item() in end_token_ids: break
            if i < max_length:
                input_ids = torch.cat((input_ids, word.to(input_ids.device).view(1, 1)), dim=-1)
    
    output_ids = input_ids[0, prefill_length:]
    output = processor.tokenizer.decode(output_ids, skip_special_tokens=True)
    return output, past_key_values, seen_tokens

def split_question_into_content(question, image_pool):
    text_parts = re.split(r"(<ImageHere>|<image>)", question)
    content = []
    for seg in text_parts:
        if seg == "<ImageHere>":
            if not image_pool:
                raise ValueError(
                    "Number of <ImageHere> placeholders exceeds the number of provided images"
                )
            content.append(image_pool.pop(0))
        elif seg.strip():
            content.append({"type": "text", "text": seg.strip()})
        else:
            print(seg, text_parts)
            raise ValueError("Empty text segment found in the question.")
    return content


def run(**kwargs):
    config = Config()
    config.update_from_dict(kwargs)
    set_seed(config.seed)
    
    output_dir = os.path.join(config.output_dir, f"{config.model_name}")
    if config.start_idx > 0:
        save_path = os.path.join(output_dir, f"{config.run_name}-from{config.start_idx}-to{config.end_idx}.json")
    else:
        save_path = os.path.join(output_dir, f"{config.run_name}.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Saving results to {save_path}")
    done = defaultdict(int)
    if not config.rerun and os.path.exists(save_path):
        print(f"Resuming from {save_path}")
        with open(save_path, "r") as f:
            for line in f.readlines():
                example = json.loads(line)
                sample_idx = example.get("id", -1)
                done[sample_idx] += 1
        fout = open(save_path, "a")
    else:
        fout = open(save_path, "w")

    dataset = load_dataset(config)
    model, processor, prepare_inputs_for_generation_fn = load_model(config)
    model.config.memory_size = config.kv_budget
    model.config.text_config.memory_size = config.kv_budget
    turn_gap_str = find_gap_between_turns(processor)

    for sample_idx, sample in tqdm(enumerate(dataset)):
        if sample_idx < config.start_idx:
            continue
        if config.end_idx is not None and sample_idx >= config.end_idx:
            break
        record_data = sample.copy()
        if done[record_data['id']] > 0:
            print(f"Sample {sample_idx} already done, skipping.")
            continue
        record_data['generation_time'] = {record_data['id']: []}
        img_paths = sample["image"]
        turn_questions = [msg["value"] for msg in sample["conversations"] if msg["from"] == "user"]
        
        pic_index = 0
        seen_tokens = 0
        past_key_values = None # Initialize to None at the start of each sample
        history = []
        for turn_index, turn_question in enumerate(turn_questions):
            num_images_this_turn = turn_question.count('<ImageHere>')
            tagged_images = img_paths[pic_index : pic_index + num_images_this_turn]
            image_pool = [{"type": "image", "image": img} for img in tagged_images]
            content = split_question_into_content(turn_question, image_pool)
            message = {"role": "user", "content": content}
            history.append(message)

            start_time = time.time()
            print(f"\n{GREEN}Q{sample_idx}-Turn{turn_index}: {turn_question}{RESET}")
            output, past_key_values, seen_tokens = decode(
                model,
                processor,
                prepare_inputs_for_generation_fn,
                [message],
                max_length=config.max_new_tokens,
                extra_end_token_ids=[],
                seen_tokens=seen_tokens,
                past_key_values=past_key_values,
                num_images_this_turn=num_images_this_turn,
                turn_index=turn_index,
                disable_thinking=config.disable_thinking,
                turn_gap_str=turn_gap_str)

            end_time = time.time()
            print(f"Generation time for Q{sample_idx}-Turn{turn_index}: {end_time - start_time:.2f} seconds")
            record_data['generation_time'][record_data['id']].append(end_time - start_time)
            
            history.append({"role": "assistant", "content": output})
            record_data["conversations"][turn_index*2+1]["value"] = output
            
            print(f"\n{YELLOW}A (forward partial):{RESET} {output}")
            # print(f"\n{CYAN}A (full):{RESET} {full_output}")
            # print(f"\n{GREEN}A (generate full):{RESET} {generate_output}")
        record_data['history'] = history

        torch.cuda.empty_cache()
        fout.write(json.dumps(record_data, ensure_ascii=False) + "\n")
        fout.flush()
        done[record_data['id']] += 1

    fout.close()
    return save_path

def main(**kwargs):
    run_eval = kwargs.pop("run_eval", False)
    save_path = run(**kwargs)

    if not run_eval:
        return

    # inference_backend = kwargs.get("inference_backend", "transformers")
    # evaluate(
    #     input_file=save_path,
    #     dataset_dir=kwargs.get("dataset_dir", "./data"),
    #     rerun=kwargs.get("rerun", False),
    #     inference_backend=inference_backend,
    # )

if __name__ == "__main__":
    load_dotenv()
    fire.Fire(main)
