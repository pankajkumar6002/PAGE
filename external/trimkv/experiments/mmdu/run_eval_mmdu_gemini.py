import os
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from dotenv import load_dotenv
import fire
from PIL import Image
from google.genai import types
import base64
import time
from utils import RED, GREEN, YELLOW, CYAN, RESET, load_dataset
from google import genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "EMPTY")

@dataclass
class EvalConfig:
    model_name: str = field(default="gemini-2.5-flash", metadata={"help": "Model name"})
    rerun: bool = field(default=False, metadata={"help": "Whether to rerun the evaluation"})
    eval_only: bool = field(default=False, metadata={"help": "Whether to only evaluate without generation"})
    input_file: str = field(default="", metadata={"help": "File to be scored"})
    dataset_dir: str = field(default="./data/", metadata={"help": "Path to the benchmark dataset"})
    retries: int = field(default=2, metadata={"help": "Number of retries for API calls"})
    start_idx: int = field(default=0, metadata={"help": "Start index for evaluation"})
    end_idx: int = field(default=None, metadata={"help": "End index for evaluation"})


def calculate_score(samples):
    total_scores = defaultdict(float)
    total_turn = 0
    
    for i, sample in enumerate(samples):
        # print(f"calculating score for sample {i}: {sample.keys()}")
        number_of_sample_turns = len(sample['result_dict'])
        total_turn += number_of_sample_turns
        for result in sample['result_dict']:
            for key, value in result.items():
                # print(result)
                total_scores[key] += value

    print(f"Total scored turns: {total_turn} from total samples: {len(samples)}")
    # overall_averages = {key: total / total_turn for key, total in total_scores.items()}
    report = {
        "total": dict(total_scores),
        "mean": {key: total / total_turn for key, total in total_scores.items()},
        "count": total_turn,
    }
    return report

def build_prompt(meta_prompt, question, reference, answer):
    question_begin_prompt = "[Question]"
    reference_begin_prompt = "[The Start of Reference Answer]"
    reference_end_prompt = "[The End of Reference Answer]"
    answers_begin_prompt = "[The Start of Assistant’s Answer]"
    answers_end_prompt = "[The End of Assistant’s Answer]"
    prompt = meta_prompt + question_begin_prompt + '\n' \
        + question + '\n\n' + reference_begin_prompt + '\n' \
            + reference + '\n' + reference_end_prompt + '\n\n' \
                + answers_begin_prompt + '\n' + answer + '\n' \
                    + answers_end_prompt
    return prompt

def finalize_result_file(result_file):
    basename = os.path.basename(result_file)
    dirname = os.path.dirname(result_file)
    finalized_file = os.path.join(dirname, f"{basename}.score")
    kept, total = 0, 0
    samples = []
    with open(result_file, "r") as fin, open(finalized_file, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            sample = json.loads(line)
            total += 1
            if "result_dict" in sample and all(d != {} for d in sample['result_dict']):
                kept += 1
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                samples.append(sample)
    print(f"Kept {kept} out of {total} samples.")
    print(f"save to {finalized_file}")
    return samples 


def infer(client, request, config=None):
    result_dict = {}
    retries = config.retries
    while result_dict == {} and retries > 0:
        try:
            start_time = time.time()
            assistant_response = client.models.generate_content(model=config.model_name, contents=request,)
            print(f"{RED}generation time takes: {time.time() - start_time}{RESET}")
            
            response = assistant_response.text
            print(f"{GREEN}model response (last few tokens): {response[-150:]}{RESET}")
            start_index = response.find('{')
            end_index = response.rfind('}') + 1
            dictionary_str = response[start_index:end_index]
            result_dict = eval(dictionary_str)
            print(result_dict)
            
        except Exception as e:
            print(f"Error: {e}")
            result_dict = {}

        if result_dict == {}:
            print(f"{YELLOW}Retrying {config.retries - retries}/{config.retries} to get valid response...{RESET}")
            time.sleep(20)
            retries -= 1
    
    if result_dict == {}: print(f"{RED}Failed to get valid response after {config.retries} retries. Skipping...{RESET}")
    return result_dict
    

def build_multistep_requests(sample, dataset_ids, meta_prompt):
    benchmark_sample = dataset_ids[sample["id"]]
    
    requests = []
    questions = []
    ground_truth_response = []
    model_response = []
    assert len(sample["conversations"]) == len(benchmark_sample["conversations"]), \
        f"Number of conversation turns do not match for sample id {sample['id']}: {len(sample['conversations'])} vs {len(benchmark_sample['conversations'])}"
    
    conversation_length = len(sample["conversations"])
    for i in range(conversation_length):
        if benchmark_sample["conversations"][i]["from"] == "user":
            questions.append(benchmark_sample["conversations"][i]["value"])
        if benchmark_sample["conversations"][i]["from"] == "assistant":
            ground_truth_response.append(benchmark_sample["conversations"][i]["value"])
        if sample["conversations"][i]["from"] == "assistant":
            model_response.append(sample["conversations"][i]["value"])
    
    processed_images = []
    for image_path in benchmark_sample["image"]:
        processed_images.append(Image.open(image_path))
    
    for j in range(len(questions)):
        prompt = build_prompt(meta_prompt, questions[j], ground_truth_response[j], model_response[j])
        requests.append([prompt, *processed_images])
    
    return requests

def evaluate(**kwargs):
    config = EvalConfig()
    config.__dict__.update(kwargs)
    
    dirname = os.path.dirname(config.input_file)
    basename = os.path.basename(config.input_file)
    if config.start_idx == 0:
        config.output_file = os.path.join(dirname, f"{basename}.evaluation")
    else:
        config.output_file = os.path.join(dirname, f"{basename}_{config.start_idx}_{config.end_idx}.evaluation")

    # buffer_file = config.output_file + ".buffer"
    
    unused_keys = ['image', 'conversations', 'set', 'generation_time', 'history']
    with open('meta_prompt.txt', 'r', encoding='utf-8') as file:
        meta_prompt = file.read()
    
    samples = []
    with open(config.input_file, "r") as f:
        for line in f.readlines():
            sample = json.loads(line)
            samples.append(sample)
    sample_id_to_idx = {sample["id"]: idx for idx, sample in enumerate(samples)}

    done = defaultdict(int)
    if not config.rerun and os.path.exists(config.output_file):
        print(f"Resuming from {config.output_file}, skipping already processed samples.")
        with open(config.output_file, "r") as f:
            for line in f.readlines():
                cached_sample = json.loads(line)
                if 'result_dict' in cached_sample:
                    # print(cached_sample['result_dict'])
                    current_sample_idx = sample_id_to_idx[cached_sample['id']]
                    
                    if 'result_dict' not in samples[current_sample_idx]:
                        samples[current_sample_idx]['result_dict'] = cached_sample['result_dict'].copy()
                    else:
                        if sum(d != {} for d in cached_sample['result_dict']) > sum(d != {} for d in samples[current_sample_idx]['result_dict']):
                            samples[current_sample_idx]['result_dict'] = cached_sample['result_dict'].copy()
                    if all(d != {} for d in cached_sample['result_dict']):
                        done[cached_sample['id']] += 1
                    else:
                        done[cached_sample['id']] = 0
                else:
                    done[cached_sample['id']] = 0

        fout = open(config.output_file, "a")
    else:
        fout = open(config.output_file, "w")
    
    dataset = load_dataset(config)
    print(f"Number of samples: {len(dataset)}")
    print(f"Number of evaluated samples: {sum(1 for v in done.values() if v > 0)}")
    print(f"Number of remaining samples to evaluate: {len(samples) - sum(1 for v in done.values() if v > 0)}")
    dataset_ids = {item["id"]: item for item in dataset}

    client = genai.Client(api_key=GEMINI_API_KEY)
    for sample in samples:
        if sample['id'] < config.start_idx or (config.end_idx is not None and sample['id'] >= config.end_idx) or config.eval_only:
            continue
        if done[sample['id']] > 0 and not config.rerun:
            print(f"{YELLOW}Skipping already processed sample id {sample['id']}{RESET}")
            continue
        requests = build_multistep_requests(sample, dataset_ids, meta_prompt)
        sample['result_dict'] = [{} for _ in range(len(requests))]
        
        for request_idx, request in enumerate(requests):
            if sample['result_dict'][request_idx] != {}:
                print(f"{YELLOW}Skipping already processed sample id {sample['id']}, question {request_idx+1}/{len(requests)}{RESET}")
                continue
            
            print(f"{CYAN}Evaluating sample id {sample['id']}, question {request_idx+1}/{len(requests)}{RESET}")
            result_dict = infer(client, request, config=config)
            sample['result_dict'][request_idx] = result_dict
            print(f"{YELLOW}extracted scores: {result_dict}{RESET}")
        
        for key in unused_keys: # to make output file compact
            sample.pop(key, None)
            
        fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
        fout.flush()
        done[sample['id']] += 1
        
    fout.close()
    
    samples = finalize_result_file(config.output_file)
    report = calculate_score(samples)

    with open(config.output_file + ".report", "w") as f:
        json.dump(report, f, indent=4)

    print('total turns:', report.pop("count"))

    for key, result in report.items():
        print(f"\n{GREEN}{key} scores:{RESET}")
        for metric, value in result.items():
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    load_dotenv()
    fire.Fire(evaluate)