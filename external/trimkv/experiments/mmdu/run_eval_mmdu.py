import os
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from dotenv import load_dotenv
import fire
from openai import OpenAI
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
    model_name: str = field(default="Qwen/Qwen3-VL-32B-Instruct", metadata={"help": "Model name"})
    rerun: bool = field(default=False, metadata={"help": "Whether to rerun the evaluation"})
    input_file: str = field(default="", metadata={"help": "File to be scored"})
    dataset_dir: str = field(default="./data/", metadata={"help": "Path to the benchmark dataset"})
    retries: int = field(default=3, metadata={"help": "Number of retries for API calls"})
    inference_backend: str = field(default="gemini", metadata={"help": "Inference backend: vllm or transformers"})
    start_idx: int = field(default=0, metadata={"help": "Start index for evaluation"})
    end_idx: int = field(default=None, metadata={"help": "End index for evaluation"})

    # generation parameters
    max_new_tokens: int = field(default=8096, metadata={"help": "Maximum number of new tokens to generate"})
    do_sample: bool = field(default=False, metadata={"help": "Whether to use sampling during generation"})


def init_transformer_model(model_name_or_path):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor 
    import torch 

    print("Initializing transformer model and processor...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        padding_side="left",
    )
    # update_processor_pixels(processor, config)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    
    return model, processor

def transformer_infer(messages, model=None, processor=None, config=None):
    from qwen_vl_utils import process_vision_info

    
    text_inputs = processor.apply_chat_template(messages,tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=text_inputs, images=image_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    prefill_length = inputs['input_ids'].shape[1]
    output_ids = model.generate(**inputs, max_new_tokens=config.max_new_tokens, do_sample=config.do_sample, use_cache=True)
    model_generate_output = processor.tokenizer.batch_decode(output_ids[:, prefill_length:], skip_special_tokens=True)
    return model_generate_output


def infer(query, image_paths, api_key='EMPTY', base_url='http://localhost:5022/v1', config=None, model=None, processor=None):
    # 1. Setup Clients
    if config.inference_backend == "gemini":
        # Initialize Gemini Client
        client = genai.Client(api_key=GEMINI_API_KEY) # Use the passed api_key or global GEMINI_API_KEY
    else:
        client = OpenAI(api_key=api_key, base_url=base_url)
    
    # 2. Prepare Images
    # We use a new variable 'processed_images' to avoid overwriting the input 'image_paths'
    processed_images = []
    
    # Specific list for Gemini (needs PIL objects)
    gemini_images = [] 

    for image_path in image_paths:
        if config.inference_backend == "transformers":
            processed_images.append({"type": "image", "image": image_path})
            
        elif config.inference_backend == "vllm":                
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                image_b64 = base64.b64encode(image_data).decode('utf-8')
                processed_images.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
        
        elif config.inference_backend == "gemini":
            # Gemini SDK prefers PIL Images or bytes directly
            processed_images.append(Image.open(image_path))

    print(f"Prepared {len(processed_images)} images for inference.")
    # print(f"processed_images: {processed_images}")
    # 3. Inference
    if config.inference_backend == "vllm":
        # Structure for OpenAI/vLLM
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    *processed_images,
                ],
            },
        ]
        response = client.chat.completions.create(
            model="Qwen3-VL-32B-Instruct", # Or use variable 'model'
            messages=messages,
            max_tokens=config.max_new_tokens,
        )
        assistant_response = response.choices[0].message.content

    elif config.inference_backend == "transformers":
        # Reconstruct messages for transformers if needed
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    *processed_images,
                ],
            },
        ]
        assistant_response = transformer_infer(messages, model=model, processor=processor, config=config)
        assistant_response = assistant_response[0]

    elif config.inference_backend == "gemini":
        # Gemini takes a flat list: [text, img1, img2...]
        contents = [query, *processed_images]
        response = client.models.generate_content(
            model="gemini-2.5-flash", # Updated to a valid public model name
            contents=contents,
        )
        assistant_response = response.text

    else:
        raise ValueError(f"Unknown inference backend: {config.inference_backend}")
    
    return assistant_response

def calculate_score(samples):
    total_scores = defaultdict(float)
    total_turn = 0
    
    for i, sample in enumerate(samples):
        # print(f"calculating score for sample {i}: {sample.keys()}")
        number_of_sample_turns = len(sample['result_dict'])
        total_turn += number_of_sample_turns
        for result in sample['result_dict']:
            for key, value in result.items():
                total_scores[key] += value


    print(f"Total scored turns: {total_turn} from total samples: {len(samples)}")
    # overall_averages = {key: total / total_turn for key, total in total_scores.items()}
    report = {
        "total": dict(total_scores),
        "mean": {key: total / total_turn for key, total in total_scores.items()},
        "count": total_turn,
    }
    return report

meta_prompt = """
You are an assistant skilled at evaluating the quality of creative text.
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. You'll need to assess the response on the following dimensions: Creativity, Richness, Visual Perception, Logical Coherence, Answer Accuracy and Image Relationship Understanding. We will provide you with a creative question and the AI model's response and a reference answer for your evaluation. As you begin your assessment, follow this process:
1. Evaluate the AI model's answers on different dimensions, pointing out its strengths or weaknesses in each dimension and assigning a score of 1 to 10 for each.
2. Finally, based on the assessments across dimensions, provide an overall score of 1 to 10 for the AI model's response.
3. Your scoring should be as stringent as possible and follow the scoring rules below:

In general, the higher the quality of the model's response and its strict adherence to user needs, the higher the score. Responses that do not meet user needs will receive lower scores.

Scoring rules:
Creativity:
Scores 1-2 when there is no innovation or uniqueness in the content.
Scores 3-4 when providing partially original content but with low creative quality.
Scores 5-6 when mostly creative but lacks significant novelty, with moderate quality.
Scores 7-8 when having novelty and high-quality content.
Scores 9-10 when highly novel and of exceptional quality compared to the reference answer.

Richness:
Scores 1-2 when lacking depth and breadth, with very limited information.
Scores 3-4 when limited in depth and breadth, with fewer explanations and examples, showing low diversity.
Scores 5-6 when limited in depth and breadth but provides basic necessary information.
Scores 7-8 when providing depth and useful additional information.
Scores 9-10 when providing exceptional depth, breadth, and high diversity compared to the reference answer.

Visual Perception:
Scores 1-2 when the description of the visual information in the image contains errors or is significantly inconsistent with the content of the image.
Scores 3-4 When the description of the visual information in the image reflects only a small amount of the image's information and contains some errors.
Scores 5-6 when the description of the visual information in the image includes the basic information of the image but contains minimal information.
Scores 7-8 when the description of the visual information in the image matches the image well and is rich in content, providing a substantial amount of information about the image.
Scores 9-10 when the description of the visual information in the image not only matches the image but also is more detailed and informative compared to the reference answer, providing more information about the image.

Logical Coherence:
Scores 1-2 when entirely incoherent, lacking any logic, and not matching the question or known information.
Scores 3-4 when somewhat coherent but with many logical errors or inconsistencies.
Scores 5-6 when mostly coherent, with few errors, but may struggle to maintain complete coherence in complex situations.
Scores 7-8 when excellent logical handling, very few errors.
Scores 9-10 when flawless logic, impeccable in handling complexity, and significantly higher logical coherence compared to the reference answer.

Answer Accuracy
Scores 1-2 when the answer is significantly inconsistent with the question or contains obvious errors.
Scores 3-4 when the answer is partially correct but contains some errors or is incomplete.
Scores 5-6 when the answer is basically correct but lacks details or is not sufficiently detailed.
Scores 7-8 when the answer is accurate and detailed, fully corresponding to the question.
Scores 9-10 when the answer is not only accurate and detailed but also provides additional useful information, exceeding expectations.

Image Relationship Understanding:
Scores 1-2 when there are significant errors or confusion in distinguishing and describing different images, unable to correctly identify and relate the content of the images.
Scores 3-4 when the description of different images reflects only minimal distinguishing information, contains some errors and confusion, and fails to clearly differentiate and relate the images.
Scores 5-6 when the description of different images includes basic distinguishing information, is able to correctly identify and relate the images in a basic manner, but the information provided is minimal and lacks detail.
Scores 7-8 when the description of different images is accurate and detailed, clearly distinguishing and relating the images, with rich content that points out the main commonalities and differences between the images.
Scores 9-10 when the description of different images is not only accurate and detailed but also provides richer information and analysis, clearly distinguishing and relating the images, more comprehensively pointing out the commonalities and differences between the images compared to the reference answer.

Overall Score:
Scores 1-2 when irrelevant to the question, factually incorrect, or generates harmful content.
Scores 3-4 when no serious errors, mostly harmless, but of low quality and does not meet requirements.
Scores 5-6 when basically meeting requirements but performing poorly in some dimensions, with moderate quality.
Scores 7-8 when performing well in all dimensions.
Scores 9-10 when fully addressing user questions and all requirements, significantly surpassing the reference answer.

Please remember, you must evaluate and explain before scoring. After your explanation for each dimension, add the score for that dimension. Finally, at the end of your response, in the format of the dictionary (including brackets), return all your scoring results, ensuring your scores are integers:
{'Dimension One': Score, 'Dimension Two': Score, ..., 'Overall Score': Score}, for example: {'Creativity': 9, 'Richness': 6, ..., 'Overall Score': 7}.\n
"""

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


def evaluate(**kwargs):
    config = EvalConfig()
    config.__dict__.update(kwargs)
    
    dirname = os.path.dirname(config.input_file)
    basename = os.path.basename(config.input_file)
    config.output_file = os.path.join(dirname, f"{basename}.evaluation")

    samples = []
    with open(config.input_file, "r") as f:
        for line in f.readlines():
            sample = json.loads(line)
            samples.append(sample)
    sample_id_to_idx = {sample["id"]: idx for idx, sample in enumerate(samples)}

    done = defaultdict(int)
    print(done)
    if not config.rerun and os.path.exists(config.output_file):
        print(f"Resuming from {config.output_file}, skipping already processed samples.")
        with open(config.output_file, "r") as f:
            for line in f.readlines():
                # try:
                cached_sample = json.loads(line)
                if 'result_dict' in cached_sample:
                    if all(d != {} for d in cached_sample['result_dict']):
                        done[cached_sample['id']] += 1
                        current_sample_idx = sample_id_to_idx[cached_sample['id']]
                        samples[current_sample_idx]['result_dict'] = cached_sample['result_dict'].copy()
                else:
                    done[cached_sample['id']] = 0
                # except Exception as e:
                #     print(f"Error loading cached sample: {e}")
                #     continue

        fout = open(config.output_file, "a")
    else:
        fout = open(config.output_file, "w")
    
    dataset = load_dataset(config)
    print(f"Number of samples: {len(dataset)}")

    if config.inference_backend == "transformers":
        model, processor = init_transformer_model(config.model_name)
    else:
        model, processor = None, None

    dataset_ids = {item["id"]: item for item in dataset}
    # print(dataset_ids.keys())
    for sample in samples:
        if done[sample['id']] > 0 or sample['id'] < config.start_idx or (config.end_idx is not None and sample['id'] >= config.end_idx):
            continue
        benchmark_sample = dataset_ids[sample["id"]]
        conv_model = sample["conversations"]
        images = benchmark_sample["image"]
        conv_benchmarks = benchmark_sample["conversations"]
        questions = []
        ground_truth_response = []
        model_response = []
        for i in conv_benchmarks:
            if i["from"] == "user":
                questions.append(i["value"])
            if i["from"] == "assistant":
                ground_truth_response.append(i["value"])
                
        for i in conv_model:
            if i["from"] == "assistant":
                model_response.append(i["value"])

        sample['result_dict'] = [{} for _ in range(len(questions))]
        for j in range(len(questions)):
            if sample['result_dict'][j] != {}:
                continue
            prompt = build_prompt(meta_prompt, questions[j], ground_truth_response[j], model_response[j])
            
            # print(f"{RED}input prompt: {prompt}{RESET}")
            print(f"{CYAN}Evaluating sample id {sample['id']}, question {j+1}/{len(questions)}{RESET}")
            result_dict = {}
            retries = config.retries
            while result_dict == {} and retries > 0:
                try:
                    start_time = time.time()
                    response = infer(prompt, images, config=config, model=model, processor=processor)
                    print(f"{GREEN}model response: {response}{RESET}")
                    print(f"{RED}generation time takes: {time.time() - start_time}{RESET}")
                    start_index = response.find('{')
                    end_index = response.rfind('}') + 1
                    dictionary_str = response[start_index:end_index]
                    result_dict = eval(dictionary_str)
                    print(result_dict)
                except Exception as e:
                    print({e})
                    result_dict = {}

                if result_dict == {}:
                    print(f"{YELLOW}Retrying to get valid response...{RESET}")
                    time.sleep(30)
                    retries -= 1
            
            if result_dict == {}:
                print(f"{RED}Failed to get valid response after retries. Skipping...{RESET}")

            sample['result_dict'][j] = result_dict
            print(f"{YELLOW}extracted scores: {result_dict}{RESET}")
        
        sample.pop('image')
        sample.pop('conversations')
        sample.pop('set')
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