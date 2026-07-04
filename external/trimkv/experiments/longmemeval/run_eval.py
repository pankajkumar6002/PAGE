import os
import argparse
from collections import defaultdict
import sys
import json
from tqdm import tqdm
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


model_zoo = {
    'llama-3.1-70b-instruct': ('meta-llama/Meta-Llama-3.1-70B-Instruct', 'local'),
    'qwen3-4b-instruct': ('Qwen/Qwen3-4B-Instruct-2507', 'local'),
    'qwen3-32b': ('Qwen/Qwen3-32B', 'local'),
    'gpt-4o-mini': ('gpt-4o-mini-2024-07-18', 'openai'),
    'gpt-4o': ('gpt-4o-2024-08-06', 'openai'),
}



def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response) 
    return prompt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('metric_model', type=str, help='Metric model to use. Supported: {}'.format(', '.join(model_zoo.keys())))
    parser.add_argument('hyp_file', type=str, help='Path to the hypothesis file (JSON or JSONL)')
    parser.add_argument('ref_file', type=str, help='Path to the reference file (JSON or JSONL)')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing result file')
    args = parser.parse_args()
    metric_model_short = args.metric_model
    hyp_file = args.hyp_file
    ref_file = args.ref_file

    verbose = True

    result_file = hyp_file + '.eval-results-{}'.format(metric_model_short)
    if os.path.exists(result_file) and not args.overwrite:
        print('Result file already exists. Use --overwrite to overwrite it:', result_file)
        exit()

    if metric_model_short not in model_zoo:
        print('Requested metric model is not supported:', metric_model_short)
        exit()
    metric_model_name, metric_model_source = model_zoo[metric_model_short]

    model = AutoModelForCausalLM.from_pretrained(
        metric_model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        metric_model_name,
        trust_remote_code=True,
        use_fast=False,
    )

    try:
        hypotheses = [json.loads(line) for line in open(hyp_file).readlines()]
    except:
        hypotheses = json.load(open(hyp_file))
    try:
        references = json.load(open(ref_file))
    except:
        references = [json.loads(line) for line in open(ref_file).readlines()]
    qid2qdata = {entry['question_id']: entry for entry in references}
    qid2qtype = {entry['question_id']: entry['question_type'] for entry in references}
    # count number of samples per type
    n_samples_per_type = defaultdict(int)
    for t in qid2qtype.values():
        n_samples_per_type[t] += 1
    print('Number of samples per type:', dict(n_samples_per_type))

    qtypes = set(list(qid2qtype.values()))
    qtype2acc = {t: [] for t in qtypes}

    with open(result_file, 'w') as out_f:
        logs = []
        for entry in tqdm(hypotheses):

            if entry['question_id'] not in qid2qtype:
                print('Warning: skipping {} as it is not in reference data.'.format(entry['question_id']))
                continue
            
            qtype = qid2qtype[entry['question_id']]
            q = qid2qdata[entry['question_id']]['question']
            ans = qid2qdata[entry['question_id']]['answer']
            hyp = entry['hypothesis']
            if len(hyp.strip()) > 30000:
                hyp = hyp[:30000]  # truncate if too long
            
            prompt = get_anscheck_prompt(qtype, q, ans, hyp, abstention='_abs' in entry['question_id'])
            messages = [
                {"role": "user", "content": prompt}
            ]

            chat_messages = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_length = inputs.input_ids.shape[1]

            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                top_k=1,
                num_return_sequences=1,
            )
            eval_response = tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True).strip()

            label = 'yes' in eval_response.lower()
            entry['autoeval_label'] = {
                'model': metric_model_name,
                'label': label
            }
            logs.append(entry)
            if verbose:
                print(json.dumps({
                    'question': q,
                    'answer': ans,
                    'hypothesis': hyp,
                    'autoeval_label': label
                }, indent=4), flush=True)
            print(json.dumps(entry), file=out_f)
            qtype2acc[qid2qtype[entry['question_id']]].append(1 if label else 0)

            
    print('Accuracy:', round(np.mean([1 if x['autoeval_label']['label'] else 0 for x in logs]).item(), 4))
    for k,v in qtype2acc.items():
        print('\t{}: {} ({})'.format(k, round(np.mean(v), 4), len(v)))

    print('Saved to', result_file)

    # write a summary file
    summary_file = os.path.join(os.path.dirname(hyp_file), 'summary-{}.txt'.format(metric_model_short))
    write_mode = 'w' if args.overwrite else 'a'
    with open(summary_file, write_mode) as out_f:
        results = {}
        results['hyp_file'] = os.path.basename(hyp_file)
        results['model'] = metric_model_short
        results['overall'] = round(np.mean([1 if x['autoeval_label']['label'] else 0 for x in logs]).item(), 4)
        results['num_samples'] = len(logs)
        for k,v in qtype2acc.items():
            results[k] = round(np.mean(v), 4)
        print(json.dumps(results), file=out_f)
