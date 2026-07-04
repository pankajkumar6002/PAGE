import re
import os

DEBUG_CFG = {
    "annotation_path": "debug_data/annotation/llava_text_vqa_debug_512.json",
    "data_path": "debug_data/data",
    "min_length": 128,
}

R1_ONEVISION_CFG = {
    "annotation_path": "Processed-R1-Onevision/R1-Onevision_annotation.json",
    "data_path": "Processed-R1-Onevision",
    "min_length": 128,
}

M4_INSTRUCT_IMAGE_CFG = {
    "annotation_path": "m4-instruct/images/m4_instruct_annotations_fixed.json",
    "data_path": "m4-instruct/images/",
    "min_length": 128,
}

M4_INSTRUCT20_IMAGE_CFG = {
    "annotation_path": "m4-instruct/images/m4_instruct20_annotations_fixed.json",
    "data_path": "m4-instruct/images/",
    "min_length": 128,
}

M4_INSTRUCT50_IMAGE_CFG = {
    "annotation_path": "m4-instruct/images/m4_instruct50_annotations_fixed.json",
    "data_path": "m4-instruct/images/",
    "min_length": 128,
}

ACADEMIC_CAPTION_CFG = {
    "annotation_path": "hf_videos/academic_v0_1/0_30_s_academic_v0_1/0_30_s_academic_v0_1_cap_processed_fixed.json",
    "data_path": "hf_videos/academic_v0_1/0_30_s_academic_v0_1",
    "min_length": 128,
}

ACADEMIC_OPENENDED_CFG = {
    "annotation_path": "hf_videos/academic_v0_1/0_30_s_academic_v0_1/0_30_s_academic_oe_v0_1_qa_processed_fixed.json",
    "data_path": "hf_videos/academic_v0_1/0_30_s_academic_v0_1",
    "min_length": 128,
}

LLAVA_HOUND_CFG = {
    "annotation_path": "hf_videos/llava_hound/sharegptvideo_qa_255k_processed_fixed.json",
    "data_path": "hf_videos/llava_hound",
    "min_length": 128,
}

LLAVA_NEXT_CFG = {
    "annotation_path": "llava_next_data/annotations.json",
    "data_path": "llava_next_data/images",
    "min_length": 128,
}

MMDU_45k_CFG = {
    "annotation_path": "mmdu/mmdu_processed.json",
    "data_path": "mmdu",
    "min_length": 512,
}

MATH220k_CFG = {
    "annotation_path": "OpenR1-Math-220k/math_220k.json",
    "data_path": "",
    "min_length": 4096,
}

DATA_CONFIGS = {
    "debug_data": DEBUG_CFG,
    "m4_instruct_images": M4_INSTRUCT_IMAGE_CFG,
    "m4_instruct20_images": M4_INSTRUCT20_IMAGE_CFG,
    "m4_instruct50_images": M4_INSTRUCT50_IMAGE_CFG,
    "academic_caption": ACADEMIC_CAPTION_CFG,
    "academic_openended": ACADEMIC_OPENENDED_CFG,
    "llava_hound": LLAVA_HOUND_CFG,
    "r1_onevision": R1_ONEVISION_CFG,
    "llava_next": LLAVA_NEXT_CFG,
    "math_220k": MATH220k_CFG,
    "mmdu_45k": MMDU_45k_CFG,
}

def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0
    

def get_dataset_configs(dataset_names, dataset_dir="."):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        print(f"Dataset: {dataset_name}, Sampling Rate: {sampling_rate}")
        if dataset_name in DATA_CONFIGS.keys():
            config = DATA_CONFIGS[dataset_name].copy()
            config['annotation_path'] = os.path.join(dataset_dir, config['annotation_path'])
            config['data_path'] = os.path.join(dataset_dir, config['data_path']) if config['data_path'] is not None else None
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    
    dataset_names = ["debug_data"]
    configs = get_dataset_configs(dataset_names)
    for config in configs:
        print(config)
# define min length here
