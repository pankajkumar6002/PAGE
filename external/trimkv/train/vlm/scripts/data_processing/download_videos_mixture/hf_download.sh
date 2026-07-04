#!/bin/bash
export $(cat .env | xargs)
IMAGE_SRC_DIR="${DATASET_DIR}/M4-Instruct-Data"
VIDEO_SRC_DIR="${DATASET_DIR}/hf_videos"

hf download lmms-lab/M4-Instruct-Data --repo-type dataset --local-dir ${IMAGE_SRC_DIR}
hf download ShareGPTVideo/train_video_and_instruction --repo-type dataset --local-dir ${VIDEO_SRC_DIR}/llava_hound --include train_300k/*
hf download lmms-lab/LLaVA-Video-178K --repo-type dataset --local-dir ${VIDEO_SRC_DIR}/academic_v0_1 --include 0_30_s_academic_v0_1/*
hf download lmms-lab/LLaVA-Video-178K --repo-type dataset --local-dir ${VIDEO_SRC_DIR} --include llava_hound/sharegptvideo_qa_255k_processed.json



