# scripts/data — VLM dataset download & preprocessing

End-to-end recipes for fetching and preparing the multimodal datasets used to train TrimKV / DBTrimKV on Qwen-VL and LLaVA. Every script reads paths from `train/vlm/.env`, so configure that first:

```env
DATASET_DIR=/abs/path/to/data
HF_HOME=/abs/path/to/hf_cache    # optional
HF_TOKEN=...                     # required for gated datasets
```

All commands below assume you are running them from `train/vlm/`. After preprocessing, edit [`dataset/configs.py`](../../dataset/configs.py) if you change any annotation/data paths, then run [`precompute_seqlen.py`](../../precompute_seqlen.py) to cache per-sample sequence lengths (this dramatically speeds up `data_packing` at training time).

## Training mixture

The default mixture used to train DBTrimKV / TrimKV (see `DATASETS=` in [`scripts/train_trimkv.sh`](../train_trimkv.sh)):

```
r1_onevision%30, m4_instruct50_images%40,
academic_openended%30, academic_caption%30,
mmdu_45k%50, math_220k%20
```

Each `name%N` token means "use N% of `name`". The dataset key on the left of `%` must match a row in [`dataset/configs.py::DATA_CONFIGS`](../../dataset/configs.py).

## Per-dataset recipes

### R1-Onevision (image reasoning, [Fancy-MLLM/R1-Onevision](https://huggingface.co/datasets/Fancy-MLLM/R1-Onevision))

Streams the HF dataset, decodes the embedded base64 images, and writes both `images/` and `R1-Onevision_annotation.json` under `${DATASET_DIR}/Processed-R1-Onevision/`:

```bash
python3 scripts/data/download_and_process_r1_onevision.py
```

### M4-Instruct images + LLaVA-Video-178K academic videos ([download_videos_mixture/](download_videos_mixture/))

This bundle covers three datasets at once: M4-Instruct images, the `0_30_s_academic_v0_1` subset of LLaVA-Video-178K (used for `academic_openended` and `academic_caption`), and optionally LLaVA-Hound.

```bash
cd scripts/data/download_videos_mixture

# 1. Pull the raw archives from HF.
bash hf_download.sh

# 2. M4-Instruct images: unzip, merge the dreamsim split, and rewrite annotations.
bash process_images.sh        # writes ${DATASET_DIR}/m4-instruct/images/...

# 3. Academic v0.1 videos: untar 8 shards and rewrite the QA + caption annotations.
bash process_academic.sh      # writes ${DATASET_DIR}/hf_videos/academic_v0_1/...

# 4. (optional) LLaVA-Hound: untar 16 chunks of frames, transcode to mp4, fix annotations.
bash process_llava_hound.sh   # uses llava_hound_frames2mp4.sh under the hood
```

`fix_annotations.py` and `fix_annotations_llavahound.py` filter out records whose referenced files don't exist on disk and write `*_fixed.json`. The fixed JSONs are what `dataset/configs.py` points to.

To sanity-check the M4 video coverage after extraction:

```bash
python3 scripts/data/process_m4_videos.py \
    --json_file ${DATASET_DIR}/m4-instruct/videos/m4_instruct_video_annotations.json \
    --video_path ${DATASET_DIR}/m4-instruct/videos
```

### MMDU-45k ([laolao77/MMDU](https://huggingface.co/datasets/laolao77/MMDU))

Downloads the JSON + images zip, extracts in place, and rewrites the conversation roles into `human`/`gpt`:

```bash
python3 scripts/data/download_mmdu.py
```

### OpenR1-Math-220k

The math split uses pre-built annotations. Build them once with [build_annotations_math_220k.py](build_annotations_math_220k.py) — it imports `dataset.configs`, so it must be invoked from `train/vlm/`:

```bash
python3 scripts/data/build_annotations_math_220k.py
```

This populates `${DATASET_DIR}/OpenR1-Math-220k/math_220k.json` referenced by `MATH220k_CFG`.

### LLaVA-NeXT-Data (optional, used by the `llava_next` config)

```bash
bash scripts/data/download_llava_next_data.sh   # wraps download_parallel.py
```

`download_parallel.py` streams `lmms-lab/LLaVA-NeXT-Data`, dumps each image to `data/llava_next_data/images/<id>.jpg`, and writes a normalised annotation file.

### LLaVA-Instruct (legacy mixture: COCO, GQA, TextVQA, VG, OCR-VQA)

Only needed if you want to reproduce the LLaVA-1.5 instruct mixture. Pulls images from each upstream and OCR-VQA via the [JunHill/ocr-vqa](https://huggingface.co/datasets/JunHill/ocr-vqa) mirror:

```bash
bash scripts/data/download_llava_instruct.sh
```

### Debug data

Tiny mixture used by `DEBUG=1` runs:

```bash
bash scripts/data/download_debug_data.sh
```

## Precomputing sequence lengths

Once a dataset's annotation JSON is in place, cache its per-sample token lengths so the packing dataloader doesn't recompute them every epoch:

```bash
python3 precompute_seqlen.py \
    --dataset r1_onevision \
    --model qwen3vl \
    --mode parallel \
    --num_workers 16
```

`--dataset` accepts any key from `DATA_CONFIGS`; `--model` selects the tokenizer/processor (`qwen2_5vl`, `qwen3vl`, `llava1_5`). The script writes `*_seqlen.jsonl` next to the source annotation.

## Layout

- [download_and_process_r1_onevision.py](download_and_process_r1_onevision.py) — R1-Onevision streamer + image dumper.
- [download_mmdu.py](download_mmdu.py) — MMDU-45k downloader + role rewriter.
- [download_llava_next_data.sh](download_llava_next_data.sh) — wrapper for `download_parallel.py`.
- [download_parallel.py](download_parallel.py) — parallel HF → local image+annotation dumper for LLaVA-NeXT.
- [download_llava_instruct.sh](download_llava_instruct.sh) — LLaVA-1.5 image mixture (COCO/GQA/TextVQA/VG/OCR-VQA).
- [download_debug_data.sh](download_debug_data.sh) — small fixture for smoke tests.
- [process_m4_videos.py](process_m4_videos.py) — utility to verify M4-Instruct video coverage.
- [build_annotations_math_220k.py](build_annotations_math_220k.py) — one-time builder for the `math_220k` annotation file (run from `train/vlm/`).
- [download_videos_mixture/](download_videos_mixture/) — M4-Instruct, LLaVA-Video-178K-academic, and LLaVA-Hound (download + extract + annotation fixers).
