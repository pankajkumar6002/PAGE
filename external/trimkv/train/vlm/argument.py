import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

@dataclass
class ModelArguments:
    base_model: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    download_from: Optional[str] = field(
        default=None,
        metadata={"help": "The source to download the pre-trained model from. Options: 'huggingface', 'wandb', 'local', 'none'."},
    )
    load_trimkv_weights: bool = field(
        default=False,
        metadata={"help": "Whether to load pre-trained weights for TrimKV retention gates."},
    )
    trainable_params: str = field(
        default="self_attn.retention_gate",
        metadata={"help": "compressor trainable parameters, separated by |, e.g., self_attn.f_proj|self_attn.retention_gate|self_attn.v_proj|self_attn.q_proj|self_attn.k_proj"},
    )
    retention_gate: str = field(
        default="rg",
        metadata={"help": "The retention gate implementation to use. Options: 'rg2', 'rg'."},
    )
    base_loss: str = field(
        default="fwkl",
        metadata={"help": "The base loss to use. Options: 'ntp', 'fwkl', 'rvkl'."},
    )
    attn_impl: str = field(
        default="rg_attn_flex",
        metadata={"help": "The attention implementation to use. Options: 'rg_attn_flex', no other options."},
    )
    memory_size: float = field(
        default=256,
        metadata={"help": "The memory size of the model. If < 1, it represents the fraction of the sequence length."},
    )
    retention_weight: float = field(
        default=1.0,
        metadata={"help": "The retention weight of the model."},
    )
    retention_gate_bias_init: float = field(
        default=8.0,
        metadata={"help": "The retention gate bias init of the model."},
    )
    use_cache: bool = field(
        default=False,
        metadata={"help": "Whether to use cache in the model."},
    )
    skip_layers: Optional[int] = field(
        default=0,
        metadata={"help": "Number of layers to skip for compression. If set to 0, no layers are skipped."},
    )
    rg_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for retention gate."},
    )
    tie_retention_gate_layers: bool = field(
        default=True,
        metadata={"help": "Whether to tie retention gate layers."},
    )

@dataclass
class DataArguments:
    dataset_dir: str = field(default="./data")
    dataset_use: str = field(default="")
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    data_packing_shuffle: bool = field(default=True)
    repacking: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 2048)
    min_pixels: int = field(default=28 * 28 * 32)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=4 * 2048 * 28 * 28)
    video_min_pixels: int = field(default=4 * 32 * 28 * 28)
    video_fps: float = 2
    math220k_min_length: int = field(default=2048)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a checkpoint from which to resume training."},
    )
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=20480,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    save_entire_model: bool = field(
        default=False,
        metadata={"help": "Whether to save the entire model or just the retention gate."},
    )
    logit_block_size: int = field(
        default=-1,
        metadata={"help": "The block size for logit computation. -1 means no block."},
    )
    gradient_checkpointing_kwargs: dict = field(
        default_factory=lambda: {"use_reentrant": False},
        metadata={"help": "Additional keyword arguments for gradient checkpointing."},
    )
    global_capacity: bool = field(
        default=True,
        metadata={"help": "Whether to use global retention memory."},
    )
