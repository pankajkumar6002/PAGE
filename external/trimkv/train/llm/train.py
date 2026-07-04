import os
import datetime
import warnings
import json
import random
import importlib.metadata
_original_distributions = importlib.metadata.distributions
def _patched_distributions():
    """Filter out distributions with None metadata"""
    for dist in _original_distributions():
        if dist.metadata is not None:
            yield dist
importlib.metadata.distributions = _patched_distributions
import wandb

from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import torch
import deepspeed
import transformers
# from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
from trl import DataCollatorForCompletionOnlyLM
from transformers.models.qwen3 import Qwen3ForCausalLM, Qwen3Config
from transformers.models.llama import LlamaForCausalLM, LlamaConfig
from transformers.trainer_utils import get_last_checkpoint
from deepspeed.accelerator import get_accelerator
from torch.distributed import barrier
from dataset import load_dataset, PackedDataset, FlattenedDataCollatorForLanguageModeling

import trimkv
from trimkv.models.qwen3 import TrimKVQwen3ForCausalLM, TrimKVQwen3Config
from trimkv.models.qwen2 import TrimKVQwen2ForCausalLM, TrimKVQwen2Config
from trimkv.models.llama import TrimKVLlamaForCausalLM, TrimKVLlamaConfig
from trimkv.models.phi3 import TrimKVPhi3ForCausalLM, TrimKVPhi3Config

warnings.simplefilter(action='ignore', category=FutureWarning)


def ds_param_count(model, trainable_only=False):
    params = (p for p in model.parameters() if (p.requires_grad or not trainable_only))
    return sum(getattr(p, "ds_numel", p.numel()) for p in params)

def truncate(s, max_length=127):
    """Truncate a string to a maximum length, ensuring it does not exceed the limit."""
    s = s.replace("-", "_").replace("/", "_").replace(",", "_")
    if len(s) > max_length:
        return s[:max_length - 3] + "..."
    return s


class TrimKVTrainer(Trainer):
    def __init__(
            self,
            base_loss='ntp',
            *args,
            **kwargs
        ):
        self.base_loss = base_loss
        self._tokenizer = kwargs.pop("tokenizer", None)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if any(base_loss in self.base_loss for base_loss in ['fwkl', 'rvkl']):
            with torch.no_grad():
                ori_outputs = model(
                    **inputs,
                    vanilla_forward=True,
                )
            inputs["base_logits"] = ori_outputs.logits
            torch.cuda.empty_cache()
            get_accelerator().empty_cache()

        outputs = model(**inputs)

        loss = outputs.get("loss")
        logs = {"total_loss": loss.item()}
        if getattr(outputs, "retention_loss", None) is not None:
            logs["retention_loss"] = outputs.retention_loss.item()
        if getattr(outputs, "base_loss", None) is not None:
            logs["base_loss"] = outputs.base_loss.item() if isinstance(outputs.base_loss, torch.Tensor) else outputs.base_loss
        self.log(logs)

        return (loss, outputs) if return_outputs else  loss

    def training_step(self, *args, **kwargs):
        out = super().training_step(*args, **kwargs)
        return out


@dataclass
class ModelArguments:
    base_model: str = field(default="meta-llama/Meta-Llama-3.1-8B")
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "The local model path if any."})
    trainable_params: str = field(
        default="self_attn.retention_gate",
        metadata={"help": "compressor trainable parameters, separated by |, e.g., self_attn.f_proj|self_attn.retention_gate|self_attn.v_proj|self_attn.q_proj|self_attn.k_proj"},
    )
    retention_gate: str = field(
        default="rg4",
        metadata={"help": "The retention gate implementation to use. Options: 'rg3', 'rg2', 'rg1'."},
    )
    base_loss: str = field(
        default="ntp",
        metadata={"help": "The base loss to use. Options: 'ntp', 'fwkl', 'rvkl'."},
    )
    attn_impl: str = field(
        default="rg_attn_flex",
        metadata={"help": "The attention implementation to use. Options: 'rg_attn_flex' and ... no other options."},
    )
    memory_size: float = field(
        default=1024,
        metadata={"help": "The memory size of the model. If < 1, it represents the fraction of the sequence length."},
    )
    retention_weight: float = field(
        default=1.0,
        metadata={"help": "The retention weight of the model."},
    )
    retention_gate_bias_init: float = field(
        default=0.0,
        metadata={"help": "The retention gate bias init of the model."},
    )
    use_cache: bool = field(
        default=False,
        metadata={"help": "Whether to use cache in the model."},
    )
    rg_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for retention gate."},
    )
    global_capacity: bool = field(
        default=False,
        metadata={"help": "Whether to use global retention memory."},
    )
    tie_retention_gate_layers: bool = field(
        default=True,
        metadata={"help": "Whether to tie the retention gate weights across layers."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    training_max_length: int = field(
        default=None,
        metadata={"help": "Maximum sequence length in training."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a checkpoint to resume training from.",
        },
    )
    dataset_name: Optional[str] = field(
        default="./cache_dir/qwq-sftopenr1-newtemplate",
        metadata={"help": "The name of the dataset to use."},
    )
    max_samples: Optional[int] = field(
        default=-1,
        metadata={"help": "For debugging purposes, truncate the number of samples."},
    )
    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the dataset if any."},
    )
    save_entire_model: bool = field(
        default=False,
        metadata={"help": "Save entire model."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={"help": "Overwrite the output directory."},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether to use gradient checkpointing."},
    )
    gradient_checkpointing_kwargs: dict = field(
        default_factory=lambda: {"use_reentrant": False},
        metadata={"help": "Additional keyword arguments for gradient checkpointing."},
    )
    logit_block_size: int = field(
        default=-1,
        metadata={"help": "The block size for logit computation."},
    )
    data_packing: bool = field(
        default=False,
        metadata={"help": "Whether to use data packing."},
    )


def update_config(config, model_args, training_args):
    config.base_model = model_args.base_model
    config.retention_gate = model_args.retention_gate
    config.memory_size = model_args.memory_size
    config.retention_weight = model_args.retention_weight
    config.retention_gate_bias_init = model_args.retention_gate_bias_init
    config.trainable_params = model_args.trainable_params
    config.attn_impl = model_args.attn_impl
    config.use_cache = model_args.use_cache
    config.base_loss = model_args.base_loss
    config.rg_dropout = model_args.rg_dropout
    config.logit_block_size = training_args.logit_block_size
    config.global_capacity = model_args.global_capacity
    config.tie_retention_gate_layers = model_args.tie_retention_gate_layers

    if training_args.training_max_length is not None:
        config.max_seq_len = training_args.training_max_length
    return config


def train():
    deepspeed.init_distributed(dist_backend="nccl", init_method="env://", timeout=datetime.timedelta(minutes=120))

    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    # temporary fix an error of transformers 4.57.0 that cannot parse this argument
    training_args.lr_scheduler_kwargs = {"min_lr": 1e-6}

    if os.path.exists(os.path.join(training_args.output_dir, "trimkv_weights.pth")) and not training_args.overwrite_output_dir:
        print(f"Attn gate weights already exist at {os.path.join(training_args.output_dir, 'trimkv_weights.pth')}, skip training.")
        return

    os.makedirs(training_args.output_dir, exist_ok=True)

    if model_args.model_name_or_path is None:
        model_args.model_name_or_path = model_args.base_model

    if "qwen3" in model_args.model_name_or_path.lower():
        print("Using Qwen3 model")
        model_cls = TrimKVQwen3ForCausalLM
        config_cls = TrimKVQwen3Config
        response_template = "<|im_start|>assistant\n"
        instruction_template = "<|im_start|>user\n"
    elif "qwen" in model_args.model_name_or_path.lower():
        print("Using Qwen2 model")
        model_cls = TrimKVQwen2ForCausalLM
        config_cls = TrimKVQwen2Config
        response_template = "<|im_start|>assistant\n"
        instruction_template = "<|im_start|>user\n"
    elif "llama" in model_args.model_name_or_path.lower():
        print("Using Llama model")
        model_cls = TrimKVLlamaForCausalLM
        config_cls = TrimKVLlamaConfig
    elif "phi-3" in model_args.model_name_or_path.lower() or "phi-4" in model_args.model_name_or_path.lower():
        print("Using Phi model")
        model_cls = TrimKVPhi3ForCausalLM
        config_cls = TrimKVPhi3Config
        response_template = "<|assistant|>"
        instruction_template = "<|user|>"
    else:
        raise ValueError("Model not supported. Current only support qwen2, qwen3, and llama model.")

    config = config_cls.from_pretrained(
        model_args.model_name_or_path,
    )
    config = update_config(config, model_args, training_args)

    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        load_trimkv_weights=False,
        config=config,
        torch_dtype=torch.bfloat16,
    )
    model.enable_input_require_grads()

    # a trick to tell the language model to prepare the flex attention's mask for us, only works for transformers>=4.57.0
    model.config._attn_implementation = 'flex_attention'
    # this is to avoid overriding _attn_implementation in model's from_pretrained
    model.config.attn_impl = model_args.attn_impl

    print(model)
    print("Using model:", model_args.model_name_or_path)
    print("Config:", model.config)
    print("tokenier name:", model_args.model_name_or_path, 
          "model name:", model_args.base_model, "training max length:", training_args.training_max_length)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        model_max_length=training_args.training_max_length,
        trust_remote_code=True,
        use_fast=True,
    )
    # set the tokenizer pad token as eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Some chat template will remove the tokens between <think> and </think>, 
    # so we need modify the chat template to keep the tokens for training.
    with open("chat_template/templates.json", "r") as f:
        chat_template = json.load(f)

    if model_args.base_model in chat_template.keys():
        # print("Using modified chat template:", model_args.base_model)
        tokenizer.chat_template = chat_template[model_args.base_model]

    total_num_params = ds_param_count(model, trainable_only=False)
    total_trainable_params = 0
    trainable_params = model_args.trainable_params.split("|")
    for n, p in model.named_parameters():
        if any(trainable_param in n for trainable_param in trainable_params):
            p.requires_grad = True
            # compute the number of trainable parameters
            num_params = getattr(p, "ds_numel", p.numel())
            total_trainable_params += num_params
        else:
            p.requires_grad = False

    print(f"Total trainable parameters: {total_trainable_params} ({total_trainable_params / total_num_params * 100:.2f}%)")

    rank = int(os.environ.get('RANK', -1))
    if rank > 0:
        barrier()

    dataset = load_dataset(
        training_args=training_args,
        tokenizer=tokenizer,
    )

    print("Dataset size:", len(dataset))

    if rank == 0:
        barrier()

    if training_args.data_packing:
        print("Using data packing.")
        dataset = PackedDataset(dataset, training_args.training_max_length)
        data_collator = FlattenedDataCollatorForLanguageModeling(tokenizer=tokenizer)
    else:
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            instruction_template=instruction_template,
            tokenizer=tokenizer,
        )


    print("Output directory:", training_args.output_dir)

    trainer = TrimKVTrainer(
        model=model,
        base_loss=model_args.base_loss,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    if training_args.resume_from_checkpoint == "None":
        training_args.resume_from_checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        if training_args.resume_from_checkpoint == 'auto':
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is not None:
                print(f"Found checkpoint {last_checkpoint}.")
            training_args.resume_from_checkpoint = last_checkpoint

        if training_args.resume_from_checkpoint is not None:
            if not os.path.isdir(training_args.resume_from_checkpoint):
                raise ValueError(f"Checkpoint {training_args.resume_from_checkpoint} does not exist.")
            print(f"Resuming from checkpoint: {training_args.resume_from_checkpoint}")

    # torch.cuda.memory._record_memory_history(max_entries=100000)

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    # torch.cuda.memory._dump_snapshot("ms.pickle")
    # torch.cuda.memory._record_memory_history(enabled=None)

    print("Saving model...")
    trainer.save_state()
    if training_args.save_entire_model:
        trainer.save_model(output_dir=training_args.output_dir)
    elif rank == 0:
        if hasattr(trainer.model, 'module'):
            state_dict = trainer.model.module.state_dict()
        else:
            state_dict = trainer.model.state_dict()

        model.config.save_pretrained(training_args.output_dir)
        trainable_params = model.config.trainable_params.split("|")
        retention_gate_state_dict = {
            k: v for k, v in state_dict.items() if any(trainable_param in k for trainable_param in trainable_params)
        }
        path = os.path.join(training_args.output_dir, "trimkv_weights.pth")
        torch.save(retention_gate_state_dict, path)
        print(f"Saved retention gate weights to {path}")
        # submit to wandb, check wandb mode is online
        if wandb.run is not None:
            artifact = wandb.Artifact(name=truncate(training_args.run_name), type="model")
            artifact.add_file(path)
            artifact.add_file(os.path.join(training_args.output_dir, "config.json"))
            wandb.log_artifact(artifact)


if __name__ == "__main__":
    # set random seed for reproducibility
    # torch.autograd.set_detect_anomaly(True)
    # set print options
    torch.set_printoptions(precision=10, sci_mode=False)
    random.seed(42)
    torch.manual_seed(42)
    transformers.set_seed(42)
    train()
