import os
import json
import random
import logging
import pathlib
import torch
import transformers
import sys
import importlib.metadata
_original_distributions = importlib.metadata.distributions
def _patched_distributions():
    """Filter out distributions with None metadata"""
    for dist in _original_distributions():
        if dist.metadata is not None:
            yield dist
importlib.metadata.distributions = _patched_distributions
from dotenv import load_dotenv
from pathlib import Path
from deepspeed.accelerator import get_accelerator

from transformers import AutoProcessor, Trainer
from dataclasses import dataclass, field
from typing import Optional

from trainer import replace_qwen2_vl_attention_class
from dataset.data_processor import make_supervised_data_module
from trimkv.models.qwen2_5_vl import TrimKVQwen2_5_VLForConditionalGeneration, TrimKVQwen2_5_VLConfig
from trimkv.models.qwen3_vl import TrimKVQwen3VLForConditionalGeneration, TrimKVQwen3VLConfig
from trimkv.models.llava import TrimKVLlavaConfig, TrimKVLlavaForConditionalGeneration
from argument import ModelArguments, DataArguments, TrainingArguments

local_rank = None

def ds_param_count(model, trainable_only=False):
    params = (p for p in model.parameters() if (p.requires_grad or not trainable_only))
    return sum(getattr(p, "ds_numel", p.numel()) for p in params)

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def truncate(s, max_length=127):
    """Truncate a string to a maximum length, ensuring it does not exceed the limit."""
    s = s.replace("-", "_").replace("/", "_").replace(",", "_")
    if len(s) > max_length:
        return s[:max_length - 3] + "..."
    return s

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
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


def update_config(config, model_args, training_args, update_base_model=True):
    if update_base_model:
        config.base_model = model_args.base_model
    config.memory_size = model_args.memory_size
    config.retention_weight = model_args.retention_weight
    config.trainable_params = model_args.trainable_params
    config.base_loss = model_args.base_loss
    config.logit_block_size = training_args.logit_block_size
    config.global_capacity = training_args.global_capacity

    text_config = {
        "retention_gate_bias_init": model_args.retention_gate_bias_init,
        "use_cache": model_args.use_cache,
        "attn_impl": model_args.attn_impl,
        "rg_dropout": model_args.rg_dropout,
        "retention_gate": model_args.retention_gate,
        "tie_retention_gate_layers": model_args.tie_retention_gate_layers,
    }

    if training_args.model_max_length is not None:
        text_config['max_seq_len'] = training_args.model_max_length

    config.update_text_config(**text_config)

    return config

    
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
            inputs["base_logits"] = ori_outputs.logits.detach()
            cache_embeds = ori_outputs.get("cache_embeds", None)
            if cache_embeds is not None:
                for k, v in cache_embeds.items():
                    inputs[k] = v.detach() if isinstance(v, torch.Tensor) else v

            torch.cuda.empty_cache()
            get_accelerator().empty_cache()

        outputs = model(**inputs)

        loss = outputs.get("loss")
        logs = {"total_loss": loss.item()}
        if getattr(outputs, "retention_loss", None) is not None:
            logs["retention_loss"] = outputs.retention_loss.item()
        if getattr(outputs, "base_loss", None) is not None:
            logs["base_loss"] = outputs.base_loss.item()
        self.log(logs)

        # if outputs.base_loss > 3:
        #     rank0_print("warning: base loss is too high:", outputs.base_loss)
        #     print("inputs:", {k: v.shape if isinstance(v, torch.Tensor) else v for k, v in inputs.items()})
        #     print(self._tokenizer.decode(inputs['input_ids'][0]))
        #     # replace -100 with padding token id for better visualization
        #     inputs['labels'][inputs['labels'] == -100] = self._tokenizer.pad_token_id
        #     print(self._tokenizer.decode(inputs['labels'][0]))
        #     raise ValueError

        return (loss, outputs) if return_outputs else  loss

    def training_step(self, *args, **kwargs):
        out = super().training_step(*args, **kwargs)
        return out


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)
    print(f"output dir: {training_args.output_dir}")

    if "qwen2.5" in model_args.base_model.lower():
        model_cls = TrimKVQwen2_5_VLForConditionalGeneration
        config_cls = TrimKVQwen2_5_VLConfig
        data_args.model_type = "qwen2.5vl"
    elif "qwen3" in model_args.base_model.lower():
        model_cls = TrimKVQwen3VLForConditionalGeneration
        config_cls = TrimKVQwen3VLConfig
        data_args.model_type = "qwen3vl"
    elif "llava-1.5" in model_args.base_model.lower():
        model_cls = TrimKVLlavaForConditionalGeneration
        config_cls = TrimKVLlavaConfig
        data_args.model_type = "llava1.5"
    else:
        raise NotImplementedError(f"model {model_args.base_model} not supported yet, please implement it yourself")

    if model_args.load_trimkv_weights:
        model = model_cls.from_pretrained(
            model_args.base_model,
            load_trimkv_weights=model_args.load_trimkv_weights,
            download_from=model_args.download_from,
            torch_dtype=torch.float32,
        )
        config = update_config(model.config, model_args, training_args, update_base_model=False)
        processor = AutoProcessor.from_pretrained(
            config.base_model,
            add_eos_token=True,
        )
    else:
        config = config_cls.from_pretrained(
            model_args.base_model,
        )
        config = update_config(config, model_args, training_args)

        model = model_cls.from_pretrained(
            model_args.base_model,
            load_trimkv_weights=model_args.load_trimkv_weights,
            download_from=model_args.download_from,
            config=config,
            torch_dtype=torch.float32,
        )
        print(f'the initlized model is {model_args.base_model} the class is {model.__class__.__name__}')
        processor = AutoProcessor.from_pretrained(
            model_args.base_model,
            add_eos_token=True,
        )


    # a trick to tell the language model to prepare the flex attention's mask for us, only works for transformers>=4.57.0
    model.language_model.config._attn_implementation = 'flex_attention'
    # this is to avoid overriding _attn_implementation in model's from_pretrained
    model.language_model.config.attn_impl = 'rg_attn_flex'
    # model.language_model.config.attn_impl = 'flash_attention_2'

    # with open("chat_template/templates.json", "r") as f:
    #     chat_template = json.load(f)

    # if model_args.base_model in chat_template.keys():
    #     # modify llava_hf chat template to vicuna style
    #     processor.tokenizer.chat_template = chat_template[model_args.base_model]
    #     processor.chat_template = chat_template[model_args.base_model]
    #     print("Using modified chat template:", model_args.base_model)

    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    processor.tokenizer.padding_side = "right"
    processor.tokenizer.model_max_length = training_args.model_max_length
    set_model(model_args, model)

    # if torch.distributed.get_rank() == 0:
    #     model.visual.print_trainable_parameters()
    #     model.model.print_trainable_parameters()
    
    if data_args.data_packing:
        assert training_args.per_device_train_batch_size == 1, "data packing only supports per_device_train_batch_size=1"
    data_module = make_supervised_data_module(processor, data_args=data_args)

    trainer = TrimKVTrainer(
        base_loss=model_args.base_loss,
        model=model,
        processing_class=processor.tokenizer,
        tokenizer=processor.tokenizer,
        args=training_args,
        **data_module
    )

    if training_args.resume_from_checkpoint == 'latest' and list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print(f"checkpoint found in {training_args.output_dir}, resume training")
        print(f"the latest checkpoint is {list(pathlib.Path(training_args.output_dir).glob('checkpoint-*'))[-1]}")
        trainer.train(resume_from_checkpoint=True)
    elif isinstance(training_args.resume_from_checkpoint, str) and os.path.isdir(training_args.resume_from_checkpoint):
        print(f"checkpoint found in {training_args.resume_from_checkpoint}, resume training")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        print("no checkpoint found, start training from scratch")
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    print("Saving model...")
    trainer.save_state()

    final_model_path = os.path.join(training_args.output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)

    if training_args.save_entire_model:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=final_model_path)
    elif local_rank == 0:
        if hasattr(trainer.model, 'module'):
            state_dict = trainer.model.module.state_dict()
        else:
            state_dict = trainer.model.state_dict()

        processor.save_pretrained(final_model_path)
        model.config.save_pretrained(final_model_path)
        trainable_params = model.config.trainable_params.split("|")
        retention_gate_state_dict = {
            k: v for k, v in state_dict.items() if any(trainable_param in k for trainable_param in trainable_params)
        }
        weight_path = os.path.join(final_model_path, "trimkv_weights.pth")
        torch.save(retention_gate_state_dict, weight_path)
        print(f"Saved retention gate weights to {weight_path}")
        # submit to wandb, check wandb mode is online
        import wandb
        if wandb.run is not None:
            artifact = wandb.Artifact(name=truncate(training_args.run_name), type="model")
            for file in os.listdir(final_model_path):
                path = os.path.join(final_model_path, file)
                artifact.add_file(path)

            wandb.log_artifact(artifact)


if __name__ == "__main__":
    torch.set_printoptions(precision=10, sci_mode=False)
    random.seed(42)
    torch.manual_seed(42)
    transformers.set_seed(42)
    train()
