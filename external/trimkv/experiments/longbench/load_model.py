import torch
import json

from trimkv.models.qwen3 import TrimKVQwen3ForCausalLM, TrimKVQwen3Config
from trimkv.models.llama import TrimKVLlamaForCausalLM, TrimKVLlamaConfig
from trimkv.models.qwen2 import TrimKVQwen2ForCausalLM, TrimKVQwen2Config
from trimkv.models.phi3 import TrimKVPhi3ForCausalLM, TrimKVPhi3Config
from trimkv.cache_utils import TrimKVCache, DynamicBudgetTrimKVCache
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from transformers import GenerationConfig, SinkCache, DynamicCache
from baselines.rkv.monkeypatch import replace_llama, replace_qwen2, replace_qwen3, replace_phi3
from baselines.rkv.dynamic_cache import RKVDynamicCache


def load_rkv_model(config):
    compression_config = {
        "method": config.method,
        "method_config": {
            "budget": config.kv_budget,
            "window_size": config.window_size,
            "mix_lambda": config.mix_lambda,
            "retain_ratio": config.retain_ratio,
            "retain_direction": config.retain_direction,
            "first_tokens": config.first_tokens,
            "buffer_size": config.buffer_size,
        },
        "compression": None,
        "update_kv": config.update_kv
    }
    model_config = {
        "divide_method": config.divide_method,
        "divide_length": config.divide_length,
        "compression_content": config.compression_content,
    }
    # apply monkey patch
    if config.method.lower() != "fullkv":
        if "llama" in config.model_path.lower():
            replace_llama(compression_config)
        elif "qwen3" in config.model_path.lower():
            replace_qwen3(compression_config)
        elif "qwen" in config.model_path.lower():
            replace_qwen2(compression_config)
        elif "phi" in config.model_path.lower():
            replace_phi3(compression_config)
        else:
            raise ValueError(f"Unsupported model: {config.model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=True,
        attn_implementation=config.attn_implementation,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, use_fast=True, padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


    model.config.update(model_config)

    if config.method.lower() != "fullkv":
        model.newline_token_ids = [
            tokenizer.encode("\n")[-1],
            tokenizer.encode(".\n")[-1],
            tokenizer.encode(")\n")[-1],
            tokenizer.encode("\n\n")[-1],
            tokenizer.encode(".\n\n")[-1],
            tokenizer.encode(")\n\n")[-1],
        ]

        model.after_think_token_ids = [
            tokenizer.encode("</think>")[-1],
        ]

    def cache_creator(model, **kwargs):
        return RKVDynamicCache()

    return model, tokenizer, cache_creator


def load_trimkv_model(config):
    if "qwen3" in config.model_path.lower():
        print("Loading Qwen3 TrimKV model...")
        model_cls = TrimKVQwen3ForCausalLM
        model_config_cls = TrimKVQwen3Config
    elif "llama" in config.model_path.lower():
        print("Loading LLaMA TrimKV model...")
        model_cls = TrimKVLlamaForCausalLM
        model_config_cls = TrimKVLlamaConfig
    elif "qwen" in config.model_path.lower():
        print("Loading Qwen2 TrimKV model...")
        model_cls = TrimKVQwen2ForCausalLM
        model_config_cls = TrimKVQwen2Config
    elif "phi" in config.model_path.lower():
        print("Loading Phi3 TrimKV model...")
        model_cls = TrimKVPhi3ForCausalLM
        model_config_cls = TrimKVPhi3Config
    else:
        raise ValueError(f"Unsupported model: {config.model_path}")

    model = model_cls.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        load_trimkv_weights=True,
        download_from=config.download_from,
        use_cache=True,
        device_map="cuda",
    )
    model.config._attn_implementation = config.attn_implementation
    assert config.attn_implementation in ['flash_attention_2'], "Only 'flash_attention_2' are supported for TrimKV models. Sdpa leads to an error for some reasons."
    model.config.compress_memory = config.update_kv
    model.config.memory_size = config.kv_budget
    model.config.compress_strategy = config.compress_strategy
    model.config.buffer_size = config.buffer_size
    print(f"Using TrimKV model with config: {model.config}")

    tokenizer = AutoTokenizer.from_pretrained(
        model.config.base_model, use_fast=True, padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def cache_creator(model, **kwargs):
        max_model_len = kwargs.get("max_model_len", 32768)
        past_key_values = TrimKVCache(
            memory_size=config.kv_budget,
            buffer_size=config.buffer_size,
            device="cuda",
        )
        return past_key_values

    return model, tokenizer, cache_creator


def load_seer_attention_model(config):
    """
    Load the Seer attention model based on the provided configuration.
    This function is a placeholder and should be implemented as needed.
    """
    try:
        from seer_attn import SeerDecodingQwen3ForCausalLM ##  Sparse Decoding Modeling
    except ImportError:
        raise ImportError("Please install seer_attn package to use Seer attention model.")

    if "SeerAttention" not in config.model_path:
        raise ValueError("Model path must contain 'SeerAttention' for Seer attention model.")
    
    model_config = AutoConfig.from_pretrained(config.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.base_model, 
        padding_side="left",
    )
    model = SeerDecodingQwen3ForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        seerattn_sparsity_method='token_budget', 
        seerattn_token_budget=config.kv_budget,
    ).to("cuda")

    def cache_creator(model, **kwargs):
        return DynamicCache()

    return model, tokenizer, cache_creator



def load_fullkv_model(config):
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=True,
        attn_implementation=config.attn_implementation,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, use_fast=True, padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def cache_creator(model, **kwargs):
        return RKVDynamicCache()

    return model, tokenizer, cache_creator


def load_locret_model(config):
    from baselines.locret import generate, load_model_and_tokenizer
    assert "phi3" in config.model_type.lower(), "Currently only LocRet Phi3 model is supported."
    print("Loading LocRet Phi3 model...")
    model2path = json.load(open("configs/model2path.json", "r"))
    # config.model_path is a huggingface path to retainment head checkpoint, download if necessary
    from huggingface_hub import snapshot_download
    import os
    os.makedirs(f"ckpts/locret/{config.model_type}", exist_ok=True)
    snapshot_download(
        repo_id=config.model_path,
        repo_type="model",
        local_dir=f"ckpts/locret/{config.model_type}",
        local_dir_use_symlinks=False
    )
    retainment_head_path = os.path.join(f"ckpts/locret/{config.model_type}", "phi-3-mini-128K.bin")

    model, tokenizer = load_model_and_tokenizer(model2path[config.model_type], retainment_head_path=retainment_head_path)
    return model, tokenizer, generate


LOADER_MAP = {
    "rkv": load_rkv_model,
    "fullkv": load_fullkv_model,
    "snapkv": load_rkv_model,
    "streamingllm": load_rkv_model,
    "h2o": load_rkv_model,
    "seerattn": load_seer_attention_model,
    "trimkv": load_trimkv_model,
    "locret": load_locret_model,
}

def load_model(config):
    assert config.method.lower() in LOADER_MAP, f"Unsupported method: {config.method}"

    load_model_fn = LOADER_MAP.get(config.method.lower())

    model, tokenizer, cache_creator = load_model_fn(config)
    model.eval()
    print("Model and tokenizer loaded.")
    return model, tokenizer, cache_creator
