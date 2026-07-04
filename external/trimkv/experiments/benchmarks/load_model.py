import torch

from trimkv.models.qwen3 import TrimKVQwen3ForCausalLM, TrimKVQwen3Config
from trimkv.models.llama import TrimKVLlamaForCausalLM, TrimKVLlamaConfig
from trimkv.models.qwen2 import TrimKVQwen2ForCausalLM, TrimKVQwen2Config
from trimkv.models.phi3 import TrimKVPhi3ForCausalLM, TrimKVPhi3Config
from trimkv.cache_utils import TrimKVCache, DynamicBudgetTrimKVCache
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from minference import MInference
from transformers import GenerationConfig, SinkCache, DynamicCache
from rkv.monkeypatch import replace_llama, replace_qwen2, replace_qwen3, replace_phi3


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
        },
        "compression": True,
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
        return None

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
        low_cpu_mem_usage=True,
        download_from=config.download_from,
        use_cache=True,
        device_map="cuda",
    )
    model.config.attn_impl = config.attn_implementation
    model.config.compress_memory = config.update_kv
    model.config.memory_size = config.kv_budget
    model.config.compress_strategy = config.compress_strategy
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
            max_seq_len=max_model_len,
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


def global_kv_update_mode(model, mode):
    try:
        attn_class = model.model.layers[0].self_attn.__class__
    except:
        attn_class = model.transformer.encoder.layers[
            0
        ].self_attention.__class__
    model.apply(
        lambda m: setattr(m, "update_global_past_kv", mode)
        if isinstance(m, attn_class)
        else None
    )


def load_snapkv_model(config):
    minference_patch = MInference(
        'dense',
        config.model_path,
        config_path="",
        starting_layer=-1,
        kv_type=config.method,
        is_search=False,
        kv_cache_cpu=False,
        kv_cache_cpu_device='cpu',
        attn_kwargs={
            "max_capacity_prompt": config.kv_budget,
        },
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=True,
        attn_implementation=config.attn_implementation,
    )
    model = minference_patch(model)
    global_kv_update_mode(model, True)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, use_fast=True, padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def cache_creator(model, **kwargs):
        model_inputs = {}
        model._prepare_cache_for_generation(
            GenerationConfig(), model_inputs, None, None, None, None
        )
        past_key_values = model_inputs["past_key_values"]
        return past_key_values

    return model, tokenizer, cache_creator


def load_streamingllm_model(config):
    minference_patch = MInference(
        'dense',
        config.model_path,
        config_path="",
        starting_layer=-1,
        kv_type=config.method,
        is_search=False,
        kv_cache_cpu=False,
        kv_cache_cpu_device='cpu',
        attn_kwargs={
            "n_local": config.kv_budget - config.first_tokens,
            "first_tokens": config.first_tokens,
        },
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=True,
        attn_implementation=config.attn_implementation,
    )
    model = minference_patch(model)
    global_kv_update_mode(model, True)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, use_fast=True, padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def cache_creator(model, **kwargs):
        model_inputs = {}
        model._prepare_cache_for_generation(
            GenerationConfig(), model_inputs, None, None, None, None
        )
        past_key_values = model_inputs["past_key_values"]
        return past_key_values

    return model, tokenizer, cache_creator


def load_fullkv_model(config):
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

    def cache_creator(model, **kwargs):
        return DynamicCache()

    return model, tokenizer, cache_creator

def load_dbtrimkv_model(config):
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
        low_cpu_mem_usage=True,
        download_from=config.download_from,
        use_cache=True,
        device_map="cuda",
    )
    assert config.attn_implementation == "dbtrimkv_flash", "Dynamic Budget TrimKV requires 'dbtrimkv_flash' attention implementation."
    model.config.attn_impl = config.attn_implementation
    model.config.compress_memory = config.update_kv
    model.config.memory_size = config.kv_budget
    assert config.compress_strategy in ['alpha', 'knorm_alpha'], "Dynamic Budget TrimKV only supports 'alpha' and 'knorm_alpha' compress strategies."
    model.config.compress_strategy = config.compress_strategy
    
    if config.method == 'lwdbtrimkv' and not config.compress_strategy.startswith('layer_wise_'):
        print(f"Overriding compress_strategy to layer_wise_{config.compress_strategy} for lwdbtrimkv method.")
        model.config.compress_strategy = 'layer_wise_' + config.compress_strategy

    if config.method in ['dbtrimkv_v1']:
        print("Using DBTrimKV v1 settings: setting floor_budget_ratio to 0.5")
        model.config.floor_budget_ratio = 0.5
        
    print(f"Using TrimKV model with config: {model.config}")

    tokenizer = AutoTokenizer.from_pretrained(
        model.config.base_model, use_fast=True, padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def cache_creator(model, **kwargs):
        max_model_len = kwargs.get("max_model_len", 32768)
        past_key_values = DynamicBudgetTrimKVCache(
            max_seq_len=max_model_len,
            device="cuda",
        )
        return past_key_values

    return model, tokenizer, cache_creator


LOADER_MAP = {
    "rkv": load_rkv_model,
    "fullkv": load_fullkv_model,
    "snapkv": load_snapkv_model,
    "streamingllm": load_streamingllm_model,
    "h2o": load_rkv_model,
    "trimkv": load_trimkv_model,
    "dbtrimkv": load_dbtrimkv_model,
    "seerattn": load_seer_attention_model,
}

def load_model(config):
    assert config.method.lower() in LOADER_MAP, f"Unsupported method: {config.method}"

    load_model_fn = LOADER_MAP.get(config.method.lower())

    model, tokenizer, cache_creator = load_model_fn(config)
    model.eval()
    print("Model and tokenizer loaded.")
    return model, tokenizer, cache_creator
