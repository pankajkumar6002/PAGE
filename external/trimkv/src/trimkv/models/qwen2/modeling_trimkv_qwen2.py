from functools import partial
import os
import math
from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    logging,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2Model

from .configuration_trimkv_qwen2 import TrimKVQwen2Config
from trimkv.attn import get_attention_interface 
from trimkv.triton import retention_sum_triton, retention_sum_packed_triton
from trimkv.cache_utils import TrimKVCache, DynamicBudgetTrimKVCache, PagedTrimKVCache


logger = logging.get_logger(__name__)


class TrimKVBaseModelOutputWithPast(BaseModelOutputWithPast):
    """
    Base class for outputs of TrimKV models with past key values.
    It extends `BaseModelOutputWithPast` to include the retention loss.
    """

    def __init__(
        self,
        retention_loss: Optional[torch.FloatTensor] = None,
        retention_weights: Optional[torch.FloatTensor] = None,
        summarized_retention_weights: Optional[torch.FloatTensor] = None,
        last_ori_hidden_state: Optional[torch.FloatTensor] = None,
        **kwargs: Union[torch.Tensor, Tuple[torch.Tensor, ...], None]
    ):
        super().__init__(**kwargs)
        self.retention_loss = retention_loss
        self.retention_weights = retention_weights
        self.summarized_retention_weights = summarized_retention_weights
        self.last_ori_hidden_state = last_ori_hidden_state


class TrimKVCausalLMOutputWithPast(CausalLMOutputWithPast):
    """
    Base class for outputs of TrimKV models with past key values and language modeling head.
    It extends `CausalLMOutputWithPast` to include the retention loss.
    """

    def __init__(
        self,
        retention_loss: Optional[torch.FloatTensor] = None,
        base_loss: Optional[torch.FloatTensor] = None,
        retention_weights: Optional[torch.FloatTensor] = None,
        summarized_retention_weights: Optional[torch.FloatTensor] = None,
        **kwargs: Union[torch.Tensor, Tuple[torch.Tensor, ...], None]
    ):
        super().__init__(**kwargs)
        self.retention_loss = retention_loss
        self.base_loss = base_loss
        self.retention_weights = retention_weights
        self.summarized_retention_weights = summarized_retention_weights
    

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RetentionGate(nn.Module):
    """
    Projects each attention-head vector (head_dim) to a single scalar,
    using a separate learnable linear layer per head.

    Input shape : (batch_size, seq_len, input_dim)
    Output shape: (batch_size, seq_len, num_heads)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.retention_gate_intermediate_size = config.retention_gate_intermediate_size
        self.linear1 = nn.Linear(self.hidden_size, self.retention_gate_intermediate_size, bias=True)
        self.linear2 = nn.Linear(self.retention_gate_intermediate_size, config.num_key_value_heads, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.num_key_value_heads))

        self.act_fn = ACT2FN[config.hidden_act]
        self.reset_parameters()

    def reset_parameters(self):
        initializer_range = getattr(self.config, "initializer_range", 0.02)
        self.linear1.weight.data.normal_(mean=0.0, std=initializer_range)
        if self.linear1.bias is not None:
            self.linear1.bias.data.zero_()
        self.linear2.weight.data.normal_(mean=0.0, std=initializer_range)
        if self.linear2.bias is not None:
            self.linear2.bias.data.zero_()
        if self.bias is not None:
            self.bias.data.fill_(self.config.retention_gate_bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        out = self.linear1(x)  # (B, S, D) -> (B, S, H')
        out = self.act_fn(out)  # (B, S, H')
        out = self.linear2(out)  # (B, S, H') -> (B, S, H)
        out = out + self.bias  # (B, S, H)
        # avoid sigmoid here to prevent numerical issues
        out = F.logsigmoid(out)  # (B, S, H)
        return out


class RetentionGate10(nn.Module):
    """
    Projects each attention-head vector (head_dim) to a single scalar,
    using a separate learnable linear layer per head.

    Input shape : (batch_size, seq_len, input_dim)
    Output shape: (batch_size, seq_len, num_heads)
    """
    def __init__(self, config, layer_idx: int = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.retention_gate_intermediate_size = config.retention_gate_intermediate_size
        # self.input_norm = Qwen3VLTextRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.linear1 = nn.Linear(self.hidden_size, self.retention_gate_intermediate_size, bias=True)
        self.linear2 = nn.Linear(self.retention_gate_intermediate_size, self.retention_gate_intermediate_size, bias=True)
        self.linear3 = nn.Linear(self.retention_gate_intermediate_size, config.num_key_value_heads, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.num_key_value_heads))

        self.act_fn = ACT2FN[config.hidden_act]
        self.reset_parameters()

    def reset_parameters(self):
        initializer_range = getattr(self.config, "initializer_range", 0.02)
        self.linear1.weight.data.normal_(mean=0.0, std=initializer_range)
        if self.linear1.bias is not None:
            self.linear1.bias.data.zero_()
        self.linear2.weight.data.normal_(mean=0.0, std=initializer_range)
        if self.linear2.bias is not None:
            self.linear2.bias.data.zero_()
        self.linear3.weight.data.normal_(mean=0.0, std=initializer_range)
        if self.linear3.bias is not None:
            self.linear3.bias.data.zero_()
        if self.bias is not None:
            self.bias.data.fill_(self.config.retention_gate_bias_init)

        # self.input_norm.weight.data.fill_(1.0)
        # self.input_norm.variance_epsilon = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        out = self.linear1(x)  # (B, S, D) -> (B, S, H')
        out = self.act_fn(out)  # (B, S, H')
        out = self.linear2(out)  # (B, S, H') -> (B, S, H)
        out = self.act_fn(out)  # (B, S, H')
        out = self.linear3(out)  # (B, S, H') -> (B, S, H)
        out = out + self.bias  # (B, S, H)
        # avoid sigmoid here to prevent numerical issues
        out = F.logsigmoid(out)  # (B, S, H)
        return out


class TrimKVQwen2Attention(nn.Module):

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        if config.retention_gate == 'rg':
            self.retention_gate = RetentionGate(config)
        elif config.retention_gate == 'rg10':
            self.retention_gate = RetentionGate10(config)
        elif config.retention_gate == None:
            self.retention_gate = None
        else:
            raise ValueError(f"Unsupported retention gate type: {config.retention_gate}")

        self.sliding_window = config.sliding_window
        if not (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            self.sliding_window = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vanilla_forward: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        if self.config.retention_gate in ['rg', 'rg10']:
            retention_weights = self.retention_gate(hidden_states).transpose(1, 2)
        else:
            retention_weights = None

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.config.retention_gate == 'rg2':
            kv_states = torch.cat([key_states, value_states], dim=-1)  # (B, H, S, 2*D)
            retention_weights = self.retention_gate(kv_states).transpose(1, 2)

        offset = past_key_values.get_seq_length() if past_key_values is not None else 0

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "attention_mask": attention_mask,
                "retention_weights": retention_weights,
            }
            key_states, value_states, retention_weights, kv_positions, flash_attn_kwargs = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        else:
            kv_positions = None 

        if not vanilla_forward and self.training:
            attn_impl = self.config.attn_impl
            assert attn_impl in ["rg_attn_flex"], f"Unsupported attention implementation during training: {attn_impl}"
        else:
            attn_impl = self.config._attn_implementation

        if isinstance(past_key_values, DynamicBudgetTrimKVCache):
            # use dynamic budget trimkv attention
            attn_impl = "db_" + attn_impl
        elif isinstance(past_key_values, PagedTrimKVCache):
            # use paged trimkv attention
            attn_impl = "paged_" + attn_impl

        attention_interface: Callable = get_attention_interface(attn_impl)

        attn_output, attn_weights, summarized_retention_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            retention_weights=retention_weights,
            kv_positions=kv_positions if past_key_values is not None else None,
            offset=offset,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            is_causal=self.is_causal,
            flash_attn_kwargs=flash_attn_kwargs if past_key_values is not None else {},
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, retention_weights, summarized_retention_weights


class TrimKVQwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = TrimKVQwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        vanilla_forward: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, retention_weights, summarized_retention_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            vanilla_forward=vanilla_forward,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states, self_attn_weights if output_attentions else None, retention_weights, summarized_retention_weights)


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class TrimKVQwen2PreTrainedModel(PreTrainedModel):
    config_class = TrimKVQwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TrimKVQwen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        if isinstance(module, RetentionGate):
            module.reset_parameters()
        elif isinstance(module, RetentionGate10):
            module.reset_parameters()
        else:
            super()._init_weights(module)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        load_trimkv_weights=True,
        download_from='local',  # or 'wandb'
        *model_args,
        **kwargs,
    ):
        # Call the original method first
        if load_trimkv_weights:
            if download_from == 'wandb':
                import wandb
                api = wandb.Api()
                artifact = api.artifact(pretrained_model_name_or_path, type='model')
                if artifact is not None:
                    print(f"Using wandb artifact: {artifact.name}")
                    if not os.path.exists(artifact._default_root()) or not os.path.exists(os.path.join(artifact._default_root(), "trimkv_weights.pth")) or not os.path.exists(os.path.join(artifact._default_root(), "config.json")):
                        pretrained_model_name_or_path = artifact.download()
                        print(f"Downloaded model from wandb to: {pretrained_model_name_or_path}")
                    else:
                        pretrained_model_name_or_path = artifact._default_root()
                        print(f"Using existing local artifact at: {pretrained_model_name_or_path}")
                else:
                    raise ValueError(f"Artifact {pretrained_model_name_or_path} not found in wandb.")
            elif download_from == 'huggingface':
                from huggingface_hub import snapshot_download
                pretrained_model_name_or_path = snapshot_download(pretrained_model_name_or_path)
                print(f"Downloaded model from HuggingFace to: {pretrained_model_name_or_path}")
            elif download_from == 'local':
                print(f"Loading model from local path: {pretrained_model_name_or_path}")
            else:
                raise ValueError(f"Unsupported download_from value: {download_from}")

            config = TrimKVQwen2Config.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            if hasattr(config, "base_model"):
                base_model = config.base_model
            else:
                base_model = pretrained_model_name_or_path
                config.base_model = pretrained_model_name_or_path

            for key in list(kwargs.keys()):
                if hasattr(config, key) and key != "torch_dtype" and key != "dtype":
                    setattr(config, key, kwargs.pop(key))
            model = super().from_pretrained(base_model, config=config, *model_args, **kwargs)

            if os.path.exists(pretrained_model_name_or_path):
                gate_weights = torch.load(os.path.join(pretrained_model_name_or_path, "trimkv_weights.pth"))
                trainable_params = config.trainable_params.split("|")
                trainble_gate_state_keys = [
                    key for key in model.state_dict().keys() if any(
                        trainable_param in key for trainable_param in trainable_params
                    )
                ]
                # trainable_gate_state_keys and gate_weights.keys() should match
                if set(trainble_gate_state_keys) != set(gate_weights.keys()):
                    raise ValueError(
                        f"Mismatch between trainable gate state keys: {trainble_gate_state_keys} and loaded weights keys: {gate_weights.keys()}"
                    )

                model.load_state_dict(gate_weights, strict=False)
                print("Retention gate weights loaded successfully.")
            else:
                print("Could not load the trimkv gate weights.")
        else:
            model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        return model


class TrimKVQwen2Model(TrimKVQwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [TrimKVQwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

        if config.retention_gate in ['rg10'] and config.tie_retention_gate_layers:
            self._tie_retention_gate_layers()

    def _tie_retention_gate_layers(self):
        # tie bias as well
        shared_linear3 = None
        shared_bias = None
        for layer in self.layers:
            if shared_linear3 is None:
                # shared_linear2 = layer.self_attn.retention_gate.linear2
                shared_linear3 = layer.self_attn.retention_gate.linear3
            else:
                # layer.self_attn.retention_gate.linear2 = shared_linear2
                layer.self_attn.retention_gate.linear3 = shared_linear3

            if shared_bias is None:
                shared_bias = layer.self_attn.retention_gate.bias
            else:
                layer.self_attn.retention_gate.bias = shared_bias

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vanilla_forward: bool = False,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            raise ValueError("`use_cache=True` requires `past_key_values` to be provided, but `past_key_values` is `None`. Please initialize TrimKVCache yourself and pass it in.")

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        retention_weights = () if self.config.retention_gate is not None else None
        summarized_retention_weights = () if self.config.retention_gate is not None else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask_mapping[decoder_layer.attention_type],
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    vanilla_forward,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    vanilla_forward=vanilla_forward,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if self.config.retention_gate is not None and layer_outputs[2] is not None:
                retention_weights += (layer_outputs[2],)

            if self.config.retention_gate is not None and layer_outputs[3] is not None:
                summarized_retention_weights += (layer_outputs[3],)

        if retention_weights is not None and len(retention_weights) > 0:
            retention_weights = torch.stack(retention_weights, dim=1)
        else:
            retention_weights = None

        if summarized_retention_weights is not None and len(summarized_retention_weights) > 0:
            summarized_retention_weights = torch.stack(summarized_retention_weights, dim=1)
        else:
            summarized_retention_weights = None

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if past_key_values is not None and self.config.compress_memory:
            past_key_values.compress()

        return TrimKVBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            retention_weights=retention_weights,
            summarized_retention_weights=summarized_retention_weights,
        )


class TrimKVQwen2ForCausalLM(TrimKVQwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = TrimKVQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def compute_retention_loss(self, retention_weights, summarized_retention_weights, position_ids=None):
        bz, num_layers, num_key_value_heads, seqlen = retention_weights.shape
        if position_ids is not None:
            first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
            position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
            doc_mask = (position_diff != 1).cumsum(-1)

            lens_per_seg = torch.zeros_like(position_ids).scatter_reduce(
                1, doc_mask, position_ids + 1, reduce="amax", include_self=False
            )
            seqlen = lens_per_seg.gather(1, doc_mask)
        else:
            doc_mask = None
            seqlen = torch.full((bz, seqlen), seqlen, device=retention_weights.device, dtype=torch.long)
            position_ids = torch.arange(
                seqlen.max(), device=retention_weights.device
            ).unsqueeze(0).expand(bz, -1)

        if summarized_retention_weights is None:
            retention_weights = retention_weights.view(bz, -1, retention_weights.shape[-1])
            summarized_retention_weights = retention_sum_packed_triton(retention_weights, doc_mask, 128, 128)

        dtype, device = summarized_retention_weights.dtype, summarized_retention_weights.device

        # hinge loss
        if self.config.global_capacity:
            memory_size = self.config.memory_size * self.config.num_hidden_layers * self.config.num_key_value_heads
            position_ids = position_ids * self.config.num_hidden_layers * self.config.num_key_value_heads
            summarized_retention_weights = summarized_retention_weights.sum(dim=1)
            retention_loss = torch.maximum(
                (summarized_retention_weights - memory_size) / (position_ids - memory_size).clamp(min=1.0),
                torch.zeros_like(summarized_retention_weights, dtype=dtype, device=device)
            )
        else:
            memory_size = self.config.memory_size
            retention_loss = torch.maximum(
                (summarized_retention_weights - memory_size) / (position_ids - memory_size).clamp(min=1.0),
                torch.zeros_like(summarized_retention_weights, dtype=dtype, device=device)
            )
        # get all non_zero retention losses and take the mean
        if retention_loss.numel() == 0:
            print("No retention loss to compute, returning 0")
            # raise ValueError
            return torch.tensor(0.0, device=device)

        num_non_zero = (retention_loss > 0).sum()
        retention_loss = retention_loss.sum() / num_non_zero if num_non_zero > 0 else torch.tensor(0.0, device=device)
        return retention_loss

    def compute_ntp_loss(self, hidden_states: torch.Tensor, labels: Optional[torch.LongTensor] = None, **kwargs: Unpack[TransformersKwargs]) -> Optional[torch.Tensor]:
        """
        Computes the NTP loss as described in the Qwen3 paper.
        """
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states)  # [bs, max_lenth, dim]
        base_loss = None
        if labels is not None:
            base_loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return base_loss

    def compute_fwkl_loss(self, hidden_states: torch.Tensor, base_logits: torch.Tensor, labels: Optional[torch.LongTensor] = None, **kwargs: Unpack[TransformersKwargs]) -> torch.Tensor:
        """
        Computes the logits distillation loss
        """
        logits = self.lm_head(hidden_states)  # [bs, max_lenth, dim]
        p_t = F.softmax(base_logits, dim=-1)
        log_p_s = F.log_softmax(logits, dim=-1)
        mask = (labels != -100).float()
        kl = F.kl_div(log_p_s, p_t, reduction='none').sum(-1)  # per token
        distil_loss = (kl * mask).sum() / mask.sum()

        # logits = self.lm_head(hidden_states)  # [bs, max_lenth, dim]
        # ori_probs = F.softmax(base_logits, dim=-1)
        # inf_mask = torch.isinf(logits)
        # logprobs = F.log_softmax(logits, dim=-1)
        # prod_probs = torch.masked_fill(ori_probs * logprobs, inf_mask, 0) # [bs, max_lenth, dim]
        # x = torch.sum(prod_probs, dim=-1).view(-1) # [bs * max_lenth]
        # mask = (labels != -100).int() # [bs, max_lenth], view(-1)->[bs*max_lenth]
        # distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0) # num
        return distil_loss

    def compute_rvkl_loss(self, hidden_states: torch.Tensor, base_logits: torch.Tensor, labels: Optional[torch.LongTensor] = None, **kwargs: Unpack[TransformersKwargs]) -> torch.Tensor:
        """
        Computes the logits distillation loss
        """
        logits = self.lm_head(hidden_states)  # [bs, max_lenth, dim]
        ori_logprobs = F.log_softmax(base_logits, dim=-1)
        inf_mask = torch.isinf(logits)
        logprobs = F.log_softmax(logits, dim=-1)
        probs = logprobs.exp()  # [bs, max_lenth, dim]
        prod_probs = torch.masked_fill(
            probs * (logprobs - ori_logprobs), inf_mask, 0
        )  # [bs, max_lenth, dim]
        # prod_probs = torch.masked_fill(
        #     F.kl_div(ori_logprobs, logprobs, log_target=True, reduction='none'), inf_mask, 0
        # ) # [bs, max_lenth, dim]
        x = torch.sum(prod_probs, dim=-1).view(-1) # [bs * max_lenth]
        mask = (labels != -100).int() # [bs, max_lenth], view(-1)->[bs*max_lenth]
        distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0) # num
        return distil_loss

    def compute_base_loss(self, hidden_states: torch.Tensor, labels: Optional[torch.LongTensor] = None, base_logits: Optional[torch.Tensor] = None, **kwargs: Unpack[TransformersKwargs]) -> Optional[torch.Tensor]:
        """
        Computes the base loss based on the configuration.
        """
        assert labels is not None, "Labels must be provided."
        base_loss = torch.tensor(0.0, device=hidden_states.device)
        n_losses = 0
        if 'ntp' in self.config.base_loss:
            base_loss += self.compute_ntp_loss(hidden_states=hidden_states, labels=labels, **kwargs)
            n_losses += 1
        if "fwkl" in self.config.base_loss:
            assert base_logits is not None, "Base logits must be provided for forward logits distillation loss computation."
            base_loss += self.compute_fwkl_loss(hidden_states=hidden_states, base_logits=base_logits, labels=labels, **kwargs)
            n_losses += 1
        if "rvkl" in self.config.base_loss:
            assert base_logits is not None, "Base logits must be provided for reverse logits distillation loss computation."
            base_loss += self.compute_rvkl_loss(hidden_states=hidden_states, base_logits=base_logits, labels=labels, **kwargs)
            n_losses += 1

        base_loss = base_loss / n_losses if n_losses > 0 else None
        return base_loss

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        vanilla_forward: bool = False,
        base_logits: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            vanilla_forward=vanilla_forward,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        base_loss = None
        if self.training and not vanilla_forward:
            if self.config.logit_block_size > 0:
                assert hidden_states.shape[0] == 1, "Logit block size is only supported for batch size 1."
                num_valid_labels = (labels != -100).sum().item()
                hidden_state_blocks = torch.split(hidden_states, self.config.logit_block_size, dim=1)
                base_logit_blocks = torch.split(base_logits, self.config.logit_block_size, dim=1) if base_logits is not None else [None] * len(hidden_state_blocks)
                label_blocks = torch.split(labels, self.config.logit_block_size, dim=1) if labels is not None else [None] * len(hidden_state_blocks)

                base_loss = sum(
                    ((label_block != -100).sum().item() / num_valid_labels) * 
                    torch.utils.checkpoint.checkpoint(
                        self.compute_base_loss,
                        hidden_states=hidden_state_block,
                        base_logits=base_logit_block,
                        labels=label_block,
                        use_reentrant=False,
                    ) for hidden_state_block, base_logit_block, label_block in zip(
                        hidden_state_blocks, base_logit_blocks, label_blocks
                    ) if (label_block != -100).sum().item() > 0
                )
            else:
                base_loss = self.compute_base_loss(
                    hidden_states=hidden_states,
                    base_logits=base_logits,
                    labels=labels,
                    **kwargs,
                )
            logits = None
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            slice_hidden_states = hidden_states[:, slice_indices, :]
            logits = self.lm_head(slice_hidden_states)

        retention_weights = outputs.retention_weights
        retention_loss = None
        if retention_weights is not None and self.training:
            retention_loss = self.compute_retention_loss(retention_weights, outputs.summarized_retention_weights, position_ids=position_ids)

        loss = None
        if base_loss is not None and retention_loss is not None:
            loss = base_loss + self.config.retention_weight * retention_loss
        elif base_loss is not None:
            loss = base_loss
        elif retention_loss is not None:
            loss = retention_loss

        out = TrimKVCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            retention_loss=retention_loss,
            base_loss=base_loss,
        )
        return out



__all__ = [
    "TrimKVQwen2ForCausalLM",
    "TrimKVQwen2Model",
    "TrimKVQwen2PreTrainedModel",
]
