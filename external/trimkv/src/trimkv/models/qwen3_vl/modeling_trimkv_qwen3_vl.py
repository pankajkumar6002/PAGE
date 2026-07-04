from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs


from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextMLP,
    Qwen3VLVisionModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLPreTrainedModel,
    Qwen3VLCausalLMOutputWithPast,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
)

from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLVisionConfig,
)

from .configuration_trimkv_qwen3_vl import TrimKVQwen3VLConfig, TrimKVQwen3VLTextConfig 

from torch.nn.attention.flex_attention import (
    create_block_mask,
)

from trimkv.attn import get_attention_interface 
from trimkv.triton import retention_sum_packed_triton
from trimkv.cache_utils import TrimKVCache, DynamicBudgetTrimKVCache, PagedTrimKVCache


logger = logging.get_logger(__name__)
create_block_mask_compiled = torch.compile(create_block_mask)


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


class RetentionGate(nn.Module):
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


class TrimKVQwen3VLTextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: TrimKVQwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3VLTextRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape

        if config.retention_gate == 'rg':
            self.retention_gate = RetentionGate(config, layer_idx)
        elif config.retention_gate == 'rg10':
            self.retention_gate = RetentionGate10(config, layer_idx)
        elif config.retention_gate is None:
            self.retention_gate = None
        else:
            raise ValueError(f"Unknown retention_gate type: {config.retention_gate}")

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vanilla_forward: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        if self.config.retention_gate in ['rg', 'rg7', 'rg10']:
            retention_weights = self.retention_gate(hidden_states).transpose(1, 2)
        else:
            retention_weights = None

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.config.retention_gate in ['rg2', 'rg3', 'rg4', 'rg5', 'rg6', 'rg9']:
            kv_states = torch.cat([key_states, value_states], dim=-1)  # (B, H, S, 2*D)
            retention_weights = self.retention_gate(kv_states).transpose(1, 2)
        elif self.config.retention_gate in ['rg8']:
            retention_weights = self.retention_gate(key_states, value_states).transpose(1, 2)
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
            dropout=0.0 if not self.training else self.config.rg_dropout,
            scaling=self.scaling,
            is_causal=self.is_causal,
            flash_attn_kwargs=flash_attn_kwargs if past_key_values is not None else {},
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if self.training:
            return attn_output, attn_weights, retention_weights, summarized_retention_weights
        else:
            return attn_output, attn_weights, None, None


class TrimKVQwen3VLTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: TrimKVQwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = TrimKVQwen3VLTextAttention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        vanilla_forward: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, retention_weights, summarized_retention_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
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
        return hidden_states, self_attn_weights, retention_weights, summarized_retention_weights


@dataclass
class TrimKVQwen3VLModelOutputWithPast(Qwen3VLModelOutputWithPast):
    cache_embeds: Optional[dict[str, Any]] = None
    retention_weights: Optional[torch.Tensor] = None
    summarized_retention_weights: Optional[torch.Tensor] = None
    text_position_ids: Optional[torch.Tensor] = None


class TrimKVQwen3VLPreTrainedModel(Qwen3VLPreTrainedModel):
    config: TrimKVQwen3VLConfig
    _no_split_modules = ["TrimKVQwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]
    _can_record_outputs = {
        "hidden_states": TrimKVQwen3VLTextDecoderLayer,
        "attentions": TrimKVQwen3VLTextAttention,
    }

    def _init_weights(self, module):
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        else:
            # 0.02 is the standard default value across the library
            std = getattr(self.config.get_text_config(), "initializer_range", 0.02)

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
        download_from='local',
        *model_args,
        **kwargs
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

            config = TrimKVQwen3VLConfig.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
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
                raise ValueError(f"Path {pretrained_model_name_or_path} does not exist.")
        else:
            model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        return model


class TrimKVQwen3VLTextModel(TrimKVQwen3VLPreTrainedModel):
    config: TrimKVQwen3VLTextConfig
    _no_split_modules = ["TrimKVQwen3VLTextDecoderLayer"]

    def __init__(self, config: TrimKVQwen3VLTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [TrimKVQwen3VLTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

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

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        vanilla_forward: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, TrimKVQwen3VLModelOutputWithPast]:
        r"""
        visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
            The mask of the visual positions.
        deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
            The deepstack visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).
            The feature is extracted from the different visual encoder layers, and fed to the decoder
            hidden states. It's from the paper DeepStack(https://arxiv.org/abs/2406.04334).
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # torch.jit.trace() doesn't support cache objects in the output

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            raise ValueError("`use_cache=True` requires `past_key_values` to be provided, but `past_key_values` is `None`. Please initialize TrimKVCache yourself and pass it in.")

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        retention_weights = () if self.config.retention_gate is not None else None
        summarized_retention_weights = () if self.config.retention_gate is not None else None

        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                vanilla_forward=vanilla_forward,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

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

        if past_key_values is not None and self.config.compress_memory:
            past_key_values.compress()

        return TrimKVQwen3VLModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            text_position_ids=text_position_ids,
            retention_weights=retention_weights,
            summarized_retention_weights=summarized_retention_weights,
        )

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states


class TrimKVQwen3VLModel(TrimKVQwen3VLPreTrainedModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: TrimKVQwen3VLConfig
    _no_split_modules = ["TrimKVQwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = TrimKVQwen3VLTextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

        # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        # Same implementation as for images
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        vanilla_forward: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, TrimKVQwen3VLModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError("You must specify input_ids or inputs_embeds, if both are specified, inputs_embeds is used.")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            image_mask = None
            video_mask = None

            if pixel_values is not None:
                image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask, _ = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                _, video_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            visual_pos_masks = None
            deepstack_visual_embeds = None
            if image_mask is not None and video_mask is not None:
                # aggregate visual_pos_masks and deepstack_visual_embeds
                image_mask = image_mask[..., 0]
                video_mask = video_mask[..., 0]
                visual_pos_masks = image_mask | video_mask
                deepstack_visual_embeds = []
                image_mask_joint = image_mask[visual_pos_masks]
                video_mask_joint = video_mask[visual_pos_masks]
                for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                    embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                    embed_joint[image_mask_joint, :] = img_embed
                    embed_joint[video_mask_joint, :] = vid_embed
                    deepstack_visual_embeds.append(embed_joint)
            elif image_mask is not None:
                image_mask = image_mask[..., 0]
                visual_pos_masks = image_mask
                deepstack_visual_embeds = deepstack_image_embeds
            elif video_mask is not None:
                video_mask = video_mask[..., 0]
                visual_pos_masks = video_mask
                deepstack_visual_embeds = deepstack_video_embeds
            # No need to backprop through embeddings
            inputs_embeds = inputs_embeds.detach().requires_grad_(True)
        else:
            inputs_embeds = inputs_embeds.requires_grad_(True)

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            vanilla_forward=vanilla_forward,
            **kwargs,
        )

        cache_embeds = dict(
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        return TrimKVQwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            cache_embeds=cache_embeds,
            rope_deltas=self.rope_deltas,
            text_position_ids=outputs.text_position_ids,
            retention_weights=outputs.retention_weights,
            summarized_retention_weights=outputs.summarized_retention_weights,
        )


@dataclass
class TrimKVQwen3VLCausalLMOutputWithPast(Qwen3VLCausalLMOutputWithPast):
    retention_weights: Optional[torch.Tensor] = None
    summarized_retention_weights: Optional[torch.Tensor] = None
    text_position_ids: Optional[torch.Tensor] = None
    retention_loss: Optional[torch.FloatTensor] = None
    base_loss: Optional[torch.FloatTensor] = None
    cache_embeds: Optional[dict] = None


class TrimKVQwen3VLForConditionalGeneration(TrimKVQwen3VLPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: TrimKVQwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = TrimKVQwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

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
            memory_size = self.config.memory_size * self.config.text_config.num_hidden_layers * self.config.text_config.num_key_value_heads
            position_ids = position_ids * self.config.text_config.num_hidden_layers * self.config.text_config.num_key_value_heads
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
            base_loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs)
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
        mask = (labels != -100).float() # [bs, max_lenth], view(-1)->[bs*max_lenth]
        distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0) # num
        return distil_loss

    def compute_base_loss(self, hidden_states: torch.Tensor, labels: Optional[torch.LongTensor] = None, base_logits: Optional[torch.Tensor] = None, **kwargs: Unpack[TransformersKwargs]) -> Optional[torch.Tensor]:
        """
        Computes the base loss based on the configuration.
        """
        assert labels is not None, "Labels must be provided."
        base_loss = 0.0
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

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        base_logits: Optional[torch.Tensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        vanilla_forward: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, TrimKVQwen3VLCausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            cache_position=cache_position,
            vanilla_forward=vanilla_forward,
            **kwargs,
        )

        hidden_states = outputs[0]

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
        if retention_weights is not None and self.training and not vanilla_forward and self.config.retention_weight > 0:
            retention_loss = self.compute_retention_loss(retention_weights, outputs.summarized_retention_weights, position_ids=outputs.text_position_ids)

        loss = None
        if base_loss is not None and retention_loss is not None:
            loss = base_loss + self.config.retention_weight * retention_loss
        elif base_loss is not None:
            loss = base_loss
        elif retention_loss is not None:
            loss = retention_loss

        return TrimKVQwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            retention_loss=retention_loss,
            base_loss=base_loss,
            cache_embeds=outputs.cache_embeds,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        # Qwen3VL position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if inputs_embeds is not None:
            vision_start_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(vision_start_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            video_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        else:
            vision_start_mask = input_ids == vision_start_token_id
            image_mask = input_ids == image_token_id
            video_mask = input_ids == video_token_id

        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=list(video_nums), repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

__all__ = [
    "TrimKVQwen3VLForConditionalGeneration",
    "TrimKVQwen3VLModel",
    "TrimKVQwen3VLPreTrainedModel",
    "TrimKVQwen3VLTextModel"
]
