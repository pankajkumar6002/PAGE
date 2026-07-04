from dataclasses import dataclass
from typing import Optional, Union, Callable, Any

import os
import torch
from torch import nn
from torch.nn import functional as F

from transformers.activations import ACT2FN
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple, logging
from transformers.models.auto import AutoModel

from transformers.models.llava.modeling_llava import (
    LlavaModelOutputWithPast,
    LlavaCausalLMOutputWithPast,
    LlavaMultiModalProjector,
    LlavaPreTrainedModel,
)

from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaRotaryEmbedding,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
)

from torch.nn.attention.flex_attention import (
    create_block_mask,
)

from trimkv.attn import get_attention_interface 
from trimkv.triton import retention_sum_packed_triton
from trimkv.cache_utils import TrimKVCache, DynamicBudgetTrimKVCache, PagedTrimKVCache

from .configuration_trimkv_llava import TrimKVLlavaConfig, TrimKVLlavaTextConfig


logger = logging.get_logger(__name__)


create_block_mask_compiled = torch.compile(create_block_mask)




@dataclass
class TrimKVLlavaModelOutputWithPast(LlavaModelOutputWithPast):
    cache_embeds: Optional[dict[str, Any]] = None
    retention_weights: Optional[torch.Tensor] = None
    summarized_retention_weights: Optional[torch.Tensor] = None
    text_position_ids: Optional[torch.Tensor] = None


@dataclass
class TrimKVLlavaCausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    retention_weights: Optional[torch.Tensor] = None
    summarized_retention_weights: Optional[torch.Tensor] = None
    text_position_ids: Optional[torch.Tensor] = None
    retention_loss: Optional[torch.FloatTensor] = None
    base_loss: Optional[torch.FloatTensor] = None
    cache_embeds: Optional[dict] = None

@dataclass
class TrimKVBaseModelOutputWithPast(BaseModelOutputWithPast):
    cache_embeds: Optional[dict[str, Any]] = None
    retention_weights: Optional[torch.Tensor] = None
    summarized_retention_weights: Optional[torch.Tensor] = None
    text_position_ids: Optional[torch.Tensor] = None


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


class TrimKVLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: TrimKVLlavaTextConfig, layer_idx: int):
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

        if config.retention_gate == 'rg':
            self.retention_gate = RetentionGate(config, layer_idx=layer_idx)
        elif config.retention_gate == 'rg10':
            self.retention_gate = RetentionGate10(config, layer_idx=layer_idx)
        elif config.retention_gate is None:
            self.retention_gate = None
        else:
            raise ValueError(f"Unknown retention_gate type: {config.retention_gate}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vanilla_forward: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        if self.config.retention_gate in ['rg2', 'rg3', 'rg4']:
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
            rg_dropout=0.0 if not self.training else self.config.rg_dropout,
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


class TrimKVLlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: TrimKVLlavaTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = TrimKVLlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
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


class TrimKVLlavaPreTrainedModel(LlavaPreTrainedModel):
    config: TrimKVLlavaConfig

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

            config = TrimKVLlavaConfig.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
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


class TrimKVLlamaModel(TrimKVLlavaPreTrainedModel):
    config: TrimKVLlavaTextConfig

    def __init__(self, config: TrimKVLlavaTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [TrimKVLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
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

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        vanilla_forward: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> TrimKVBaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            raise ValueError("`use_cache=True` requires `past_key_values` to be provided, but `past_key_values` is `None`. Please initialize TrimKVCache yourself and pass it in.")

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        retention_weights = () if self.config.retention_gate is not None else None
        summarized_retention_weights = () if self.config.retention_gate is not None else None


        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                vanilla_forward=vanilla_forward,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
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

        return TrimKVBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            text_position_ids=position_ids,
            retention_weights=retention_weights,
            summarized_retention_weights=summarized_retention_weights,
        )


class TrimKVLlavaModel(TrimKVLlavaPreTrainedModel):
    _checkpoint_conversion_mapping = {"language_model.model": "language_model"}

    def __init__(self, config: TrimKVLlavaConfig):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)

        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.language_model = TrimKVLlamaModel._from_config(config.text_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        **kwargs,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`):
               The tensors corresponding to the input images.
            vision_feature_layer (`Union[int, list[int]]`, *optional*):
                The index of the layer to select the vision feature. If multiple indices are provided,
                the vision feature of the corresponding indices will be concatenated to form the
                vision features.
            vision_feature_select_strategy (`str`, *optional*):
                The feature selection strategy used to select the vision feature from the vision backbone.
                Can be one of `"default"` or `"full"`
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # this is not memory efficient at all (output_hidden_states=True) will save all the hidden states.
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True, **kwargs)

        # If we have one vision feature layer, return the corresponding hidden states,
        # otherwise, select the hidden states of each feature layer and concatenate them
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
        else:
            hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            # For default; crop CLS from each hidden state in the hidden state pool
            if vision_feature_select_strategy == "default":
                hs_pool = [hs[:, 1:] for hs in hs_pool]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        image_features = self.multi_modal_projector(selected_image_feature)

        if "image_sizes" in kwargs:
            split_sizes = [
                (height // self.vision_tower.patch_size) * (width // self.vision_tower.patch_size)
                for height, width in kwargs["image_sizes"]
            ]
            image_features = torch.split(image_features.squeeze(0), split_sizes)
        else:
            image_features = list(image_features)
        return image_features

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
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
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_features.shape[0] * image_features.shape[1]
        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        return special_image_mask

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        cache_position: Optional[torch.LongTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        vanilla_forward: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, LlavaModelOutputWithPast]:
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            if pixel_values is not None:
                image_features = self.get_image_features(
                    pixel_values=pixel_values,
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                    image_sizes=image_sizes,
                )
                image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                special_image_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_features
                )
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

            # No need to backprop through embeddings
            inputs_embeds = inputs_embeds.detach().requires_grad_(True)
        else:
            inputs_embeds = inputs_embeds.requires_grad_(True)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            vanilla_forward=vanilla_forward,
            **kwargs,
        )

        cache_embeds = dict(
            inputs_embeds=inputs_embeds,
        )

        return TrimKVLlavaModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            # image_hidden_states=image_features if pixel_values is not None else None,
            cache_embeds=cache_embeds,
            text_position_ids=outputs.text_position_ids,
            retention_weights=outputs.retention_weights,
            summarized_retention_weights=outputs.summarized_retention_weights,
        )


class TrimKVLlavaForConditionalGeneration(TrimKVLlavaPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^vision_tower": "model.vision_tower",
        "^multi_modal_projector": "model.multi_modal_projector",
        "^language_model.lm_head": "lm_head",
    }
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: TrimKVLlavaConfig):
        super().__init__(config)
        self.model = TrimKVLlavaModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        **kwargs,
    ):
        return self.model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            **kwargs,
        )

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def multi_modal_projector(self):
        return self.model.multi_modal_projector

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
        mask = (labels != -100).int() # [bs, max_lenth], view(-1)->[bs*max_lenth]
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

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_sizes: Optional[torch.Tensor] = None,
        vanilla_forward: bool = False,
        base_logits: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, TrimKVLlavaCausalLMOutputWithPast]:
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            cache_position=cache_position,
            image_sizes=image_sizes,
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

        return TrimKVLlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            retention_loss=retention_loss,
            base_loss=base_loss,
            cache_embeds=outputs.cache_embeds,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if cache_position[0] == 0:
            # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
            # Otherwise we need pixel values to be passed to model
            model_inputs["pixel_values"] = pixel_values

        return model_inputs


__all__ = ["TrimKVLlavaForConditionalGeneration", "TrimKVLlavaPreTrainedModel", "TrimKVLlavaModel"]
