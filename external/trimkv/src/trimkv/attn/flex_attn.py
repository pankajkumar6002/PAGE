from typing import Callable, Optional, Tuple, Union, Dict, Any
from functools import lru_cache, partial

import torch
import torch.nn as nn

from torch.nn.attention.flex_attention import (
    create_block_mask,
    _mask_mod_signature,
    BlockMask,
    flex_attention,
    _score_mod_signature,
)


# flex_attention_compiled = torch.compile(flex_attention, mode="max-autotune-no-cudagraphs", dynamic=False)
flex_attention_compiled = torch.compile(flex_attention, dynamic=True)
# flex_attention_compiled = torch.compile(flex_attention, dynamic=False)
# flex_attention_compiled = torch.compile(flex_attention)
# flex_attention_compiled = flex_attention

kernel_options = {}


def is_power_of_two(n: int) -> bool:
    return (n != 0) and (n & (n - 1)) == 0


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda", **kwargs):
    block_mask = create_block_mask(score_mod, B=B, H=H, Q_LEN=M, KV_LEN=N, device=device, **kwargs)
    return block_mask


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def get_mask_mod(mask_mod: _mask_mod_signature, offset: int):
    def _mask_mod(b, h, q, kv):
        return mask_mod(b, h, q + offset, kv)

    return _mask_mod


def retention_gated_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: BlockMask,
    retention_weights: torch.Tensor,
    kv_positions: torch.Tensor = None,
    offset: int = 0,
    scaling: float = None,
    rg_dropout: float = 0.0,
    **kwargs,
):
    _, N_Q_HEADS, _, _ = query.shape
    _, N_KV_HEADS, _, _ = key.shape
    N_Q_PER_GROUP = N_Q_HEADS // N_KV_HEADS
    enable_gqa = (N_Q_HEADS != N_KV_HEADS) and is_power_of_two(N_Q_PER_GROUP)
    if not enable_gqa and N_Q_PER_GROUP != 1:
        key = repeat_kv(key, N_Q_PER_GROUP)
        value = repeat_kv(value, N_Q_PER_GROUP)

    if rg_dropout > 0.0 and module.training:
        raise NotImplementedError("rg_dropout is not implemented for retention_gated_attention_forward.")

    retention_weights = retention_weights.to(torch.float32)

    def score_mod_w_kv_pos(score, b, h, q_idx, kv_idx):
        return score + (retention_weights[b, h // N_Q_PER_GROUP, kv_idx] * (q_idx + offset - kv_positions[b, h // N_Q_PER_GROUP, kv_idx]))

    def score_mod_wo_kv_pos(score, b, h, q_idx, kv_idx):
        return score + (retention_weights[b, h // N_Q_PER_GROUP, kv_idx] * (q_idx + offset - kv_idx))

    score_mod = score_mod_w_kv_pos if kv_positions is not None else score_mod_wo_kv_pos

    attn_output = flex_attention_compiled(
        query,
        key,
        value,
        scale=scaling,
        block_mask=attention_mask,
        score_mod=score_mod,
        enable_gqa=enable_gqa,
        kernel_options=kernel_options,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None, None


__all__ = [
    "retention_gated_attention_forward",
]
