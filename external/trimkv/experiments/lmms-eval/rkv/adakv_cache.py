import warnings

import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
from typing import List, Optional, Tuple, Union, Any,Dict
from transformers.cache_utils import Cache, DynamicCache
from trimkv.triton import update_flatten_view_triton
from flash_attn import flash_attn_func
# perform qk calculation and get indices
# this version will not update in inference mode

class AdaKVDynamicCache(DynamicCache):
    """
    Flattened version of DynamicCacheSplitHead
    """
    def __init__(self) ->None:
        # Token wise List[]  Head wise KV List[torch.Tensor]
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0
        self.head_lens: List[torch.Tensor] = []
        self.cu_seqlens_k: List[torch.Tensor] = []

    def __len__(self):
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))

    def __getitem__(self, layer_idx: int) -> Tuple[Tuple[torch.Tensor],Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # NOTE: k, v = [head_num](bs, 1, seqlen, dim)
        # each layer is a flatten layout like:
        # [head_0_len + head_1_len + ..., dim]

        bs, num_heads, seq_len, dim = key_states.shape
        device = key_states.device
        self._seen_tokens += seq_len

        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        if len(self.key_cache) <= layer_idx:
            key_states = key_states.view(-1, dim)
            value_states = value_states.view(-1, dim)
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.head_lens.append(torch.tensor([seq_len] * num_heads, device=device, dtype=torch.int32))
            self.cu_seqlens_k.append(torch.arange(0, (seq_len * num_heads) + 1, step=seq_len, device=device, dtype=torch.int32))
        else:
            assert self.key_cache[layer_idx].dim() == 2

            new_key_cache = update_flatten_view_triton(self.key_cache[layer_idx].view(-1,dim), key_states, self.head_lens[layer_idx], self.cu_seqlens_k[layer_idx])
            new_value_cache = update_flatten_view_triton(self.value_cache[layer_idx].view(-1,dim), value_states, self.head_lens[layer_idx], self.cu_seqlens_k[layer_idx])

            self.key_cache[layer_idx] = new_key_cache
            self.value_cache[layer_idx] = new_value_cache

            # Update head_lens and cu_seqlens_k
            self.head_lens[layer_idx] = self.head_lens[layer_idx] + seq_len
            cu_offset = torch.arange(0, (num_heads * seq_len) + 1, step=seq_len, device=device, dtype=torch.int32)
            self.cu_seqlens_k[layer_idx] = self.cu_seqlens_k[layer_idx] + cu_offset


        return self.key_cache[layer_idx], self.value_cache[layer_idx], self.head_lens[layer_idx], self.cu_seqlens_k[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCacheEachHead":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

