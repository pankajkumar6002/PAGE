# The code is adapted from https://github.com/FFY0/AdaKV/blob/main/adaptive_snapkv/monkeypatch/snapkv_utils.py
# The codebase from AdaKV only compresses the KV cache during the prefill phase.
# Here, we adapt the code to compress the KV cache during both prefill and decode phases.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Copied from transformers.models.llama.modeling_llama.repeat_kv for gqa_support
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


class AdaKV():
    def __init__(self, budget = None, window_size = 32, kernel_size = 7, pooling = 'maxpool', floor_alpha = 0.5,skip = None,normalize=None, 
                 layer_idx = None, num_hidden_layers = None, pyram_mode = False, pyram_beta = 20,gqa_support=True, gqa_func='mean', **kwargs):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.budget = budget - window_size
        self.floor_ratio = floor_alpha
        self.floor_capacity = int(self.budget * self.floor_ratio)
        self.adaptive_capacity = self.budget - self.floor_capacity
        self.skip = skip

        self.normalize = normalize
        self.pyram_init = False
        self.pyram_mode = pyram_mode
        self.pyram_beta = pyram_beta
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

         # support gqa
        self.gqa_support = gqa_support
        self.gqa_func = gqa_func
        if self.gqa_support:
            assert gqa_func is not None, "gqa_func should not be None"
            assert gqa_func in ['max','mean'], "currently gqa_func should be in ['max','mean']"

    def update_compression_config(
        self,
        **compression_config,
    ):
        self.budget = compression_config.get("budget", self.budget)
        self.window_size = compression_config.get("window_size", self.window_size)
        self.kernel_size = compression_config.get("kernel_size", self.kernel_size)

    
    def update_kv(self, origin_key_states, query_states, origin_value_states, head_lens, cu_klen):
        if self.gqa_support:
            return self.update_kv_gqa(origin_key_states, query_states, origin_value_states, head_lens, cu_klen)
        else:
            return self.update_kv_wo_gqa(origin_key_states, query_states, origin_value_states, head_lens, cu_klen)

    def calc_attn_score(self, key_states, query_states, head_lens, cu_klen):
        bsz, num_heads, q_len, head_dim = query_states.shape
        assert bsz == 1, "Only batch size 1 is supported in calc_attn_score"
        assert key_states.dim() == 2, "key_states should be in flatten view"
        num_kv_heads = head_lens.shape[0]
        num_heads_per_kv = num_heads // num_kv_heads

        cur_query_states = query_states[:, :, -self.window_size:, :]  # (1, num_heads, window_size, head_dim)

        # because the key_states is flattened, we need to compute the attention score separately for each head
        attn_weights = []
        for head_idx in range(num_kv_heads):
            head_key_states = key_states[cu_klen[head_idx]:cu_klen[head_idx+1], :].view(1, 1, -1, head_dim)  # (1, 1, seqlen_k, head_dim)
            head_key_states = head_key_states.expand(1, num_heads_per_kv, -1, -1)  # (1, num_heads_per_kv, seqlen_k, head_dim)
            head_query_states = cur_query_states[:, head_idx * num_heads_per_kv:(head_idx + 1) * num_heads_per_kv, :, :]  # (1, num_heads_per_kv, window_size, head_dim)
            head_attn_weights = torch.matmul(head_query_states, head_key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(head_attn_weights.dtype).min,
                              device=head_attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=head_attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(head_attn_weights.device)
            attention_mask = mask[None, None, :, :]
            head_attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask
            head_attn_weights = nn.functional.softmax(head_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            head_attn_weights_mean = head_attn_weights.mean(dim=-2)  # (1, num_heads_per_kv, seqlen_k)

            if self.gqa_support:
                if self.gqa_func == 'max':
                    head_attn_weights_mean = head_attn_weights_mean.max(dim=-2, keepdim=True).values
                elif self.gqa_func == 'mean':
                    head_attn_weights_mean = head_attn_weights_mean.mean(dim=-2, keepdim=True)
                else:
                    raise ValueError('gqa_func not supported')
            
            if self.pooling == 'avgpool':
                head_attn_weights_mean = F.avg_pool1d(head_attn_weights_mean, kernel_size=self.kernel_size,
                                                         padding=self.kernel_size // 2,
                                                         stride=1)
            elif self.pooling == 'maxpool':
                head_attn_weights_mean = F.max_pool1d(head_attn_weights_mean, kernel_size=self.kernel_size,
                                                         padding=self.kernel_size // 2,
                                                         stride=1)
            else:
                raise ValueError('Pooling method not supported')

            attn_weights.append(head_attn_weights_mean)

        # concat back to the flatten view
        attn_weights_mean = torch.cat(attn_weights, dim=2).squeeze()  # (num_heads, seqlen_k)

        return attn_weights_mean

    # update kv with gqa_support
    def update_kv_gqa(self, origin_key_states, query_states, origin_value_states, head_lens, cu_klen):
        key_states = origin_key_states
        value_states = origin_value_states
        num_heads = query_states.shape[1]
        num_kv_heads = head_lens.shape[0]
        num_key_value_groups = num_heads // num_kv_heads
        # check if prefix phase        assert key_states.shape[-2] == query_states.shape[-2]
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape

        # compute pyramidal capacity
        if self.pyram_mode and not self.pyram_init:
            # NOTE: (max_num + min_num) / 2 == budget to restrict the total capacity
            min_num = self.budget // self.pyram_beta
            max_num = self.budget * 2 - min_num
                
            # if the max_num is larger than the query length, we need to adjust the max_num
            # if max_num >= q_len - self.window_size:
            #     max_num = q_len - self.window_size
            #     min_num = self.budget * 2 - max_num
            # [MODIFIED] Dont adjust max_num based on q_len to support decode phase, if q_len is small it leads to negative steps
        
            # NOTE: compute interval
            steps = (max_num - min_num) // (self.num_hidden_layers - 1)

            # renew adaptive capacity
            self.budget = max_num - self.layer_idx * steps
            self.floor_budget = int(self.budget * self.floor_ratio)
            self.adaptive_budget = self.budget - self.floor_capacity
            self.pyram_init = True
            print(f"Pyram mode adaptive capacity, layer: {self.layer_idx}, acap: {self.adaptive_capacity}, bcap: {self.budget}, fcap: {self.floor_budget}",  flush=True)

        if num_kv_heads * self.budget >= key_states.shape[0]:
            # not compress
            return origin_key_states, origin_value_states, head_lens, cu_klen

        attn_score = self.calc_attn_score(key_states, query_states, head_lens,cu_klen)

        topk_indices = torch.topk(attn_score,k=num_kv_heads * self.budget, dim=-1).indices
        topk_mask = torch.zeros_like(attn_score,dtype=torch.bool)
        topk_mask.scatter_(-1,topk_indices,True)

        # floor_alpha capacity set
        head_adaptive_capacity = torch.tensor([topk_mask[cu_klen[i]:cu_klen[i+1]].sum().item() for i in range(num_kv_heads)], device=_device, dtype=torch.float32)
        assert head_adaptive_capacity.sum().item() == num_kv_heads*self.budget, "head_adaptive_capacity sum error"
        head_adaptive_capacity = torch.round(head_adaptive_capacity * (1-self.floor_ratio) + self.floor_capacity).int()

        heads_key_states = []
        heads_value_states = []
        assert bsz == 1
        # per head

        # reinit varlen metadata
        new_head_lens = []

        for head_idx in range(num_kv_heads):
            head_attn_score = attn_score[cu_klen[head_idx]:cu_klen[head_idx+1] - self.window_size]  # (seqlen_k,)
            head_key_states = origin_key_states[cu_klen[head_idx]:cu_klen[head_idx+1], :]  # (seqlen_k, head_dim)
            head_value_states = origin_value_states[cu_klen[head_idx]:cu_klen[head_idx+1], :]

            if head_adaptive_capacity[head_idx] >= head_attn_score.shape[-1]:
                # not compress this head
                selected_k = head_key_states
                selected_v = head_value_states

                l = selected_k.shape[0]
                new_head_lens.append(l)

                heads_key_states.append(selected_k.view(-1, head_dim))
                heads_value_states.append(selected_v.view(-1, head_dim))
                continue

            cache_index = torch.topk(head_attn_score,k=head_adaptive_capacity[head_idx],dim=-1).indices

            l = cache_index.shape[-1] + self.window_size
            new_head_lens.append(l)

            cache_index = cache_index.to(torch.long).unsqueeze(-1).expand(-1, head_dim)
            top_Kcache = head_key_states.gather(dim=0,index=cache_index)
            top_Vcache = head_value_states.gather(dim=0,index=cache_index)
            selected_k = torch.cat([top_Kcache, head_key_states[-self.window_size:, :]],dim=0)
            selected_v = torch.cat([top_Vcache, head_value_states[-self.window_size:, :]],dim=0)

            # NOTE: flatten view
            heads_key_states.append(selected_k.view(-1, head_dim))
            heads_value_states.append(selected_v.view(-1, head_dim))

        # NOTE: compose as flatten view
        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)

        head_lens = torch.tensor(new_head_lens, device=_device, dtype=torch.int32)
        cu_klen = torch.cat([torch.tensor([0], device=_device, dtype=torch.int32), torch.cumsum(head_lens, dim=0)], dim=0).to(torch.int32)

        return heads_key_states, heads_value_states, head_lens, cu_klen
