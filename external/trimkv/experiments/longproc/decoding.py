# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import gc
import torch
from transformers import GenerationConfig, SinkCache
from vllm import SamplingParams, TokensPrompt


class GreedySearch_vLLM:
    def __init__(self, llm, tokenizer, is_kv_compress: bool = False):
        self.llm = llm
        self.tokenizer = tokenizer
        self.is_kv_compress = is_kv_compress

    def generate(self, prompt, max_tokens=100, temperature=0.0, top_p=1.0, sampling_kwargs={}, tokenizer_kwargs={}):
        if self.is_kv_compress:
            sampling_kwargs.update({
                "max_cache_tokens": 4096,
                "protected_window_size": 32,
                "metric_collection_buffer_size": 0,
                "compress_once": True,
            })

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **sampling_kwargs,
        )

        prompt = [{"role": "user", "content": prompt}]
        # NOTE: pass in token ids which we find returns better results closer to hf generate
        token_prompt = TokensPrompt(prompt_token_ids=self.tokenizer.apply_chat_template(
            conversation=prompt,
            add_generation_prompt=True,
            tokenize=True,
            **tokenizer_kwargs,
        ))

        outputs = self.llm.generate(
            prompts=[token_prompt],
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        output = outputs[0].outputs[0].text
        return output


class GreedySearch:
    def __init__(self, llm, tokenizer):
        llm.eval()
        self.device = llm.device
        self.llm = llm
        self.tokenizer = tokenizer
        self.past_kv = None
        self.add_eos_to_next_prompt = False

    def clear(self):
        self.past_kv = None
        gc.collect()
        torch.cuda.empty_cache()

    def generate(self, prompt, max_tokens=100, temperature=0.0, top_p=1.0, sampling_kwargs={}, tokenizer_kwargs={}):
        prompt = [{"role": "user", "content": prompt}]
        # NOTE: pass in token ids which we find returns better results closer to hf generate
        token_prompt = TokensPrompt(prompt_token_ids=self.tokenizer.apply_chat_template(
            conversation=prompt,
            add_generation_prompt=True,
            tokenize=True,
            **tokenizer_kwargs,
        ))

        input_ids = torch.tensor(token_prompt['prompt_token_ids']).int().unsqueeze(0).to(self.device)

        with torch.inference_mode():
            result = self._decode(
                input_ids,
                max_tokens=max_tokens,
            )

        output = self.tokenizer.decode(result[0, len(input_ids[0]) :])
        torch.cuda.empty_cache()
        self.clear()

        return output

    def _encode(self, input_ids, max_tokens=None):
        if self.past_kv is None:
            past_key_values = self.llm.prepare_inputs_for_generation(input_ids)[
                "past_key_values"
            ]
        else:
            past_key_values = self.past_kv

        out = self.llm(
            input_ids=input_ids,
            # attention_mask=torch.ones_like(input_ids),
            use_cache=True,
            return_dict=True,
            past_key_values=past_key_values,
            num_logits_to_keep=1,
        )
        _, past_key_values = out.logits, out.past_key_values

        self.past_kv = past_key_values

    def _decode(
        self,
        input_ids,
        max_tokens=100,
        extra_end_token_ids=[],
        dense_prefix=False,
        update_global_past_kv=True,
        disable_golden_context=False,
    ):
        if input_ids.dim() == 1:
            input_ids = input_ids[None, :]
        input_ids = input_ids.cuda()
        assert input_ids.size(0) == 1
        end_token_ids = (
            extra_end_token_ids
            + [self.tokenizer.eos_token_id]
            + [self.llm.config.eos_token_id]
        )

        logits = None
        if self.past_kv is None:
            model_inputs = {}
            self.llm._prepare_cache_for_generation(
                GenerationConfig(), model_inputs, None, None, None, None
            )
            past_key_values = model_inputs["past_key_values"]
        else:
            past_key_values = self.past_kv

        if not update_global_past_kv:
            self.global_kv_update_mode(False)

        for i in range(max_tokens):
            if i == 0:  # prefilling
                start_timer = torch.cuda.Event(enable_timing=True)
                end_timer = torch.cuda.Event(enable_timing=True)

                start_timer.record()
                out = self.llm(
                    input_ids=input_ids,
                    use_cache=True,
                    return_dict=True,
                    past_key_values=past_key_values,
                    num_logits_to_keep=1,
                )
                logits, past_key_values = out.logits, out.past_key_values
                end_timer.record()

                torch.cuda.synchronize()
                print(f"Prefill time: {start_timer.elapsed_time(end_timer)} ms")

            else:  # decoding
                if (
                    not disable_golden_context
                ):  # if use golden context, then decoding should not update global past_kv
                    self.global_kv_update_mode(False)
                out = self.llm(
                    input_ids=input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                logits, past_key_values = out.logits, out.past_key_values

            logits = logits[:, -1, :]
            word = logits.argmax(dim=-1)
            if word.item() in end_token_ids or i == max_tokens:
                break

            input_ids = torch.cat(
                (input_ids, word.to(input_ids.device).view(1, 1)), dim=-1
            )

        if not update_global_past_kv or not disable_golden_context:
            self.global_kv_update_mode(True)
            past_key_values.clear_temp_kv_cache()

        self.past_kv = past_key_values
        return input_ids

    def global_kv_update_mode(self, mode):
        try:
            attn_class = self.llm.model.layers[0].self_attn.__class__
        except:
            attn_class = self.llm.transformer.encoder.layers[
                0
            ].self_attention.__class__
        self.llm.apply(
            lambda m: setattr(m, "update_global_past_kv", mode)
            if isinstance(m, attn_class)
            else None
        )
