# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging
from transformers import LlamaConfig


logger = logging.get_logger(__name__)


class TrimKVLlamaConfig(LlamaConfig):
    def __init__(
        self,
        retention_gate_bias_init=10.0,
        retention_weight=1.0,
        memory_size=2048,
        retention_gate='rg',
        base_loss='fwkl',
        attn_impl='rg_attn_flex',
        trainable_params=None,
        max_seq_len=20480,
        retention_gate_intermediate_size=512,
        tie_retention_gate_layers=True,
        global_capacity=True,
        **kwargs,
    ):
        self.retention_gate_bias_init = retention_gate_bias_init
        self.retention_weight = retention_weight
        self.retention_gate_intermediate_size = retention_gate_intermediate_size
        self.memory_size = memory_size
        self.retention_gate = retention_gate
        self.attn_impl = attn_impl
        self.base_loss = base_loss
        self.trainable_params = trainable_params
        self.max_seq_len = max_seq_len
        self.tie_retention_gate_layers = tie_retention_gate_layers
        self.global_capacity = global_capacity

        super().__init__(
            **kwargs,
        )


__all__ = ["TrimKVLlamaConfig"]
