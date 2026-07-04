from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


logger = logging.get_logger(__name__)


class TrimKVQwen3Config(Qwen3Config):
    def __init__(
        self,
        retention_gate_bias_init=10.0,
        retention_weight=1.0,
        memory_size=1024,
        retention_gate='rg',
        base_loss='fwkl',
        attn_impl='rg_attn_flex',
        trainable_params=None,
        max_seq_len=131072,
        retention_gate_intermediate_size=512,
        logit_block_size=-1,
        global_capacity=True,
        tie_retention_gate_layers=True,
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
        self.logit_block_size = logit_block_size
        self.global_capacity = global_capacity
        self.tie_retention_gate_layers = tie_retention_gate_layers
        super().__init__(
            **kwargs,
        )


__all__ = ["TrimKVQwen3Config"]
