from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLTextConfig,
    Qwen2_5_VLVisionConfig,
)

class TrimKVQwen2_5_VLTextConfig(Qwen2_5_VLTextConfig):
    def __init__(
        self,
        retention_gate_bias_init=10.0,
        retention_gate='rg',
        attn_impl='rg_attn_flex',
        max_seq_len=20480,
        retention_gate_intermediate_size=512,
        tie_retention_gate_layers=True,
        **kwargs,
    ):
        self.retention_gate_bias_init = retention_gate_bias_init
        self.retention_gate_intermediate_size = retention_gate_intermediate_size
        self.retention_gate = retention_gate
        self.attn_impl = attn_impl
        self.max_seq_len = max_seq_len
        self.tie_retention_gate_layers = tie_retention_gate_layers
        super().__init__(
            **kwargs,
        )

class TrimKVQwen2_5_VLConfig(Qwen2_5_VLConfig):
    model_type = "qwen2_5_vl"
    sub_configs = {"vision_config": Qwen2_5_VLVisionConfig, "text_config": TrimKVQwen2_5_VLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        memory_size=1024,
        retention_weight=1.0,
        base_loss='fwkl',
        logit_block_size=-1,
        trainable_params=None,
        global_capacity=True,
        **kwargs,
    ):
        self.base_loss = base_loss
        self.retention_weight = retention_weight
        self.memory_size = memory_size
        self.logit_block_size = logit_block_size
        self.trainable_params = trainable_params
        self.global_capacity = global_capacity
        super().__init__(**kwargs)

    def update_text_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.text_config, key, value)

__all__ = ["TrimKVQwen2_5_VLConfig", "TrimKVQwen2_5_VLTextConfig"]
