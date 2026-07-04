from transformers import LlamaConfig, AutoConfig, PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING


class TrimKVLlavaTextConfig(LlamaConfig):
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


class TrimKVLlavaConfig(PretrainedConfig):
    model_type = "llava"
    attribute_map = {
        "image_token_id": "image_token_index",
    }
    sub_configs = {"text_config": TrimKVLlavaTextConfig, "vision_config": AutoConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=32000,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        image_seq_length=576,
        multimodal_projector_bias=True,
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
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.image_seq_length = image_seq_length
        self.global_capacity = global_capacity

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}"
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer

        if isinstance(vision_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "clip_vision_model")
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["clip_vision_model"](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=336,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                projection_dim=768,
            )

        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "llama")
            text_config = TrimKVLlavaTextConfig(**text_config)
        elif text_config is None:
            text_config = TrimKVLlavaTextConfig()

        self.text_config = text_config
        self.multimodal_projector_bias = multimodal_projector_bias

        super().__init__(**kwargs)

    def update_text_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.text_config, key, value)


__all__ = ["TrimKVLlavaConfig", "TrimKVLlavaTextConfig"]
