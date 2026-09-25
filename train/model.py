"""
Loading OpenVLA for fine-tuning: the checkpoint's processor, the model (or a tiny random-init
stand-in with the same architecture), and the LoRA adapter on top.
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name, device):
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_processor(checkpoint):
    """Tokenizer + image processor of the checkpoint (resize to 224, per-backbone normalization)."""
    return AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)


def build_tiny(checkpoint):
    """A random-init OpenVLA with tiny ViTs and a 2-layer Llama, built from the checkpoint's own model code
    and config so the tokenizer, processor, prompt and action tokens are exactly the real ones."""
    OpenVLAConfig = get_class_from_dynamic_module("configuration_prismatic.OpenVLAConfig", checkpoint)
    OpenVLA = get_class_from_dynamic_module("modeling_prismatic.OpenVLAForActionPrediction", checkpoint)
    base = OpenVLAConfig.from_pretrained(checkpoint)
    text = base.text_config
    config = OpenVLAConfig(
        vision_backbone_id=base.vision_backbone_id,
        llm_backbone_id=base.llm_backbone_id,
        arch_specifier=base.arch_specifier,
        use_fused_vision_backbone=base.use_fused_vision_backbone,
        image_resize_strategy=base.image_resize_strategy,
        text_config={
            "model_type": text.model_type, "vocab_size": text.vocab_size, "pad_token_id": text.pad_token_id,
            "max_position_embeddings": text.max_position_embeddings,
            "hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 4,
        },
        pad_token_id=base.pad_token_id,
        pad_to_multiple_of=base.pad_to_multiple_of,
        n_action_bins=base.n_action_bins,
        norm_stats={},
    )
    config.timm_model_ids = ["vit_tiny_patch16_224"] * len(base.timm_model_ids)
    config.timm_override_act_layers = [None] * len(base.timm_model_ids)
    # Register the remote classes so save_pretrained copies the model code next to the weights.
    OpenVLAConfig.register_for_auto_class()
    OpenVLA.register_for_auto_class("AutoModelForVision2Seq")
    return OpenVLA(config)


def load_base(model_cfg, device, dtype):
    """The model to fine-tune, on `device`, before LoRA."""
    checkpoint = model_cfg["checkpoint"]
    if model_cfg["tiny"]:
        model = build_tiny(checkpoint).to(device=device, dtype=dtype)
    elif model_cfg["load_in_4bit"]:
        if device.type != "cuda":
            raise SystemExit("model.load_in_4bit needs a CUDA device (bitsandbytes)")
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        model = AutoModelForVision2Seq.from_pretrained(
            checkpoint, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
            ),
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            checkpoint, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True,
        ).to(device)

    if model_cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model


def add_lora(model, lora_cfg):
    """LoRA on every linear layer except the output head, initialized as in OpenVLA's fine-tuning script."""
    from peft import LoraConfig, get_peft_model

    return get_peft_model(model, LoraConfig(
        r=lora_cfg["rank"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
        target_modules="all-linear", init_lora_weights="gaussian",
    ))


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
