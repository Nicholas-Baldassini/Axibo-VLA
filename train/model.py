"""
Loading OpenVLA for fine-tuning: the checkpoint's processor, the model, and the LoRA adapter on top.
"""

from contextlib import contextmanager

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_utils import PreTrainedModel


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


@contextmanager
def _allow_quantized_to():
    """Let accelerate move a 4-bit model during loading.

    transformers 4.40.1, the version OpenVLA's checkpoint code targets, refuses `.to()` on a
    bitsandbytes model. accelerate >= 1.x ends dispatch_model with `model.to(device)` whenever every
    module lands on a single GPU, so loading in 4-bit raises before it returns. Pinning cannot
    reconcile them: lerobot requires accelerate >= 1.10 and train/data.py needs lerobot to read the
    dataset, so both live in one environment. device_map has already placed every weight on the
    right device by this point, which makes that call a no-op; only the guard has to stand aside.
    """
    guarded = PreTrainedModel.to
    PreTrainedModel.to = torch.nn.Module.to
    try:
        yield
    finally:
        PreTrainedModel.to = guarded


def load_base(model_cfg, device, dtype):
    """The model to fine-tune, on `device`, before LoRA."""
    checkpoint = model_cfg["checkpoint"]
    if model_cfg["load_in_4bit"]:
        if device.type != "cuda":
            raise SystemExit("model.load_in_4bit needs a CUDA device (bitsandbytes)")
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        with _allow_quantized_to():
            model = AutoModelForVision2Seq.from_pretrained(
                checkpoint, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True,
                # Place every module on the one GPU up front, so nothing has to be moved later.
                device_map={"": device.index or 0},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4",
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
