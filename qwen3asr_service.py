# qwen3asr_service.py
import torch
from qwen_asr import Qwen3ASRModel
from peft import PeftModel

BASE_MODEL   = "Qwen/Qwen3-ASR-1.7B"         # 或你的本地路径
CHECKPOINT_DIR = "/home/wuguinan/Qwen3-asr-finetuning/qlora_adapter/zh-epoch1*"   # 全量微调 or LoRA adapter 目录
DEVICE = "cuda:0"

_model = None

def get_model():
    global _model
    if _model is not None:
        return _model

    # 判断是全量微调还是 LoRA adapter
    import os
    is_lora = os.path.exists(os.path.join(CHECKPOINT_DIR, "adapter_config.json"))

    wrapper = Qwen3ASRModel.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="cpu"
    )

    if is_lora:
        wrapper.model = PeftModel.from_pretrained(
            wrapper.model, CHECKPOINT_DIR, device_map="cpu"
        )
    else:
        # 全量微调：加载 safetensors 权重
        from safetensors.torch import load_file
        sd = load_file(os.path.join(CHECKPOINT_DIR, "model.safetensors"))
        wrapper.model.load_state_dict(sd, strict=False)

    wrapper.model = wrapper.model.to(DEVICE)
    wrapper.model.eval()
    _model = wrapper
    return _model


def transcribe(audio_path: str) -> str:
    """对音频文件做语音识别，返回文字字符串"""
    model = get_model()
    results = model.transcribe(audio=audio_path)

    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, dict):
            text = first.get("text") or first.get("transcription") or ""
        else:
            text = getattr(first, "text", str(first))
    else:
        text = str(results)

    return text.strip()