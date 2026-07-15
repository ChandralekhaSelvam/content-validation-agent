"""
Voice classifier: scores a draft on two axes using a fine-tuned small model
(Llama 3.2 3B + LoRA, same base as slm-toolcall-lora):

  1. voice_score       - does this sound like the author's own writing?
  2. ai_sounding_score  - does this read as generic AI-generated copy?

This is the piece that reuses your existing fine-tuning work but for a new
task: classification/scoring instead of tool-call generation. Same base
model, same PEFT/Unsloth training pattern, different LoRA adapter.

Falls back to a rule-based heuristic when no adapter has been trained yet
(ADAPTER_PATH doesn't exist), so this repo runs end-to-end before you've
done any fine-tuning — train the real adapter with train_voice_lora.py
whenever you're ready, and swap it in with zero changes elsewhere.
"""

import os
import re
from pathlib import Path
from typing import Tuple

ADAPTER_PATH = os.environ.get("VOICE_LORA_ADAPTER_PATH", "slm/adapters/voice-lora")
BASE_MODEL = os.environ.get("VOICE_BASE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")

_AI_TELLTALE_PHRASES = [
    r"\bin today'?s (?:fast-paced|ever-evolving|digital)\b",
    r"\blet'?s dive in\b",
    r"\bunlock(?:ing)? the (?:power|potential) of\b",
    r"\bit'?s not just .*, it'?s\b",
    r"\bat the end of the day\b",
    r"\bgame[- ]changer\b",
    r"\btake it to the next level\b",
    r"\bin conclusion,\b",
    r"\bwhether you'?re .* or .*,\b",
]

_model_cache = {}


def _adapter_available() -> bool:
    return Path(ADAPTER_PATH).exists()


def _load_model():
    """Lazy-load the base model + LoRA adapter. Cached across calls."""
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["tokenizer"]

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="auto", load_in_4bit=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    _model_cache["model"] = model
    _model_cache["tokenizer"] = tokenizer
    return model, tokenizer


def _score_with_slm(text: str) -> Tuple[float, float]:
    """Real path: ask the fine-tuned SLM to emit a structured score."""
    import torch

    model, tokenizer = _load_model()
    prompt = (
        "Rate this LinkedIn/Medium draft on two 0-1 scores.\n"
        "voice_score: does it sound like a specific human's authentic writing?\n"
        "ai_sounding_score: does it read as generic AI-generated copy?\n"
        f"Respond ONLY as JSON: {{\"voice_score\": <float>, \"ai_sounding_score\": <float>}}\n\n"
        f"Draft:\n{text}"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    raw = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    import json
    try:
        parsed = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        return float(parsed["voice_score"]), float(parsed["ai_sounding_score"])
    except (ValueError, KeyError, json.JSONDecodeError):
        # Model returned malformed output — fall back rather than crash the graph
        return _score_with_heuristic(text)


def _score_with_heuristic(text: str) -> Tuple[float, float]:
    """
    Fallback path: no trained adapter yet. Cheap, deterministic, no GPU needed.
    Not a substitute for the real classifier — just keeps the graph runnable.
    """
    lowered = text.lower()
    hits = sum(1 for pattern in _AI_TELLTALE_PHRASES if re.search(pattern, lowered))
    ai_sounding_score = min(1.0, hits * 0.25)

    # Very rough voice proxy: sentence length variance. Human writing tends to
    # vary sentence length more than templated AI output. This is a stand-in
    # only — the real signal should come from the trained classifier.
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    lengths = [len(s.split()) for s in sentences] or [0]
    mean_len = sum(lengths) / len(lengths)
    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    voice_score = min(1.0, variance / 40)

    return voice_score, ai_sounding_score


def score_draft(text: str) -> Tuple[float, float]:
    """Public entry point. Returns (voice_score, ai_sounding_score)."""
    if _adapter_available():
        return _score_with_slm(text)
    return _score_with_heuristic(text)
