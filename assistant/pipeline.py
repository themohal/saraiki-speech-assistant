"""The three pipeline stages from the notebook: Whisper STT, Gemini, OmniVoice TTS.

Models are loaded lazily and cached for the life of the Streamlit process, so
the first request pays the download/warm-up cost and later ones do not.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from . import config
from .audio_utils import decode_to_waveform, is_silent
from .prompt import build_prompt

_lock = threading.Lock()
_stt_pipeline = None
_tts_model = None
_llm_client = None


# --------------------------------------------------------------------------
# Device selection
# --------------------------------------------------------------------------
def resolve_device() -> tuple[str, "object"]:
    """Pick the best available device.

    The notebook hardcodes ``cuda:0`` + float16; that crashes on a CPU-only
    box, so fall back to CPU + float32 when no GPU is present.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda:0", torch.float16
    return "cpu", torch.float32


def device_label() -> str:
    device, dtype = resolve_device()
    return f"{device} ({str(dtype).replace('torch.', '')})"


# --------------------------------------------------------------------------
# Stage 1: speech to text
# --------------------------------------------------------------------------
def load_stt():
    """Load the Saraiki Whisper ASR pipeline (cached)."""
    global _stt_pipeline
    with _lock:
        if _stt_pipeline is None:
            import torch
            from transformers import pipeline

            device, _ = resolve_device()
            _stt_pipeline = pipeline(
                "automatic-speech-recognition",
                model=config.STT_MODEL,
                device=device,
                torch_dtype=torch.float32,
            )
        return _stt_pipeline


def transcribe(audio_bytes: bytes) -> str:
    """Recorded bytes -> Saraiki text."""
    waveform = decode_to_waveform(audio_bytes, config.STT_SAMPLE_RATE)
    if is_silent(waveform):
        return ""
    asr = load_stt()
    result = asr({"raw": waveform, "sampling_rate": config.STT_SAMPLE_RATE})
    return str(result.get("text", "")).strip()


# --------------------------------------------------------------------------
# Stage 2: the Saraiki LLM reply
# --------------------------------------------------------------------------
@dataclass
class LlmResult:
    text: str
    ok: bool
    error: str | None = None


def load_llm():
    """Configure google-generativeai with the API key (cached)."""
    global _llm_client
    with _lock:
        if _llm_client is None:
            api_key = config.google_api_key()
            if not api_key:
                raise RuntimeError(
                    "GOOGLE_API_KEY not found. Put it in .streamlit/secrets.toml "
                    "or set it as an environment variable."
                )
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            _llm_client = genai.GenerativeModel(config.LLM_MODEL)
        return _llm_client


def generate_reply(user_text: str) -> LlmResult:
    """Ask Gemini for a reply, constrained to pure Saraiki by the notebook prompt."""
    if not user_text.strip():
        return LlmResult(text="", ok=False, error="Nothing was transcribed.")
    try:
        model = load_llm()
        response = model.generate_content(build_prompt(user_text))
        reply = (getattr(response, "text", "") or "").strip()
        if not reply:
            return LlmResult(text="", ok=False, error="The model returned an empty reply.")
        return LlmResult(text=reply, ok=True)
    except Exception as exc:
        return LlmResult(text="", ok=False, error=str(exc))


# --------------------------------------------------------------------------
# Stage 3: text to speech
# --------------------------------------------------------------------------
def load_tts():
    """Load OmniVoice (cached). This is the heavyweight download."""
    global _tts_model
    with _lock:
        if _tts_model is None:
            from omnivoice import OmniVoice

            device, dtype = resolve_device()
            _tts_model = OmniVoice.from_pretrained(
                config.TTS_MODEL,
                device_map=device,
                dtype=dtype,
                # The notebook loads OmniVoice's own ASR; we already have
                # Whisper for that, so skip it and save memory.
                load_asr=False,
            )
        return _tts_model


def synthesize(text: str) -> np.ndarray:
    """Saraiki text -> 24 kHz waveform in the reference speaker's voice."""
    if not text.strip():
        return np.zeros(0, dtype=np.float32)
    if not config.REF_AUDIO.exists():
        raise RuntimeError(
            f"Reference audio not found at {config.REF_AUDIO}. OmniVoice needs it "
            "to clone a Saraiki voice."
        )
    model = load_tts()
    audio = model.generate(
        text=text,
        language="saraiki",
        ref_audio=str(config.REF_AUDIO),
    )
    waveform = np.asarray(audio[0], dtype=np.float32).squeeze()
    return waveform
