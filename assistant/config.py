"""Shared configuration for the Saraiki speech assistant."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Models, matching the notebook.
STT_MODEL = os.getenv("SARAIKI_STT_MODEL", "themohal/saraiki-whisper-small")
# bf16 community mirror of k2-fsa/OmniVoice: identical file layout and
# architecture, but 2.0 GB instead of 3.3 GB - it fits Streamlit Cloud's disk
# and memory budget. Set SARAIKI_TTS_MODEL=k2-fsa/OmniVoice for the original.
TTS_MODEL = os.getenv("SARAIKI_TTS_MODEL", "drbaph/OmniVoice-bf16")
LLM_MODEL = os.getenv("SARAIKI_LLM_MODEL", "gemini-2.5-flash")

# OmniVoice speaks Saraiki by cloning the timbre of this reference clip.
REF_AUDIO = Path(os.getenv("SARAIKI_REF_AUDIO", PROJECT_ROOT / "ref_audio.wav"))

# OmniVoice emits 24 kHz audio (see notebook: sf.write(..., 24000)).
TTS_SAMPLE_RATE = 24000

# Whisper wants 16 kHz mono.
STT_SAMPLE_RATE = 16000


def google_api_key() -> str | None:
    """Read the Gemini key from the environment or Streamlit secrets."""
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        return key.strip()
    # st.secrets raises if no secrets file exists, so guard it.
    try:
        import streamlit as st

        return str(st.secrets["GOOGLE_API_KEY"]).strip()
    except Exception:
        return None
