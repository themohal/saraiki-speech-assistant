"""Saraiki Speech Assistant - Streamlit front end.

Hold the button, speak Saraiki, release. The app transcribes with a Saraiki
Whisper model, answers in Saraiki with Gemini, and speaks the answer back with
OmniVoice using ref_audio.wav as the voice.
"""
from __future__ import annotations

import base64
import time

import streamlit as st

from assistant import config, pipeline
from assistant.audio_utils import duration_seconds, waveform_to_wav_bytes
from assistant.mic import push_to_talk

st.set_page_config(page_title="Saraiki Speech Assistant", page_icon="🗣️", layout="centered")

STYLE = """
<style>
  .block-container { padding-top: 2.2rem; max-width: 780px; }
  .hero { text-align: center; margin-bottom: 1.4rem; }
  .hero h1 { margin: 0; font-size: 1.85rem; font-weight: 720; letter-spacing: -.4px; }
  .hero p { margin: .35rem 0 0; opacity: .62; font-size: .92rem; }
  .saraiki {
    direction: rtl; text-align: right;
    font-family: "Noto Naskh Arabic", "Jameel Noori Nastaleeq", "Segoe UI", serif;
    font-size: 1.32rem; line-height: 2.15; unicode-bidi: plaintext;
  }
  .turn { border-radius: 14px; padding: .85rem 1.15rem; margin: .5rem 0; }
  .turn.user { background: rgba(47,125,93,.10); border-right: 3px solid #2f7d5d; }
  .turn.bot  { background: rgba(120,120,120,.10); border-right: 3px solid #8a8a8a; }
  .turn .who {
    direction: ltr; text-align: left; font-size: .68rem; font-weight: 700;
    letter-spacing: .09em; text-transform: uppercase; opacity: .5; margin-bottom: .3rem;
  }
  .stAudio { margin-top: .3rem; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><h1>سرائیکی سپیچ اسسٹنٹ</h1>'
    "<p>Saraiki Speech Assistant &middot; hold the button, speak, release</p></div>",
    unsafe_allow_html=True,
)

if "turns" not in st.session_state:
    st.session_state.turns = []


def autoplay(wav_bytes: bytes) -> None:
    """Play the reply immediately - st.audio cannot autoplay on older versions."""
    b64 = base64.b64encode(wav_bytes).decode()
    st.markdown(
        f'<audio autoplay controls style="width:100%" src="data:audio/wav;base64,{b64}"></audio>',
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.subheader("Status")
    key_ok = bool(config.google_api_key())
    st.write("Gemini key:", "✅ found" if key_ok else "❌ missing")
    if not key_ok:
        st.caption("Add GOOGLE_API_KEY to .streamlit/secrets.toml or your environment.")
    st.write("Reference voice:", "✅ ready" if config.REF_AUDIO.exists() else "❌ missing")
    try:
        st.write("Compute:", f"`{pipeline.device_label()}`")
    except Exception:
        st.write("Compute:", "`torch not installed`")

    st.divider()
    st.caption(
        f"STT `{config.STT_MODEL}`  \nLLM `{config.LLM_MODEL}`  \nTTS `{config.TTS_MODEL}`"
    )
    speak_replies = st.toggle("Speak replies", value=True)
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.turns = []
        st.rerun()
    st.caption("First run downloads several GB of model weights.")

# ---------------------------------------------------------------- mic input
clip = push_to_talk(key="ptt")
st.caption("Tip: you can also hold the spacebar.")

typed = st.chat_input("...or type Saraiki text instead")

user_text: str | None = None

if clip:
    with st.status("Transcribing your Saraiki speech...", expanded=False) as status:
        try:
            user_text = pipeline.transcribe(clip["bytes"])
        except Exception as exc:
            status.update(label="Transcription failed", state="error")
            st.error(str(exc))
            user_text = None
        else:
            if user_text:
                status.update(label="Transcribed", state="complete")
            else:
                status.update(label="No speech detected", state="error")
                st.warning("I did not catch any speech - hold the button a little longer.")
                user_text = None
elif typed and typed.strip():
    user_text = typed.strip()

# --------------------------------------------------------- generate + speak
if user_text:
    turn: dict = {"user": user_text, "bot": None, "audio": None, "error": None}

    with st.spinner("Thinking in Saraiki..."):
        result = pipeline.generate_reply(user_text)

    if not result.ok:
        turn["error"] = result.error
    else:
        turn["bot"] = result.text
        if speak_replies:
            with st.spinner("Speaking the reply..."):
                try:
                    started = time.perf_counter()
                    waveform = pipeline.synthesize(result.text)
                    turn["audio"] = waveform_to_wav_bytes(waveform, config.TTS_SAMPLE_RATE)
                    turn["tts_seconds"] = round(time.perf_counter() - started, 1)
                    turn["audio_seconds"] = round(
                        duration_seconds(waveform, config.TTS_SAMPLE_RATE), 1
                    )
                except Exception as exc:
                    turn["error"] = f"Text was generated but speech synthesis failed: {exc}"

    st.session_state.turns.append(turn)

# ---------------------------------------------------------------- history
if not st.session_state.turns:
    st.info("Hold the green button and say something in Saraiki to begin.")

for index, turn in enumerate(st.session_state.turns):
    st.markdown(
        f'<div class="turn user"><div class="who">You</div>'
        f'<div class="saraiki">{turn["user"]}</div></div>',
        unsafe_allow_html=True,
    )
    if turn.get("bot"):
        st.markdown(
            f'<div class="turn bot"><div class="who">Assistant</div>'
            f'<div class="saraiki">{turn["bot"]}</div></div>',
            unsafe_allow_html=True,
        )
    if turn.get("error"):
        st.error(turn["error"])
    if turn.get("audio"):
        is_latest = index == len(st.session_state.turns) - 1
        if is_latest:
            autoplay(turn["audio"])
        else:
            st.audio(turn["audio"], format="audio/wav")
        st.download_button(
            "Download reply audio",
            data=turn["audio"],
            file_name=f"saraiki_reply_{index + 1}.wav",
            mime="audio/wav",
            key=f"dl_{index}",
        )
