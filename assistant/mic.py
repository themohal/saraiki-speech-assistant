"""Push-and-hold microphone component.

streamlit-mic-recorder is click-to-start / click-to-stop, so this wraps a small
static HTML+JS component that records while the button is held and returns the
clip on release.
"""
from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
_component = components.declare_component("saraiki_push_to_talk", path=str(_FRONTEND_DIR))


def push_to_talk(key: str = "ptt", height: int = 120) -> dict | None:
    """Render the hold-to-record button.

    Returns ``{"bytes": ..., "mime": ..., "id": ...}`` once per new recording,
    and None on the reruns in between.
    """
    payload = _component(key=key, default=None, height=height)
    if not payload or not payload.get("audio_base64"):
        return None

    clip_id = payload.get("id")
    seen_key = f"_{key}_last_id"
    if st.session_state.get(seen_key) == clip_id:
        # Same clip replayed on a rerun - already handled.
        return None
    st.session_state[seen_key] = clip_id

    return {
        "bytes": base64.b64decode(payload["audio_base64"]),
        "mime": payload.get("mime", "audio/webm"),
        "id": clip_id,
    }
