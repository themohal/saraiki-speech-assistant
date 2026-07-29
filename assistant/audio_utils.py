"""Audio decoding helpers.

The browser hands us whatever container MediaRecorder produced (usually
WebM/Opus), while Whisper wants a 16 kHz mono float32 waveform. soundfile
handles WAV natively; anything else goes through librosa/audioread.
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import STT_SAMPLE_RATE


def _to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples.astype(np.float32, copy=False)


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear resample - adequate for speech fed to Whisper."""
    if src_rate == dst_rate or samples.size == 0:
        return samples
    duration = samples.size / float(src_rate)
    dst_len = max(1, int(round(duration * dst_rate)))
    src_positions = np.linspace(0.0, samples.size - 1, num=samples.size, dtype=np.float64)
    dst_positions = np.linspace(0.0, samples.size - 1, num=dst_len, dtype=np.float64)
    return np.interp(dst_positions, src_positions, samples).astype(np.float32)


def decode_to_waveform(raw: bytes, target_rate: int = STT_SAMPLE_RATE) -> np.ndarray:
    """Decode arbitrary recorded audio bytes into mono float32 at target_rate."""
    if not raw:
        return np.zeros(0, dtype=np.float32)

    # Fast path: WAV/FLAC/OGG that libsndfile understands directly.
    try:
        samples, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        return _resample(_to_mono(samples), rate, target_rate)
    except Exception:
        pass

    # Fallback for WebM/Opus and friends. librosa needs a real file path here.
    tmp_path: Path | None = None
    try:
        import librosa

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)
        samples, rate = librosa.load(str(tmp_path), sr=target_rate, mono=True)
        return samples.astype(np.float32, copy=False)
    except Exception as exc:  # pragma: no cover - depends on local codecs
        raise RuntimeError(
            "Could not decode the recorded audio. Install ffmpeg and make sure it is "
            f"on PATH so WebM/Opus microphone input can be read. Original error: {exc}"
        ) from exc
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def waveform_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode a float32 waveform as WAV bytes for st.audio / download."""
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(samples, dtype=np.float32), sample_rate, format="WAV")
    return buffer.getvalue()


def duration_seconds(samples: np.ndarray, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return float(np.asarray(samples).size) / float(sample_rate)


def is_silent(samples: np.ndarray, threshold: float = 2e-3) -> bool:
    """True when the clip carries no real speech energy."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.size == 0:
        return True
    return float(np.sqrt(np.mean(np.square(arr)))) < threshold
