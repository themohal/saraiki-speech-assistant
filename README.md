# Saraiki Speech Assistant

A Streamlit app built from `saraiki_speech_assistant.ipynb`. Hold the button,
speak Saraiki, release — the app transcribes, answers in Saraiki, and speaks the
answer back.

```
mic (hold)  ->  Whisper STT          ->  Gemini              ->  OmniVoice TTS  ->  autoplay
                themohal/saraiki-      gemini-2.5-flash          k2-fsa/OmniVoice
                whisper-small          (strict Saraiki prompt)   voice = ref_audio.wav
```

## Setup

```powershell
python -m pip install -r requirements.txt
```

Add your Gemini API key ([get one here](https://aistudio.google.com/apikey)) — either:

```powershell
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
# then edit secrets.toml
```

or set `$env:GOOGLE_API_KEY = "..."` before launching.

## Run

```powershell
streamlit run app.py
```

Then open http://localhost:8501, allow microphone access, and **hold** the green
button while you speak. Release and the reply plays automatically. The spacebar
works as a hold key too, and there's a text box if you'd rather type Saraiki.

> The browser only grants microphone access on `localhost` or over HTTPS. If you
> serve this on a LAN address, put it behind TLS or the mic stays blocked.

## Files

| Path | Purpose |
|---|---|
| `app.py` | Streamlit UI, conversation history, autoplay |
| `assistant/pipeline.py` | STT / Gemini / TTS stages with cached model loading |
| `assistant/prompt.py` | The Saraiki system prompt, generated from the notebook |
| `assistant/mic.py` | Push-and-hold mic component wrapper |
| `assistant/frontend/index.html` | The hold-to-record button (MediaRecorder + level meter) |
| `assistant/audio_utils.py` | Decode/resample recorded audio, WAV encoding |
| `assistant/config.py` | Model IDs, sample rates, API key lookup |
| `_gen_prompt.py` | Regenerates `assistant/prompt.py` from the notebook |

`assistant/prompt.py` is generated rather than hand-written because the prompt
contains Saraiki-specific characters (ݙ ڄ ڳ ٻ ڃ ڦ ۏ) that are easy to corrupt by
retyping. If you edit the prompt in the notebook, re-run `python _gen_prompt.py`.

## Notes and differences from the notebook

- **Device**: the notebook hardcodes `device_map="cuda:0"` with `float16`.
  `assistant/pipeline.py` detects CUDA and falls back to CPU + `float32`, so it
  runs without a GPU — just slowly. OmniVoice is a diffusion TTS model, so
  expect roughly a minute or more per reply on CPU.
- **Memory**: OmniVoice on CPU in `float32` needs well over the 7.5 GB of RAM on
  this machine once Whisper is also loaded. A CUDA GPU is strongly recommended;
  `load_asr=False` (the notebook used `True`) saves what it can by skipping
  OmniVoice's bundled ASR, since Whisper already covers transcription.
- **First run** downloads several GB of weights from Hugging Face and will look
  like it's hanging. Watch the terminal for progress.
- **WebM decoding**: browsers record WebM/Opus, which libsndfile can't read.
  `audio_utils` falls back to librosa, which needs a working audio backend —
  install **ffmpeg** and put it on PATH if you get a decode error.
- The reference clip drives the output voice; swap `ref_audio.wav` (24 kHz mono)
  or point `SARAIKI_REF_AUDIO` elsewhere to change it.

## Configuration

Override via environment variables: `GOOGLE_API_KEY`, `SARAIKI_STT_MODEL`,
`SARAIKI_TTS_MODEL`, `SARAIKI_LLM_MODEL`, `SARAIKI_REF_AUDIO`.
