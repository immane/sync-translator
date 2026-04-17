# sync-translator

Real-time speech translator. Listens for Chinese speech, transcribes it with [Whisper](https://github.com/openai/whisper), translates to English via Google Translate, and reads the result aloud using [Edge TTS](https://github.com/rany2/edge-tts).

```
[Microphone] → VAD → Whisper ASR → Google Translate → Edge TTS → [Speaker]
```

---

## Features

- **Real-time pipeline** — records, transcribes, translates, and speaks with minimal delay
- **Local Whisper model** — runs fully offline for ASR; no API key required
- **Smart VAD** — uses WebRTC VAD to detect speech onset/end; falls back to energy-based detection when unavailable
- **Silence trimming** — strips leading/trailing silence before inference for faster, cleaner transcription
- **Non-overlapping playback** — waits for TTS to finish before the next recording cycle to avoid self-feedback
- **Configurable** — beam size, language, voice, VAD aggressiveness, etc. are all top-of-file constants

---

## Requirements

- **Python 3.9+**
- **FFmpeg** (for `ffplay` playback) — [download](https://ffmpeg.org/download.html) and add to `PATH`
- A Whisper model in CTranslate2 format placed under `./models/` (see [Model Setup](#model-setup))

---

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt
```

> **Windows note — `webrtcvad`**: building the wheel requires the [MSVC build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). If the install fails, remove `webrtcvad` from `requirements.txt`; the program will fall back to energy-based VAD automatically.

---

## Model Setup

`faster-whisper` expects models in [CTranslate2](https://github.com/OpenNMT/CTranslate2) format. Each model directory must contain:

```
models/
└── small/          ← set model_path in translator.py to this directory
    ├── config.json
    ├── model.bin
    ├── tokenizer.json
    └── vocabulary.txt
```

Download Systran/faster-whisper-small(https://huggingface.co/Systran/faster-whisper-small).

### Option A — Let faster-whisper download automatically

Change the `model_path` line in `translator.py`:

```python
model_path = "small"   # downloads whisper-small from Hugging Face on first run
```

Supported sizes: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`.

### Option B — Convert a local Whisper checkpoint

```bash
pip install ct2-opus-mt-forward-converter   # or use ct2-transformers-converter
ct2-transformers-converter --model openai/whisper-small --output_dir models/small --quantization int8
```

---

## Usage

```bash
.venv\Scripts\python.exe translator.py
```

Speak in Chinese — the translated English text is printed and read aloud.

### TTS voice

Override the default voice (`en-US-JennyNeural`) with the `TTS_VOICE` environment variable:

```powershell
$env:TTS_VOICE = "en-US-AriaNeural"
.venv\Scripts\python.exe translator.py
```

Browse available voices:

```bash
edge-tts --list-voices
```

---

## Configuration

All tunable constants are at the top of `translator.py`:

| Constant | Default | Description |
|---|---|---|
| `RECORD_DURATION` | `5` | Max recording length (seconds) |
| `SAMPLE_RATE` | `16000` | Microphone sample rate (Hz) |
| `BEAM_SIZE` | `10` | Whisper beam search width — higher = more accurate but slower |
| `TRANSCRIBE_LANGUAGE` | `"zh"` | ASR language hint |
| `VAD_MODE` | `1` | WebRTC VAD aggressiveness (0–3) |
| `VAD_END_SILENCE_FRAMES` | `10` | Consecutive silent frames before stopping recording |
| `VAD_MAX_WAIT_SEC` | `5` | Max seconds to wait for speech before skipping |
| `PLAYBACK_TIMEOUT` | `60` | Watchdog timeout (s) to unblock if TTS hangs |
| `ENERGY_VAD_THRESHOLD` | `220` | Fallback energy gate when webrtcvad is unavailable |

---

## How It Works

```
1. Wait until TTS playback is idle (play_event cleared)
2. Record audio via sounddevice + WebRTC VAD
   - Skip frames until speech is detected (up to VAD_MAX_WAIT_SEC)
   - Stop after VAD_END_SILENCE_FRAMES of trailing silence
3. Trim leading/trailing silence (energy-based)
4. Transcribe with faster-whisper (int8, CPU)
5. Translate each segment via Google Translate (no API key)
6. Print original + translation, then speak via Edge TTS + ffplay
7. Return to step 1
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `faster-whisper` | CTranslate2-accelerated Whisper inference |
| `sounddevice` | Cross-platform microphone capture |
| `numpy` | Audio buffer processing / silence trimming |
| `webrtcvad` | WebRTC-based voice activity detection |
| `requests` | HTTP calls to Google Translate |
| `edge-tts` | Microsoft Edge neural TTS |
| `pygame-ce` | Fallback audio playback (if ffplay unavailable) |
| `rich` | Colored terminal output |

---

## License

MIT
