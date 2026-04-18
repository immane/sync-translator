import os

# Adjustable parameters
RECORD_DURATION = int(os.environ.get("RECORD_DURATION", 5))
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", 16000))
BEAM_SIZE = int(os.environ.get("BEAM_SIZE", 10))
TRANSCRIBE_LANGUAGE = os.environ.get("TRANSCRIBE_LANGUAGE", "zh")

# VAD settings
VAD_MODE = int(os.environ.get("VAD_MODE", 1))
VAD_FRAME_MS = int(os.environ.get("VAD_FRAME_MS", 30))
VAD_END_SILENCE_FRAMES = int(os.environ.get("VAD_END_SILENCE_FRAMES", 10))
VAD_MAX_WAIT_SEC = int(os.environ.get("VAD_MAX_WAIT_SEC", 5))

PLAYBACK_TIMEOUT = int(os.environ.get("PLAYBACK_TIMEOUT", 60))
ENERGY_VAD_THRESHOLD = int(os.environ.get("ENERGY_VAD_THRESHOLD", 220))

MODEL_PATH = os.environ.get("MODEL_PATH", "./models/small")
TMP_DIR = os.path.join(os.path.dirname(__file__), "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

# TTS
TTS_VOICE = os.environ.get("TTS_VOICE", "en-US-JennyNeural")

# Recording mode
RECORD_MODE = os.environ.get("RECORD_MODE", "default")
RECORD_STRIDE = float(os.environ.get("RECORD_STRIDE", "1.0"))

# GPU/compute config
USE_GPU = os.environ.get("USE_GPU", "0").lower() in ("1", "true", "yes")
MODEL_DEVICE = os.environ.get("MODEL_DEVICE", "cuda" if USE_GPU else "cpu")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16" if MODEL_DEVICE == "cuda" else "int8")
