import os
import time
from config import MODEL_PATH, BEAM_SIZE, TRANSCRIBE_LANGUAGE, RECORD_MODE, RECORD_DURATION, RECORD_STRIDE, MODEL_DEVICE, COMPUTE_TYPE
from audio import trim_silence
from faster_whisper import WhisperModel


def load_model():
    print(f"Loading model at '{MODEL_PATH}' device={MODEL_DEVICE} compute_type={COMPUTE_TYPE}...")
    model = None
    candidates = []
    if MODEL_DEVICE == "cuda":
        candidates.append(("cuda", COMPUTE_TYPE))
        if COMPUTE_TYPE == "float16":
            candidates.append(("cuda", "float32"))
        candidates.append(("cpu", "int8"))
        candidates.append(("cpu", "float32"))
    else:
        candidates.append(("cpu", COMPUTE_TYPE))
        candidates.append(("cpu", "int8"))
        candidates.append(("cpu", "float32"))

    for dev, comp in candidates:
        try:
            model = WhisperModel(MODEL_PATH, device=dev, compute_type=comp)
            print(f"Model loaded successfully on device={dev} compute_type={comp}")
            return model
        except Exception as e:
            print(f"Model load failed for device={dev} compute_type={comp}: {e}")
    raise RuntimeError("Failed to load model with available configurations")


def transcribe_file(model, wav_path, beam_size=BEAM_SIZE, language=TRANSCRIBE_LANGUAGE, vad_filter=True, condition_on_previous_text=False):
    segments, _ = model.transcribe(
        wav_path,
        beam_size=beam_size,
        language=language,
        vad_filter=vad_filter,
        condition_on_previous_text=condition_on_previous_text,
    )
    texts = [s.text.strip() for s in segments if s.text.strip()]
    return texts
