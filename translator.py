import os
import asyncio
import threading
import subprocess
import time
import urllib.parse
import requests
from rich.console import Console
from rich.text import Text

from config import *
from audio import record_audio, record_fixed_chunk, trim_silence
from transcribe import load_model, transcribe_file
from tts import tts_and_play

# Globals
play_event = threading.Event()
console = Console()


def print_status(recog_state: str, recog_color: str):
    t = Text()
    t.append(recog_state, style=recog_color)
    console.print(t, end='\r')


def clear_status_line():
    pass


def clear_and_print(msg):
    clear_status_line()
    console.print(msg)


def translate_text(text):
    try:
        if not text:
            return ""
        encoded_text = urllib.parse.quote(text)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q={encoded_text}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            return response.json()[0][0][0]
    except Exception:
        return ""
    return ""


def main_process():
    model = load_model()

    if RECORD_MODE == "realtime_text":
        print("\n$ Sync Translator started (realtime_text mode: continuous recording, text-only)")
        while True:
            try:
                rec_path = os.path.join(TMP_DIR, "rec.wav")
                record_fixed_chunk(rec_path, duration=RECORD_DURATION, fs=SAMPLE_RATE)
                seg_texts = transcribe_file(model, rec_path, beam_size=BEAM_SIZE, language=TRANSCRIBE_LANGUAGE, vad_filter=False, condition_on_previous_text=True)
                text = " ".join(seg_texts).strip()
                if text:
                    clear_and_print(f">>> {text}")
            except Exception as e:
                clear_and_print(f"Error: {e}")
    else:
        print("\n$ Sync Translator is started (FFmpeg mode)")
        while True:
            try:
                while play_event.is_set():
                    print_status("Playing...", "yellow")
                    time.sleep(0.05)

                rec_path = os.path.join(TMP_DIR, "rec.wav")
                rec_trim_path = os.path.join(TMP_DIR, "rec_trim.wav")
                record_audio(rec_path, duration=RECORD_DURATION, fs=SAMPLE_RATE)
                trim_silence(rec_path, rec_trim_path, threshold=500, chunk_ms=30, fs=SAMPLE_RATE)

                seg_texts = transcribe_file(model, rec_trim_path, beam_size=BEAM_SIZE, language=TRANSCRIBE_LANGUAGE, vad_filter=True, condition_on_previous_text=False)
                text = " ".join(seg_texts).strip()

                if text:
                    clear_and_print(f">>> {text}")
                    translations = [translate_text(t) for t in seg_texts]
                    eng_text = " ".join([t for t in translations if t])
                    if eng_text:
                        clear_and_print(f"<<< {eng_text}")
                        # set the play_event BEFORE starting TTS thread to avoid race
                        play_event.set()
                        t = threading.Thread(target=lambda: asyncio.run(tts_and_play(eng_text, play_event)), daemon=True)
                        t.start()
                        try:
                            watchdog = threading.Timer(PLAYBACK_TIMEOUT, lambda: play_event.clear())
                            watchdog.daemon = True
                            watchdog.start()
                        except Exception:
                            pass

            except Exception as e:
                clear_and_print(f"Error: {e}")


if __name__ == "__main__":
    main_process()
