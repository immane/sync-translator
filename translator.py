import asyncio
import os
import edge_tts
import threading
import requests
import urllib.parse
import sounddevice as sd
import wave
import subprocess
import shutil
import time
from faster_whisper import WhisperModel
from rich.console import Console
from rich.text import Text
import traceback

try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except Exception:
    webrtcvad = None
    WEBRTCVAD_AVAILABLE = False

# --- Adjustable parameters (tweak to improve recognition) ---
RECORD_DURATION = 5  # recording duration (seconds)
SAMPLE_RATE = 16000   # sample rate (Hz)
BEAM_SIZE = 10         # beam size for decoding; larger is more accurate but slower
TRANSCRIBE_LANGUAGE = "zh"  # language hint for transcription, use codes like 'zh' or 'en'
# --- VAD (voice activity detection) settings ---
VAD_MODE = 1  # aggressiveness: 0-3 (3 more aggressive); 1 is a good balance for speech
VAD_FRAME_MS = 30  # frame size in ms for VAD (10,20,30 allowed)
VAD_END_SILENCE_FRAMES = 10  # number of consecutive non-speech frames to consider end of speech
VAD_MAX_WAIT_SEC = 5  # max seconds to wait for speech before giving up
# Playback timeout (s): safety watchdog to clear `play_event` if playback hangs
PLAYBACK_TIMEOUT = 60
ENERGY_VAD_THRESHOLD = 220  # fallback energy threshold when webrtcvad is unavailable

# --- Configuration ---
model_path = "./models/small"
# Default to a pleasant female voice; allow override with environment variable `TTS_VOICE`.
# Examples: en-US-JennyNeural, en-US-AmberNeural, zh-CN-XiaoxiaoNeural
TTS_VOICE = os.environ.get("TTS_VOICE", "en-US-JennyNeural")

print(f"Loading model...")
model = WhisperModel(model_path, device="cpu", compute_type="int8")

# Event that is set while playback is active; the main loop will wait before recording until this is cleared
play_event = threading.Event()

# Console for colored inline status
console = Console()

def print_status(recog_state: str, recog_color: str):
    """Print a single English recognition status on one line, e.g. 'Recognition: Recognizing'."""
    t = Text()
    t.append(recog_state, style=recog_color)
    console.print(t, end='\r')

def clear_status_line():
    """Move to next line so status line isn't overwritten by output."""
    #console.print()
    pass

def clear_and_print(msg):
    """Print a message on a fresh line below the status line."""
    clear_status_line()
    console.print(msg)

def translate_text(text):
    """Translate a single text string using Google Translate web endpoint.

    Uses automatic source-language detection by default (sl=auto).
    Returns translated string on success or empty string on failure.
    """
    try:
        if not text:
            return ""
        encoded_text = urllib.parse.quote(text)
        # use sl=auto to let Google detect source language automatically
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q={encoded_text}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            # the response is a nested list; extract the top translation
            return response.json()[0][0][0]
    except Exception:
        return ""
    return ""

async def tts_and_play(text):
    """Use ffplay for playback; this is the most reliable method."""
    temp_mp3 = f"tmp_{hash(text)}.mp3"
    try:
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        await communicate.save(temp_mp3)
        # Prefer ffplay (from FFmpeg). If unavailable, fall back to pygame playback; set `play_event` during playback.
        ffplay_path = shutil.which("ffplay")
        play_event.set()
        try:
            if ffplay_path:
                subprocess.run([ffplay_path, "-nodisp", "-autoexit", temp_mp3],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.STDOUT)
            else:
                # Fall back to pygame playback (blocking). If that fails, try the system open command (may be non-blocking).
                try:
                    import pygame
                    pygame.mixer.init()
                    pygame.mixer.music.load(temp_mp3)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    pygame.mixer.quit()
                except Exception as e_play:
                    # Last resort: use the system default open command (may not block). Most players will read the file immediately; short delay before deletion is usually safe.
                    if os.name == "nt":
                        subprocess.run(f'start "" "{temp_mp3}"', shell=True)
                        time.sleep(1)
                    else:
                        open_cmd = shutil.which("xdg-open") or shutil.which("open")
                        if open_cmd:
                            subprocess.run([open_cmd, temp_mp3])
                            time.sleep(1)
                        else:
                            raise FileNotFoundError("ffplay is not installed and no system open command (xdg-open/open) found; playback failed") from e_play
        finally:
            play_event.clear()
        
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)
    except Exception as e:
        play_event.clear()  # ensure play_event is always cleared even if TTS save fails
        clear_and_print(f"Playback failed: {e}")

def record_audio(filename, duration=5, fs=16000):
    """Record audio using WebRTC VAD to detect speech.

    Behavior:
    - Waits up to `VAD_MAX_WAIT_SEC` for speech to start.
    - Once speech is detected, records until `VAD_END_SILENCE_FRAMES` of silence
      or until `duration` seconds of audio have been captured.
    - If `play_event` becomes set during listening/recording, recording stops immediately.
    - If webrtcvad is unavailable, falls back to simple energy-based VAD.
    """
    try:
        print("Recording...", end="\r")
        vad = webrtcvad.Vad(VAD_MODE) if WEBRTCVAD_AVAILABLE else None
        frame_ms = VAD_FRAME_MS
        frame_samples = int(fs * frame_ms / 1000)
        max_wait_frames = int((VAD_MAX_WAIT_SEC * 1000) / frame_ms)

        frames = []
        started = False
        silence_frames = 0
        wait_frames = 0
        recorded_samples = 0

        with sd.InputStream(samplerate=fs, channels=1, dtype='int16', blocksize=frame_samples) as stream:
            while True:
                if play_event.is_set():
                    # Playback started; abort recording
                    break
                data, _ = stream.read(frame_samples)
                if data.size == 0:
                    continue
                if vad is not None:
                    raw_bytes = data.tobytes()
                    is_speech = False
                    try:
                        is_speech = vad.is_speech(raw_bytes, fs)
                    except Exception:
                        # In case VAD errors, fall back to treating as speech
                        is_speech = True
                else:
                    # Fallback VAD: simple frame energy gate
                    is_speech = float(abs(data).mean()) > ENERGY_VAD_THRESHOLD

                if not started:
                    if is_speech:
                        started = True
                        frames.append(data.copy())
                        recorded_samples += data.shape[0]
                        silence_frames = 0
                    else:
                        wait_frames += 1
                        if wait_frames >= max_wait_frames:
                            # no speech detected within wait window
                            break
                        continue
                else:
                    frames.append(data.copy())
                    recorded_samples += data.shape[0]
                    if is_speech:
                        silence_frames = 0
                    else:
                        silence_frames += 1
                    # stop on enough trailing silence or max duration
                    if silence_frames >= VAD_END_SILENCE_FRAMES:
                        break
                    if recorded_samples >= int(duration * fs):
                        break

        if frames:
            audio_bytes = b''.join([f.tobytes() for f in frames])
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(fs)
                wf.writeframes(audio_bytes)
        else:
            # write an empty short file to avoid downstream errors
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(fs)
                wf.writeframes(b"")
    except Exception as e:
        clear_and_print(f"Recording device error: {e}")


def trim_silence(input_wav, output_wav, threshold=500, chunk_ms=30, fs=SAMPLE_RATE):
    """Simple energy-based silence trimming to remove long silence at start/end and improve recognition."""
    try:
        import numpy as np
        with wave.open(input_wav, 'rb') as wf:
            nframes = wf.getnframes()
            frames = wf.readframes(nframes)

        audio = np.frombuffer(frames, dtype=np.int16)
        chunk_samples = max(1, int(fs * chunk_ms / 1000))
        energies = [np.abs(audio[i:i+chunk_samples]).mean() for i in range(0, len(audio), chunk_samples)]
        voiced = [i for i, e in enumerate(energies) if e > threshold]
        if not voiced:
            shutil.copy(input_wav, output_wav)
            return

        start_chunk = max(0, voiced[0] - 1)
        end_chunk = min(len(energies) - 1, voiced[-1] + 1)
        start_sample = start_chunk * chunk_samples
        end_sample = min(len(audio), (end_chunk + 1) * chunk_samples)
        trimmed = audio[start_sample:end_sample]

        with wave.open(output_wav, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(trimmed.tobytes())
    except Exception as e:
        # If trimming fails, fall back to the original file
        shutil.copy(input_wav, output_wav)
        clear_and_print(f"Silence trimming failed, using original audio: {e}")

def main_process():
    print("\n$ Sync Translator is started (FFmpeg mode)")
    while True:
        try:
            # If it is playing, wait until the playback ends before starting the next recording to 
            # avoid self-recording. The `play_event` is set by the playback thread and cleared 
            # when playback finishes or if a timeout occurs.
            while play_event.is_set():
                time.sleep(0.05)

            # show recognition starting
            record_audio("rec.wav", duration=RECORD_DURATION, fs=SAMPLE_RATE)
            # trim silence to reduce useless frames
            trim_silence("rec.wav", "rec_trim.wav", threshold=500, chunk_ms=30, fs=SAMPLE_RATE)

            # Start recognition
            segments, _ = model.transcribe(
                "rec_trim.wav",
                beam_size=BEAM_SIZE,
                language=TRANSCRIBE_LANGUAGE,
                vad_filter=True,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
            )
            # Translate each ASR segment separately to preserve sentence boundaries
            seg_texts = [s.text.strip() for s in segments if s.text.strip()]
            text = " ".join(seg_texts).strip()
            
            if text:
                clear_and_print(f">>> {text}")
                # show translation in progress
                translations = [translate_text(t) for t in seg_texts]
                eng_text = " ".join([t for t in translations if t])
                if eng_text:
                    clear_and_print(f"<<< {eng_text}")
                    # Ensure recording does not start until playback finishes:
                    # set the play_event before starting the playback thread to avoid a race.
                    play_event.set()
                    t = threading.Thread(target=lambda: asyncio.run(tts_and_play(eng_text)), daemon=True)
                    t.start()
                    # safety watchdog: clear play_event if playback hangs for too long
                    try:
                        watchdog = threading.Timer(PLAYBACK_TIMEOUT, lambda: play_event.clear())
                        watchdog.daemon = True
                        watchdog.start()
                    except Exception:
                        pass

        except Exception as e:
            tb = traceback.format_exc()
            clear_and_print(f"Error: {e}\n{tb}")

if __name__ == "__main__":
    main_process()