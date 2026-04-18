import os
import wave
import time
import numpy as np
import sounddevice as sd
from config import SAMPLE_RATE, RECORD_DURATION, VAD_FRAME_MS, VAD_MAX_WAIT_SEC, VAD_END_SILENCE_FRAMES, TMP_DIR, ENERGY_VAD_THRESHOLD

# Avoid circular import: try to import `print_status` from translator, but
# fall back to a local minimal implementation if translator isn't ready.
try:
    from translator import print_status
except Exception:
    from rich.console import Console
    from rich.text import Text
    _console = Console()
    def print_status(recog_state: str, recog_color: str):
        t = Text()
        t.append(recog_state, style=recog_color)
        _console.print(t, end='\r')
try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except Exception:
    webrtcvad = None
    WEBRTCVAD_AVAILABLE = False


def record_audio(filename, duration=RECORD_DURATION, fs=SAMPLE_RATE):
    try:
        print_status("Recording...", "green")
        vad = webrtcvad.Vad(1) if WEBRTCVAD_AVAILABLE else None
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
                data, _ = stream.read(frame_samples)
                if data.size == 0:
                    continue
                if vad is not None:
                    try:
                        is_speech = vad.is_speech(data.tobytes(), fs)
                    except Exception:
                        is_speech = True
                else:
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
                            break
                        continue
                else:
                    frames.append(data.copy())
                    recorded_samples += data.shape[0]
                    if is_speech:
                        silence_frames = 0
                    else:
                        silence_frames += 1
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
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(fs)
                wf.writeframes(b"")
    except Exception as e:
        print(f"Recording device error: {e}")


def record_fixed_chunk(filename, duration=RECORD_DURATION, fs=SAMPLE_RATE):
    try:
        print(f"Recording chunk {duration}s...", end='\r')
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(recording.tobytes())
    except Exception as e:
        print(f"Recording device error: {e}")


def trim_silence(input_wav, output_wav, threshold=500, chunk_ms=30, fs=SAMPLE_RATE):
    try:
        import shutil
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
        import shutil
        shutil.copy(input_wav, output_wav)
        print(f"Silence trimming failed: {e}")
