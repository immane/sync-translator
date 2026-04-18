import os
import asyncio
import shutil
import subprocess
import time
from config import TMP_DIR, TTS_VOICE, PLAYBACK_TIMEOUT
import edge_tts


async def tts_and_play(text, play_event=None):
    temp_mp3 = os.path.join(TMP_DIR, f"tmp_{hash(text)}.mp3")
    try:
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        await communicate.save(temp_mp3)
        ffplay_path = shutil.which("ffplay")
        if ffplay_path:
            subprocess.run([ffplay_path, "-nodisp", "-autoexit", temp_mp3], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        else:
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(temp_mp3)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                pygame.mixer.quit()
            except Exception:
                if os.name == "nt":
                    subprocess.run(f'start "" "{temp_mp3}"', shell=True)
                    time.sleep(1)
                else:
                    open_cmd = shutil.which("xdg-open") or shutil.which("open")
                    if open_cmd:
                        subprocess.run([open_cmd, temp_mp3])
                        time.sleep(1)
    finally:
        try:
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
        except Exception:
            pass
        if play_event is not None:
            try:
                play_event.clear()
            except Exception:
                pass
