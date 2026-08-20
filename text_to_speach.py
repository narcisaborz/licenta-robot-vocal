import os
import time
import queue
import threading
import tempfile
from gtts import gTTS

tts_queue = queue.Queue()
_tts_thread = None
_tts_playing = False

def tts_to_mp3(text: str) -> str:
    tts = gTTS(text=text, lang='ro', slow=False)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    tts.save(tmp.name)
    return tmp.name

def is_tts_done() -> bool:
    return tts_queue.empty() and not _tts_playing


def _tts_worker(stop_event):
    global _tts_playing
    import pygame
    pygame.mixer.init()

    while not stop_event.is_set():
        try:
            text = tts_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if text is None:
            tts_queue.task_done()
            break

        path = None
        try:
            _tts_playing = True
            path = tts_to_mp3(text)
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() and not stop_event.is_set():
                time.sleep(0.05)
        except Exception as e:
            print("Eroare TTS:", e)
        finally:
            _tts_playing = False
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except:
                pass
            tts_queue.task_done()

def start_tts(stop_event):
    global _tts_thread
    _tts_thread = threading.Thread(target=_tts_worker, args=(stop_event,), daemon=False)
    _tts_thread.start()
    return _tts_thread

def speak(text: str):
    tts_queue.put(text.replace("_", " "))

def clear_tts_queue():
    try:
        while True:
            item = tts_queue.get_nowait()
            tts_queue.task_done()
    except queue.Empty:
        pass

def stop_tts():
    tts_queue.put(None)
    if _tts_thread:
        try:
            _tts_thread.join(timeout=2)
        except:
            pass
