"""
STEP 1 TEST — conversation only (STT + TTS, no RAG, no YOLO)
Run: python test_conversation.py
Say anything — the assistant echoes it back.
Say 'stop' or 'multumesc' to exit.
"""
import re
import time
import threading

from utils_env import make_openai_client
from speach_to_text import pick_working_input_device, record_wav, stt_whisper
from text_to_speach import start_tts, stop_tts, speak, is_tts_done
from app_config import FS, RECORD_SECONDS, MIN_AMPLITUDE

_EXIT_PATTERNS = [
    r"\bmul[tț]umesc\b",
    r"\bthank you\b",
    r"\bstop\b",
    r"\bla revedere\b",
]


def _wants_exit(text: str) -> bool:
    t = text.lower().strip()
    return any(re.search(p, t) for p in _EXIT_PATTERNS)


def main():
    stop_event = threading.Event()
    client = make_openai_client()
    start_tts(stop_event)

    input_device = pick_working_input_device(fs=FS)

    speak("Test conversație pornit. Spune ceva.")

    try:
        while not stop_event.is_set():
            while not is_tts_done() and not stop_event.is_set():
                time.sleep(0.1)

            if stop_event.is_set():
                break

            mean_amp = record_wav(input_device, "ask.wav", seconds=RECORD_SECONDS, fs=FS)

            if mean_amp < MIN_AMPLITUDE:
                print("[TEST] Liniște, se reia...")
                continue

            user_text = stt_whisper(client, "ask.wav")
            print("Ai spus:", user_text)

            if not user_text:
                continue

            if _wants_exit(user_text):
                speak("La revedere!")
                while not is_tts_done():
                    time.sleep(0.1)
                break

            # Echo back — no RAG, no agent
            speak(f"Ai spus: {user_text}")

    finally:
        stop_event.set()
        stop_tts()
        print("Test conversație terminat.")


if __name__ == "__main__":
    main()
