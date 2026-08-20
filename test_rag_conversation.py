"""
TEST — conversatie + RAG + robot (fara YOLO)
Run: python test_rag_conversation.py
Vorbeste natural — asistentul raspunde din manual si trimite comenzi robotului.
Spune 'stop' sau 'multumesc' pentru a iesi.
"""
import re
import time
import threading

from utils_env import make_openai_client
from speach_to_text import pick_working_input_device, record_wav, stt_whisper
from text_to_speach import start_tts, stop_tts, speak, is_tts_done
from agent_agent import agent_reply, tool_what_do_you_see, CAMERA_VIEW_STEPS
from agent_state import (
    set_current_assembly_step, set_post_step_context, get_scene,
    pop_next_sequence_step, has_pending_sequence,
)
from robot_bridge import RobotStatusListener, send_robot_step
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

    # --- Robot status callbacks ---
    def on_robot_done(data: dict):
        step_id = int(data.get("step_id", -1))
        print(f"[ROBOT] Pas finalizat: {step_id}")
        set_current_assembly_step(step_id)

        # --- Sequence continuation ---
        next_step = pop_next_sequence_step()
        if next_step is not None:
            speak(f"Pasul {step_id} finalizat. Execut automat pasul {next_step}...")
            send_robot_step(next_step)
            return

        # --- Camera-view steps: auto-capture and describe ---
        if step_id in CAMERA_VIEW_STEPS:
            def _do_camera_capture():
                speak("Robotul este la poziție. Activez camera...")
                description = tool_what_do_you_see()
                speak(description)
            threading.Thread(target=_do_camera_capture, daemon=True).start()
            return

        # --- Normal assembly step ---
        set_post_step_context({"step_id": step_id, "question_pending": True})
        scene = get_scene()
        scene_info = (
            f"Detectate: {', '.join(d['name'] for d in scene[:4])}. "
            if scene else "Nimic detectat in campul vizual. "
        )
        speak(
            f"Pasul {step_id} finalizat. {scene_info}"
            "Spune: detalii, verificari, probleme, corectii sau continua."
        )

    def on_robot_error(data: dict):
        step_id = int(data.get("step_id", -1))
        error_text = data.get("error", "A aparut o eroare.")
        print(f"[ROBOT] EROARE la pasul {step_id}: {error_text}")
        set_current_assembly_step(step_id)
        set_post_step_context({"step_id": step_id, "question_pending": True, "error": error_text})
        speak(f"A aparut o eroare la pasul {step_id}. Spune: verificari sau corectii.")

    status_listener = RobotStatusListener(on_done=on_robot_done, on_error=on_robot_error)
    status_listener.start()

    speak(
        "Test pornit. Pot raspunde din manual, pot controla robotul "
        "si pot descrie ce vede camera la cerere."
    )

    try:
        while not stop_event.is_set():
            while not is_tts_done() and not stop_event.is_set():
                time.sleep(0.1)

            if stop_event.is_set():
                break

            mean_amp = record_wav(input_device, "ask.wav", seconds=RECORD_SECONDS, fs=FS)

            if mean_amp < MIN_AMPLITUDE:
                print("[TEST] Liniste detectata, se reia ascultarea.")
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

            print("[AGENT] Procesez raspuns...", flush=True)
            answer = agent_reply(client, user_text)
            print("Raspuns:", answer)
            speak(answer)

    finally:
        status_listener.stop()
        stop_event.set()
        stop_tts()
        print("Test terminat.")


if __name__ == "__main__":
    main()
