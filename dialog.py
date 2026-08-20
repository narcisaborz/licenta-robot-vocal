import re
import time
import threading
from openai import OpenAI

from app_config import RECORD_SECONDS, FS, MIN_AMPLITUDE
from speach_to_text import pick_working_input_device, record_wav, stt_whisper
from text_to_speach import speak, clear_tts_queue, is_tts_done
from agent_agent import agent_reply, tool_what_do_you_see, CAMERA_VIEW_STEPS
from agent_state import (
    set_post_step_context, set_current_assembly_step, get_scene,
    pop_next_sequence_step, has_pending_sequence,
)
from robot_bridge import RobotStatusListener, send_robot_step

_EXIT_PATTERNS = [
    r"\bmul[tț]umesc\b",
    r"\bthank you\b",
    r"\bstop\b",
    r"\bie[sș]i\b",
    r"\bopre[sș]te\b",
    r"\bla revedere\b",
]


def _wants_exit(text: str) -> bool:
    t = text.lower().strip()
    return any(re.search(p, t) for p in _EXIT_PATTERNS)


def dialog_loop(client: OpenAI, stop_event):
    input_device = pick_working_input_device(fs=FS)

    def on_robot_done(data: dict):
        step_id = int(data.get("step_id", -1))
        print(f"[ROBOT] Pas finalizat: {step_id}")

        set_current_assembly_step(step_id)
        next_step = pop_next_sequence_step()
        if next_step is not None:
            speak(f"Pasul {step_id} finalizat. Execut automat pasul {next_step}...")
            send_robot_step(next_step)
            return

        if step_id in CAMERA_VIEW_STEPS:
            def _do_camera_capture():
                speak("Robotul este la poziție. Activez camera...")
                description = tool_what_do_you_see()
                speak(description)
            threading.Thread(target=_do_camera_capture, daemon=True).start()
            return

        set_post_step_context({
            "step_id": step_id,
            "question_pending": True
        })

        scene = get_scene()
        if scene:
            names = ", ".join(d["name"] for d in scene[:4])
            scene_info = f"Detectate în câmpul vizual: {names}. "
        else:
            scene_info = "Nimic detectat în câmpul vizual. "
        speak(
            f"Pasul {step_id} finalizat. {scene_info}"
            "Spune: detalii, verificări, probleme, corecții sau continuă."
        )
        
        

    def on_robot_error(data: dict):
        step_id = int(data.get("step_id", -1))
        error_text = data.get("error", "A apărut o eroare la execuția robotului.")

        print(f"[ROBOT] EROARE la pasul {step_id}: {error_text}")

        set_current_assembly_step(step_id)
        set_post_step_context({
            "step_id": step_id,
            "question_pending": True,
            "error": error_text
        })

        speak(
            f"A apărut o eroare la pasul {step_id}. "
            "Spune: verificări sau corecții."
        )

    status_listener = RobotStatusListener(
        on_done=on_robot_done,
        on_error=on_robot_error
    )
    status_listener.start()

    speak(
        "Salut! Sunt asistentul tău vocal pentru procesul de asamblare. "
        "Spune-mi cum te pot ajuta."
    )

    try:
        while not stop_event.is_set():
            # Wait for TTS to finish before recording
            while not is_tts_done() and not stop_event.is_set():
                time.sleep(0.1)

            if stop_event.is_set():
                break

            try:
                mean_amp = record_wav(input_device, "ask.wav", seconds=RECORD_SECONDS, fs=FS)

                if mean_amp < MIN_AMPLITUDE:
                    print("[DIALOG] Liniște detectată, se reia ascultarea.")
                    continue

                user_text = stt_whisper(client, "ask.wav")

                print("Ai spus:", user_text)

                if not user_text:
                    continue

                if _wants_exit(user_text):
                    speak("La revedere! O zi bună!")
                    while not is_tts_done():
                        time.sleep(0.1)
                    stop_event.set()
                    break

                answer = agent_reply(client, user_text)
                print("Răspuns:", answer)
                speak(answer)

            except KeyboardInterrupt:
                stop_event.set()
                break
            except Exception as e:
                print("Eroare:", e)

    finally:
        status_listener.stop()
