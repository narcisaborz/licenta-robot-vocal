"""
TEST ROBOT MQTT — trimite pasi direct de la tastatura, fara voce.
Run: python test_robot.py
Tasteaza un numar (1-13) si apasa Enter pentru a trimite comanda la robot.
Tasteaza 'exit' pentru a iesi.
"""
import time
import threading

from utils_env import make_openai_client
from robot_bridge import send_robot_step, RobotStatusListener
from agent_state import set_current_assembly_step, set_post_step_context

STEP_NAMES = {
    1:  "Initializare / Home position",
    2:  "Pick Profil A",
    3:  "Drop Profil A",
    4:  "Pick Profil C",
    5:  "Drop Profil C",
    6:  "Pick Coltar 1",
    7:  "Drop Coltar 1",
    8:  "Pick Coltar 2",
    9:  "Drop Coltar 2",
    10: "Pick Set imbus",
    11: "Drop Set imbus",
    12: "Control Piese (camera zona piese)",
    13: "Control Ansamblu (camera zona asamblare)",
}


def main():
    make_openai_client()  # incarca .env cu OPENAI_API_KEY

    done_event = threading.Event()

    def on_robot_done(data: dict):
        step_id = int(data.get("step_id", -1))
        print(f"\n[ROBOT] ✓ Pasul {step_id} FINALIZAT: {STEP_NAMES.get(step_id, '?')}")
        set_current_assembly_step(step_id)
        done_event.set()

    def on_robot_error(data: dict):
        step_id = int(data.get("step_id", -1))
        err = data.get("error", "eroare necunoscuta")
        print(f"\n[ROBOT] ✗ Pasul {step_id} EROARE: {err}")
        done_event.set()

    print("Conectare la broker MQTT...")
    status_listener = RobotStatusListener(on_done=on_robot_done, on_error=on_robot_error)
    status_listener.start()
    time.sleep(1)

    print("\n=== Test Robot MQTT ===")
    print("Pasi disponibili:")
    for sid, name in STEP_NAMES.items():
        print(f"  {sid:2d} — {name}")
    print("\nTasteaza numarul pasului + Enter. Tasteaza 'exit' pentru a iesi.\n")

    try:
        while True:
            try:
                raw = input("Pas: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if raw.lower() == "exit":
                break

            if not raw.isdigit():
                print("Introdu un numar intre 1 si 11.")
                continue

            step_id = int(raw)
            if not (1 <= step_id <= 13):
                print("Pas invalid. Foloseste 1-13.")
                continue

            print(f"[MQTT] Trimit pasul {step_id}: {STEP_NAMES[step_id]}...")
            done_event.clear()
            ok = send_robot_step(step_id)

            if ok:
                print(f"[MQTT] Trimis. Astept confirmarea robotului (max 60s)...")
                received = done_event.wait(timeout=60)
                if not received:
                    print("[MQTT] Timeout — niciun raspuns de la robot in 60s.")
            else:
                print("[MQTT] EROARE — comanda nu a putut fi trimisa catre broker.")

    finally:
        status_listener.stop()
        print("Test robot terminat.")


if __name__ == "__main__":
    main()
