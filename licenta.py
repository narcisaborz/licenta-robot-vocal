import threading
from utils_env import make_openai_client
from text_to_speach import start_tts, stop_tts
from dialog import dialog_loop


def main():
    stop_event = threading.Event()
    client = make_openai_client()

    start_tts(stop_event)

    th_dialog = threading.Thread(
        target=dialog_loop,
        args=(client, stop_event),
        daemon=False
    )

    th_dialog.start()

    try:
        th_dialog.join()
    finally:
        stop_event.set()
        stop_tts()


if __name__ == "__main__":
    main()
    