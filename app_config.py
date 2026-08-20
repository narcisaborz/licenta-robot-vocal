from pathlib import Path

MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC_STEP = "xarm5/cmd/action"
MQTT_TOPIC_STATUS_DONE = "xarm5/status/action_done"
MQTT_TOPIC_STATUS_ERROR = "xarm5/status/action_error"



BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = str(BASE_DIR / "chroma")
DATA_PATH = str(BASE_DIR / "data")
RAG_TOP_K = 8
RAG_MAX_DISTANCE = 0.7
RAG_ENABLED = True

MODEL_PATH = "Project_YOLOv8 (1)/Project_YOLOv8/runs/detect/train10/train10/weights/best.pt"
CONF_TH = 0.5

FS = 16000
RECORD_SECONDS = 10
MIN_AMPLITUDE = 100  # below this = silence, skip Whisper

HISTORY_MAX = 12
AUTO_DESCRIBE_NEW_OBJECTS = False

CLASS_COLORS = {
    'Profil_C':        (54, 227, 54),
    'Profil_B':        (41, 229, 215),
    'Coltar':          (246, 152, 34),
    'Set_imbus':       (242, 40, 152),
    'Surub':           (180, 26, 234),
    'Profil_A':        (204, 200, 37),
    'clema_prindere':  (255, 0, 0)
}

SYSTEM_DIALOG = (
    "Ești un asistent vocal pentru un sistem om-robot în asamblare. "
    "Răspunzi în română, scurt și clar. "
    "Folosești contextul conversației. "
    "Când utilizatorul întreabă despre piese, te bazezi pe SCENE și FOCUS, nu inventezi."
)

