import threading
import re

# ---------------- NORMALIZATION ----------------

ALIASES = {
    "profil_a": ["profil a", "profilul a", "profila"],
    "profil_c": ["profil c", "profilul c", "profilc"],
    "coltar": ["coltar", "colțar"],
    "set_imbus": ["set imbus", "setul imbus", "imbus", "cheie imbus"],
    "surub": ["surub", "șurub"],
    "clema_prindere": ["clema prindere", "clema", "clemă prindere"],
}


def _normalize_label(text: str) -> str:
    t = text.lower().strip()
    t = t.replace("_", " ").replace("-", " ")
    t = re.sub(r"[^\w\săâîșț]", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------------- SCENE ----------------
scene_lock = threading.Lock()
current_scene = []  # list[dict]: {"name":..., "dist_m":..., "conf":...}


def set_scene(dets: list[dict]):
    global current_scene
    with scene_lock:
        current_scene = list(dets)


def get_scene() -> list[dict]:
    with scene_lock:
        return list(current_scene)


# ---------------- FOCUS ----------------
focus_lock = threading.Lock()
focus_object = None  # dict: {"name":..., "dist_m":..., "conf":...}


def set_focus(obj, reason=""):
    global focus_object
    with focus_lock:
        focus_object = None if obj is None else dict(obj)

    if obj is None:
        print("[FOCUS] -> None", reason)
    else:
        print(f"[FOCUS] -> {obj['name']} ({obj['dist_m']:.2f} m) {reason}")


def get_focus():
    with focus_lock:
        return None if focus_object is None else dict(focus_object)


def set_focus_to_closest():
    dets = get_scene()
    if not dets:
        set_focus(None, "(no scene)")
        return None

    dets.sort(key=lambda d: d["dist_m"])
    set_focus(dets[0], "(closest)")
    return dets[0]


def try_match_class_in_text(user_text: str):
    t = _normalize_label(user_text)
    dets = get_scene()
    if not dets:
        return None

    for d in dets:
        key = d["name"].lower()
        variants = ALIASES.get(key, [_normalize_label(d["name"])])
        for v in variants:
            if v in t:
                set_focus(d, "(named)")
                return d

    for d in dets:
        name_norm = _normalize_label(d["name"])
        if name_norm in t:
            set_focus(d, "(named-fallback)")
            return d

    return None



# ---------------- PENDING ROBOT COMMAND ----------------
pending_lock = threading.Lock()
pending_robot_command = None


def set_pending_robot_command(cmd: dict | None):
    global pending_robot_command
    with pending_lock:
        pending_robot_command = None if cmd is None else dict(cmd)

    if pending_robot_command is None:
        print("[PENDING] -> None")
    else:
        print(f"[PENDING] -> {pending_robot_command}")


def get_pending_robot_command():
    with pending_lock:
        return None if pending_robot_command is None else dict(pending_robot_command)


def clear_pending_robot_command():
    set_pending_robot_command(None)


# ---------------- CURRENT ASSEMBLY STEP ----------------
assembly_step_lock = threading.Lock()
current_assembly_step = None


def set_current_assembly_step(step: int | None):
    global current_assembly_step
    with assembly_step_lock:
        current_assembly_step = step

    print(f"[ASSEMBLY_STEP] -> {current_assembly_step}")


def get_current_assembly_step():
    with assembly_step_lock:
        return current_assembly_step


def clear_current_assembly_step():
    set_current_assembly_step(None)


# ---------------- POST STEP FOLLOW-UP ----------------
post_step_lock = threading.Lock()
post_step_context = None


def set_post_step_context(ctx: dict | None):
    global post_step_context
    with post_step_lock:
        post_step_context = None if ctx is None else dict(ctx)

    print(f"[POST_STEP] -> {post_step_context}")


def get_post_step_context():
    with post_step_lock:
        return None if post_step_context is None else dict(post_step_context)


def clear_post_step_context():
    set_post_step_context(None)


# ---------------- STEP SEQUENCE QUEUE ----------------
# When a combo is running, remaining steps are queued here.
# on_robot_done pops and sends the next one automatically.
_seq_lock = threading.Lock()
_pending_sequence: list = []


def set_pending_sequence(steps: list):
    global _pending_sequence
    with _seq_lock:
        _pending_sequence = list(steps)
    print(f"[SEQUENCE] -> {_pending_sequence}")


def pop_next_sequence_step():
    global _pending_sequence
    with _seq_lock:
        if _pending_sequence:
            return _pending_sequence.pop(0)
        return None


def has_pending_sequence() -> bool:
    with _seq_lock:
        return bool(_pending_sequence)


def clear_pending_sequence():
    global _pending_sequence
    with _seq_lock:
        _pending_sequence = []
    print("[SEQUENCE] -> cleared")