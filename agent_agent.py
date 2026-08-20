import re
import json
from openai import OpenAI

import app_config

from agent_state import (
    get_scene,
    get_focus,
    set_scene,
    set_focus,
    set_focus_to_closest,
    try_match_class_in_text,
    set_pending_robot_command,
    get_pending_robot_command,
    clear_pending_robot_command,
    set_current_assembly_step,
    get_current_assembly_step,
    get_post_step_context,
    clear_post_step_context,
    set_pending_sequence,
    clear_pending_sequence,
)

from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
from robot_bridge import send_robot_step

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

_rag_db = None


def _normalize_romanian(text: str) -> str:
    """Convert Whisper cedilla variants ş/ţ to correct comma-below ș/ț."""
    return (
        text
        .replace('ş', 'ș').replace('Ş', 'Ș')
        .replace('ţ', 'ț').replace('Ţ', 'Ț')
    )


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------

def _get_rag_db():
    global _rag_db
    if _rag_db is None:
        _rag_db = Chroma(
            persist_directory=app_config.CHROMA_PATH,
            embedding_function=get_embedding_function()
        )
    return _rag_db


def rag_search(query_text: str):
    if not app_config.RAG_ENABLED:
        return ("", [], False)

    print("[RAG] Caut în ChromaDB...", flush=True)
    db = _get_rag_db()
    results = db.similarity_search_with_score(query_text, k=app_config.RAG_TOP_K)
    print(f"[RAG] Găsite {len(results)} rezultate brute.", flush=True)
    for doc, score in results:
        print(f"  score={score:.4f}  src={doc.metadata.get('id', '?')}  preview={doc.page_content[:60]!r}", flush=True)

    if not results:
        return ("", [], False)

    filtered = [(doc, score) for doc, score in results if score <= app_config.RAG_MAX_DISTANCE]
    print(f"[RAG] Filtrate: {len(filtered)} (threshold={app_config.RAG_MAX_DISTANCE})", flush=True)

    if not filtered:
        return ("", [], False)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in filtered])
    sources = [doc.metadata.get("id") for doc, _ in filtered]
    return (context_text, sources, True)


def _answer_from_rag(client: OpenAI, user_text: str, query: str) -> str | None:
    rag_ctx, rag_src, ok = rag_search(query)
    if not ok:
        return None

    print("[RAG] Trimit la OpenAI...", flush=True)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": app_config.SYSTEM_DIALOG},
            {
                "role": "system",
                "content": (
                    "Răspunzi DOAR din contextul RAG. "
                    "Dacă nu există răspuns în context, spui exact: "
                    "Nu găsesc informația în documentație."
                )
            },
            {"role": "system", "content": f"CONTEXT RAG:\n{rag_ctx}"},
            {"role": "user", "content": user_text},
        ],
    )
    answer = resp.choices[0].message.content.strip()
    print(f"[RAG] Surse: {rag_src}", flush=True)
    return answer




# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------

def _capture_scene() -> list[dict]:
    from vision_realsense import capture_once
    print("[VISION] Capturez imagine...", flush=True)
    dets = capture_once(show_window=True)
    set_scene(dets)
    if dets:
        set_focus(dets[0], "(capture)")
    else:
        set_focus(None, "(no detections)")
    return dets


def tool_what_do_you_see() -> str:
    dets = _capture_scene()
    if not dets:
        return "Nu văd nicio piesă clară acum."
    closest = dets[0]
    names = [d["name"] for d in dets[:6]]
    return (
        f"Văd {len(dets)} obiecte. Cel mai aproape este {closest['name']} "
        f"la aproximativ {closest['dist_m']:.2f} metri. "
        f"Primele detectate sunt: {', '.join(names)}."
    )


# ---------------------------------------------------------------------------
# Robot helpers
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _step_for_action(obj_name: str, action: str) -> int | None:
    n = _normalize_name(obj_name)
    current = get_current_assembly_step() or 0

    if n == "profil_a":
        return 2 if action == "pick" else 3
    if n == "profil_c":
        return 4 if action == "pick" else 5
    if n == "coltar":
        if action == "pick":
            return 8 if current >= 7 else 6
        else:
            return 9 if current >= 8 else 7
    if n == "set_imbus":
        return 10 if action == "pick" else 11
    return None


def _action_label(action: str) -> str:
    if action == "pick":
        return "ridicarea"
    if action == "place":
        return "așezarea"
    return "execuția"


def _resolve_object_for_robot(user_text: str, obj_name_hint: str | None):
    if obj_name_hint:
        return {"name": obj_name_hint, "dist_m": 0.0, "conf": 1.0}
    named = try_match_class_in_text(user_text)
    if named is not None:
        return named
    return get_focus() or set_focus_to_closest()


def _prepare_robot_command(user_text: str, action: str, obj_name_hint: str | None = None) -> str:
    scene = get_scene()

    # Trigger camera only if no scene at all and no hint from router
    if not scene and not obj_name_hint:
        _capture_scene()
        scene = get_scene()

    obj = _resolve_object_for_robot(user_text, obj_name_hint)
    if obj is None:
        return (
            "Nu am identificat piesa. "
            "Spune numele piesei (ex: «Profil_A») sau cere «ce vezi?» ca să scanez camera."
        )

    # For pick: check scene; warn if not found but don't block (scene may be from previous capture)
    from_scene = False
    if action == "pick" and scene:
        from_scene = any(d["name"].lower() == obj["name"].lower() for d in scene)
        if not from_scene and not obj_name_hint:
            return (
                f"Nu văd {obj['name']} în câmpul vizual al camerei. "
                "Spune «ce vezi?» pentru a rescana sau confirmă manual numele piesei."
            )

    step_id = _step_for_action(obj["name"], action)
    if step_id is None:
        return f"Nu am o comandă robot definită pentru piesa {obj['name']}."

    set_pending_robot_command({
        "step_id": step_id,
        "action": action,
        "object_name": obj["name"],
    })

    source = "detectată anterior de cameră" if (from_scene or get_focus()) else "specificată"
    if action == "pick":
        msg = f"Piesa {source}: {obj['name']}. Confirm ridicarea și trimiterea la robot?"
    else:
        msg = f"Piesa {source}: {obj['name']}. Confirm așezarea și trimiterea la robot?"

    return f"{msg} Spune «da» pentru execuție sau «nu» pentru anulare."


def _prepare_direct_step_command(step_id: int) -> str:
    if not (1 <= step_id <= 13):
        return "Pasul cerut nu este valid. Pașii disponibili sunt 1 până la 13."

    set_pending_robot_command({
        "step_id": step_id,
        "action": "direct_step",
        "object_name": f"Pasul_{step_id}",
    })
    return (
        f"Am pregătit execuția pasului {step_id}. "
        "Spune «da» pentru execuție sau «nu» pentru anulare."
    )


def _resolve_combo(obj: str | None) -> tuple | None:
    """Return (steps_list, label) for a pick+drop combo, or None if unknown."""
    if not obj:
        return None
    n = obj.lower().strip().replace(" ", "_").replace("-", "_")

    if "profil_a" in n or n in ("profila", "profil_a"):
        return [2, 3], "Profil_A"
    if "profil_c" in n or n in ("profilc", "profil_c"):
        return [4, 5], "Profil_C"
    if "coltar_2" in n or "coltar2" in n:
        return [8, 9], "Coltar 2"
    if "coltar_1" in n or "coltar1" in n:
        return [6, 7], "Coltar 1"
    if "coltar" in n:
        current = get_current_assembly_step() or 0
        return ([8, 9], "Coltar 2") if current >= 7 else ([6, 7], "Coltar 1")
    if "set_imbus" in n or "imbus" in n:
        return [10, 11], "Set_imbus"
    return None


def _prepare_combo_command(obj: str | None, user_text: str) -> str:
    result = _resolve_combo(obj) or _resolve_combo(
        next((w for w in user_text.lower().split()
              if any(k in w for k in ("profil", "coltar", "imbus"))), None)
    )
    if result is None:
        return (
            "Nu am recunoscut piesa pentru secvență. "
            "Spune de ex. «montează Profil_A» sau «execută secvența Coltar»."
        )

    steps, label = result
    set_pending_robot_command({
        "step_id":         steps[0],
        "action":          "combo",
        "object_name":     label,
        "remaining_steps": steps[1:],
    })
    steps_str = " → ".join(str(s) for s in steps)
    return (
        f"Am pregătit secvența pick+drop pentru {label}: pași {steps_str}. "
        "Spune «da» pentru execuție sau «nu» pentru anulare."
    )


def send_camera_view_step(step_id: int) -> str:
    """Send step 12 or 13 directly (no da/nu confirm) — robot moves to view position,
    camera activates automatically when DONE is received by dialog.py."""
    zone = "piese" if step_id == 12 else "asamblare"
    ok = send_robot_step(step_id)
    if ok:
        return (
            f"Mă deplasez la zona de {zone}. "
            "Activez camera automat când robotul ajunge la poziție."
        )
    return f"Nu am putut trimite comanda de deplasare la zona de {zone}."


# ---------------------------------------------------------------------------
# RAG query building
# ---------------------------------------------------------------------------

_STEP_COMPONENTS = {
    1:  "pregătire materiale unelte echipament liste verificare componente necesare",
    2:  "Profil_A profil A montare fixare introducere canal",
    3:  "Profil_A profil A pozitionare asezare orientare",
    4:  "Profil_C profil C montare fixare prindere",
    5:  "Profil_C profil C pozitionare asezare",
    6:  "Coltar colt coltar fixare prindere insurubare",
    7:  "Coltar colt coltar asezare pozitionare montaj final",
    8:  "clema prindere clema_prindere fixare strangere",
    9:  "Surub surub insurubare strangere fixare",
    10: "Set_imbus cheie imbus insurubare strangere suruburi",
    11: "Set_imbus cheie imbus strangere finala verificare",
    12: "control piese zona piese scanare camera verificare componente",
    13: "control ansamblu zona asamblare scanare camera verificare montaj final",
}

# Steps that move the robot to a camera-view position and trigger auto-capture on DONE
CAMERA_VIEW_STEPS = {12, 13}

_DOMAIN_CONTEXT = (
    "manual asamblare montaj instructiuni procedura "
    "profil aluminiu coltar surub clema imbus structura cadru"
)


def _build_step_query(step: int, mode: str = "general") -> str:
    hints = _STEP_COMPONENTS.get(step, "")
    base = f"Pasul {step} asamblare montaj {hints} {_DOMAIN_CONTEXT}"

    if mode == "details":
        return f"{base} detalii descriere etapa explicatii cum se face"
    if mode == "verifications":
        return f"{base} verificare control calitate cum stiu ca e corect rezultat asteptat"
    if mode == "errors":
        return f"{base} probleme greseli erori posibile ce poate merge prost atentionari"
    if mode == "corrections":
        return f"{base} corectii remedieri cum repar greseli solutii"
    return f"{base} instructiuni procedura descriere"


def _enrich_query(user_text: str) -> str:
    parts = [user_text, _DOMAIN_CONTEXT]

    step = get_current_assembly_step()
    if step is not None:
        parts.append(f"pasul {step} asamblare montaj")
        parts.append(_STEP_COMPONENTS.get(step, ""))

    scene = get_scene()
    if scene:
        parts.append(f"componente vizibile: {' '.join(d['name'] for d in scene[:4])}")

    focus = get_focus()
    if focus:
        parts.append(f"piesa curenta: {focus['name']}")

    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# LLM Intent Router
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM = """You are an intent classifier for a Romanian-language voice assistant that controls an assembly robot.

The user message may be preceded by context lines in square brackets — use them to resolve object references.

Classify the user's message into exactly one intent:
- robot_init     : initialize / home / start the robot (step 1)
- robot_step     : execute a specific numbered step (extract step number 1-11)
- robot_combo    : execute a full pick+drop sequence for one object (montează, preia și pune, secvența, pick și drop)
- robot_pick     : robot should pick / grab / lift an object (named OR referenced by "asta", "o", "ea", "piesa")
- robot_place    : robot should place / drop / put an object (named OR referenced by "asta", "o", "ea", "acolo")
- confirm_yes    : user confirms a pending action (da, ok, execută, bine, sigur, confirma)
- confirm_no     : user cancels a pending action (nu, anulează, stop, renunță)
- camera_query         : generic camera query — what does the camera see right now (ce vezi?)
- camera_query_parts   : view the parts/components zone — robot moves to step 12 (ce vezi in zona de piese, arata-mi piesele, verifica piesele)
- camera_query_assembly: view the assembly zone — robot moves to step 13 (ce vezi in zona de asamblare, arata-mi ansamblul, verifica ansamblul)
- post_details   : ask for details about the just-finished robot step (detalii, explică)
- post_verifications : ask what to check after the step (verificări, cum verific)
- post_errors    : ask about possible problems (probleme, erori, ce poate merge prost)
- post_corrections   : ask how to fix errors (corecții, cum repar, cum rezolv)
- post_continue  : continue to next step (continuă, mai departe, gata, următor)
- rag_query      : general question about the assembly manual or a specific step
- identity       : user asks who the assistant is (cine ești)
- other          : none of the above

Return ONLY a JSON object, no explanation, no markdown:
{"intent": "<intent>", "step": <number or null>, "object": "<object name or null>"}

IMPORTANT: If context shows a detected/focused object and user refers to it vaguely ("asta", "o", "ea", "piesa"),
set "object" to that object's name in your response.

Object names used in this system: Profil_A, Profil_C, Coltar, Set_imbus

Examples (no scene context):
"Pornește robotul"                  → {"intent": "robot_init",  "step": null, "object": null}
"Inițializează robotul"             → {"intent": "robot_init",  "step": null, "object": null}
"Fă pasul trei de la robot"         → {"intent": "robot_step",  "step": 3,    "object": null}
"Execută pasul 5"                   → {"intent": "robot_step",  "step": 5,    "object": null}
"Du-te la pasul doi"                → {"intent": "robot_step",  "step": 2,    "object": null}
"Montează Profil_A"                 → {"intent": "robot_combo", "step": null, "object": "Profil_A"}
"Pick și drop Profil_C"             → {"intent": "robot_combo", "step": null, "object": "Profil_C"}
"Execută secvența pentru Coltar"    → {"intent": "robot_combo", "step": null, "object": "Coltar"}
"Preia și pune Set imbus"           → {"intent": "robot_combo", "step": null, "object": "Set_imbus"}
"Ridică Profil_A"                   → {"intent": "robot_pick",  "step": null, "object": "Profil_A"}
"Pune Coltar"                       → {"intent": "robot_place", "step": null, "object": "Coltar"}
"Da, execută"                       → {"intent": "confirm_yes", "step": null, "object": null}
"Nu, anulează"                      → {"intent": "confirm_no",  "step": null, "object": null}
"Ce vezi?"                          → {"intent": "camera_query",         "step": null, "object": null}
"Ce vezi în zona de piese?"         → {"intent": "camera_query_parts",    "step": null, "object": null}
"Arată-mi piesele"                  → {"intent": "camera_query_parts",    "step": null, "object": null}
"Verifică piesele"                  → {"intent": "camera_query_parts",    "step": null, "object": null}
"Cum arată componentele?"           → {"intent": "camera_query_parts",    "step": null, "object": null}
"Ce vezi în zona de asamblare?"     → {"intent": "camera_query_assembly", "step": null, "object": null}
"Arată-mi ansamblul"                → {"intent": "camera_query_assembly", "step": null, "object": null}
"Verifică ansamblul"                → {"intent": "camera_query_assembly", "step": null, "object": null}
"Cum arată montajul?"               → {"intent": "camera_query_assembly", "step": null, "object": null}
"Detalii"                           → {"intent": "post_details","step": null, "object": null}
"Ce trebuie să verific?"            → {"intent": "post_verifications", "step": null, "object": null}
"Continuă"                          → {"intent": "post_continue","step": null, "object": null}
"Ce este pasul 2?"                  → {"intent": "rag_query",   "step": 2,    "object": null}
"Cum funcționează colțarul?"        → {"intent": "rag_query",   "step": null, "object": null}

Examples WITH scene context:
[Piesa detectată: Profil_A, distanța 0.45 m]
"Ridică-o"                          → {"intent": "robot_pick",  "step": null, "object": "Profil_A"}

[Piesa detectată: Profil_A, distanța 0.45 m]
"Ia asta"                           → {"intent": "robot_pick",  "step": null, "object": "Profil_A"}

[Piesa detectată: Coltar, distanța 0.30 m]
"Pune-o acolo"                      → {"intent": "robot_place", "step": null, "object": "Coltar"}

[Piesa detectată: Set_imbus, distanța 0.60 m]
"Continuă cu manipularea"           → {"intent": "robot_pick",  "step": null, "object": "Set_imbus"}

[Piesa detectată: Profil_C, distanța 0.50 m]
"Fă asta cu robotul"                → {"intent": "robot_pick",  "step": null, "object": "Profil_C"}

[Piesa detectată: Profil_A, distanța 0.45 m]
"Lasă-o jos"                        → {"intent": "robot_place", "step": null, "object": "Profil_A"}"""


def _route_intent(client: OpenAI, user_text: str) -> dict:
    """Ask GPT-4o-mini to classify user intent. Fast call, temperature=0, max_tokens=60."""
    pending  = get_pending_robot_command()
    post_step = get_post_step_context()
    focus    = get_focus()
    scene    = get_scene()

    context_lines = []

    # Scene / focus context — lets LLM resolve vague references ("asta", "o", "ea")
    if focus:
        context_lines.append(
            f"[Piesa detectată: {focus['name']}, distanța {focus.get('dist_m', 0):.2f} m]"
        )
    elif scene:
        names = ", ".join(d["name"] for d in scene[:3])
        context_lines.append(f"[Piese detectate în cameră: {names}]")

    if pending:
        context_lines.append(
            f"[Comandă robot în așteptare: pasul {pending['step_id']} ({pending['object_name']})]"
        )
    if post_step:
        err = post_step.get("error")
        ctx_txt = f"pasul {post_step['step_id']}"
        if err:
            ctx_txt += f", eroare: {err}"
        context_lines.append(f"[Robotul tocmai a finalizat {ctx_txt}]")

    prompt = ("\n".join(context_lines) + "\n" + user_text).strip() if context_lines else user_text

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=60,
        )
        raw = resp.choices[0].message.content.strip()
        print(f"[ROUTER] {raw!r}", flush=True)

        try:
            return json.loads(raw)
        except Exception:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                return json.loads(m.group())
    except Exception as e:
        print(f"[ROUTER] Eroare: {e}", flush=True)

    return {"intent": "other", "step": None, "object": None}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_POST_MODE_MAP = {
    "post_details":       "details",
    "post_verifications": "verifications",
    "post_errors":        "errors",
    "post_corrections":   "corrections",
}


def agent_reply(client: OpenAI, user_text: str) -> str:
    user_text = _normalize_romanian(user_text)
    print(f"[AGENT] Procesez: {user_text!r}", flush=True)

    route = _route_intent(client, user_text)
    intent = route.get("intent", "other")
    step   = route.get("step")
    obj    = route.get("object")
    print(f"[AGENT] intent={intent!r}  step={step}  object={obj!r}", flush=True)

    if intent == "identity":
        return (
            "Sunt asistentul tău vocal. "
            "Detectez piese cu camera și te ajut la asamblare pe baza manualului."
        )

    if intent in ("confirm_yes", "confirm_no"):
        pending = get_pending_robot_command()
        if pending is None:
            return "Nu am o comandă în așteptare."
        if intent == "confirm_no":
            clear_pending_robot_command()
            clear_pending_sequence()
            return "Am anulat comanda pentru robot."
    
        step_id   = pending["step_id"]
        obj_name  = pending["object_name"]
        action    = pending["action"]
        remaining = pending.get("remaining_steps", [])
        if remaining:
            set_pending_sequence(remaining)

        ok = send_robot_step(step_id)
        clear_pending_robot_command()
        if ok:
            set_current_assembly_step(step_id)
            if action == "combo":
                total = 1 + len(remaining)
                return (
                    f"Am trimis pasul {step_id} ({obj_name}). "
                    f"Secvență de {total} pași pornită — "
                    "execut automat pașii următori la confirmarea robotului."
                )
            if action == "direct_step":
                return (
                    f"Am trimis comanda pentru execuția pasului {step_id}. "
                    "Aștept confirmarea finalizării."
                )
            return (
                f"Am trimis comanda pentru {_action_label(action)} piesei {obj_name}. "
                "Aștept confirmarea finalizării."
            )
        clear_pending_sequence()
        return "Nu am reușit să trimit comanda către robot. Verificați conexiunea MQTT."

    # --- Post-step follow-up ---
    if intent in _POST_MODE_MAP or intent == "post_continue":
        ctx = get_post_step_context()
        if ctx is not None:
            step_id    = ctx.get("step_id")
            error_text = ctx.get("error")

            if intent == "post_continue":
                clear_post_step_context()
                return "Bine, continuăm."

            mode    = _POST_MODE_MAP[intent]
            rag_ans = _answer_from_rag(client, user_text, query=_build_step_query(step_id, mode))
            clear_post_step_context()

            if error_text and intent in ("post_verifications", "post_corrections"):
                prefix = f"A apărut eroarea: {error_text}. "
                return prefix + (rag_ans or "Nu găsesc informația în documentație.")
            return rag_ans or "Nu găsesc informația în documentație."

        # No active post-step context — serve from current step if known
        if intent in _POST_MODE_MAP:
            current = get_current_assembly_step()
            if current is not None:
                mode    = _POST_MODE_MAP[intent]
                rag_ans = _answer_from_rag(client, user_text, query=_build_step_query(current, mode))
                return rag_ans or "Nu găsesc informația în documentație."

    # --- Robot commands ---
    if intent == "robot_init":
        print("[AGENT] -> robot_init → pasul 1", flush=True)
        return _prepare_direct_step_command(1)

    if intent == "robot_step":
        if step is None:
            return "Nu am înțeles numărul pasului. Spune de exemplu «execută pasul trei»."
        print(f"[AGENT] -> robot_step → pasul {step}", flush=True)
        return _prepare_direct_step_command(int(step))

    if intent == "camera_query":
        set_focus_to_closest()
        return tool_what_do_you_see()

    if intent == "camera_query_parts":
        print("[AGENT] -> camera_query_parts → pasul 12", flush=True)
        return send_camera_view_step(12)

    if intent == "camera_query_assembly":
        print("[AGENT] -> camera_query_assembly → pasul 13", flush=True)
        return send_camera_view_step(13)

    if intent == "robot_combo":
        print(f"[AGENT] -> robot_combo pentru {obj!r}", flush=True)
        return _prepare_combo_command(obj, user_text)

    if intent == "robot_pick":
        return _prepare_robot_command(user_text, action="pick", obj_name_hint=obj)

    if intent == "robot_place":
        return _prepare_robot_command(user_text, action="place", obj_name_hint=obj)

    # --- RAG query ---
    if intent == "rag_query":
        if step is not None:
            set_current_assembly_step(int(step))
            rag_ans = _answer_from_rag(client, user_text, query=_build_step_query(int(step), "general"))
        else:
            rag_ans = _answer_from_rag(client, user_text, query=_enrich_query(user_text))
        return rag_ans or "Nu găsesc informația în documentație."

    # --- Fallback: try RAG anyway ---
    rag_ans = _answer_from_rag(client, user_text, query=_enrich_query(user_text))
    if rag_ans is not None:
        return rag_ans

    return (
        "Pot răspunde doar pe baza manualului de asamblare. "
        "Încearcă o întrebare despre pași, piese sau verificări."
    )
