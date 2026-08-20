#!/usr/bin/env python3
"""
gui_interface.py  –  Graphical frontend for the assembly assistant.
Run from the Licenta folder:  python gui_interface.py

Panels:
  Left   – live RealSense camera feed with YOLOv8 bounding boxes
  Right  – relevant page from the assembly PDF manual
           (updates automatically when a step is mentioned by the assistant)

Buttons:
  START  – connects to OpenAI, starts TTS + dialog (continuous microphone)
  STOP   – pauses the dialog session (camera feed stays on)
  CLOSE  – stops everything and exits
"""
from __future__ import annotations

import sys
import re
import threading
import queue
from pathlib import Path

import tkinter as tk

# ── ensure the Licenta folder is on sys.path ──────────────────────────────────
_HERE = Path(__file__).resolve().parent

if not (_HERE / "app_config.py").exists():
    _found = False
    # 1. Search upward (e.g. script is inside a sub-folder of the project)
    for _p in _HERE.parents:
        _c = _p / "Licenta"
        if (_c / "app_config.py").exists():
            _HERE = _c
            _found = True
            break
    # 2. Search downward up to 4 levels (e.g. script sits above the project)
    if not _found:
        for _depth in range(1, 5):
            _matches = list(_HERE.glob("/".join(["*"] * _depth) + "/app_config.py"))
            _licenta = [m.parent for m in _matches if m.parent.name == "Licenta"]
            if _licenta:
                _HERE = _licenta[0]
                _found = True
                break
            elif _matches:
                _HERE = _matches[0].parent
                _found = True
                break
    if not _found:
        print(
            f"[ERROR] Cannot find app_config.py near {_HERE}.\n"
            "        Place gui_interface.py inside the Licenta\\ folder, or run it from there."
        )

sys.path.insert(0, str(_HERE))

# Force UTF-8 output so Romanian characters don't crash on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── project imports ───────────────────────────────────────────────────────────
import app_config
import text_to_speach as _tts_mod
from text_to_speach import start_tts, stop_tts, clear_tts_queue
from agent_state import set_scene, set_focus, get_current_assembly_step
from utils_env import make_openai_client
from dialog import dialog_loop

# ── optional dependencies ─────────────────────────────────────────────────────
try:
    from PIL import Image, ImageTk
except ImportError:
    print("Pillow lipseste. Instaleaza: pip install Pillow")
    sys.exit(1)

try:
    import cv2
    import numpy as np
    import pyrealsense2 as rs
    from ultralytics import YOLO
    _HAS_CAM = True
except ImportError as _e:
    _HAS_CAM = False
    print(f"[CAM] Librarii lipsesc ({_e}) - feedul video va fi dezactivat.")

try:
    import fitz          # pip install pymupdf
    _HAS_PDF = True
except ImportError:
    _HAS_PDF = False
    print("[PDF] PyMuPDF lipseste - manualul nu va fi afisat.  pip install pymupdf")

# ── layout ────────────────────────────────────────────────────────────────────
CAM_W, CAM_H = 640, 480
MANUAL_W     = 430
POLL_MS      = 40          # ~25 fps GUI refresh


# ═══════════════════════════════════════════════════════════════════════════════
#  PDF manual index
# ═══════════════════════════════════════════════════════════════════════════════

class ManualIndex:
    def __init__(self):
        self.doc: "fitz.Document | None" = None
        self.step_pages: dict[int, list[int]] = {}

        if not _HAS_PDF:
            return

        pdf = Path(app_config.DATA_PATH) / "Manual pentru pasii de asamblare.pdf"
        if not pdf.exists():
            print(f"[PDF] Nu gasesc manualul la: {pdf}")
            return

        try:
            self.doc = fitz.open(str(pdf))
            self._index()
        except Exception as e:
            print(f"[PDF] Eroare la deschidere: {e}")
            self.doc = None

    def _index(self):
        pat = re.compile(r"\bpasul?\s+(\d+)\b", re.IGNORECASE)
        for page_no, page in enumerate(self.doc):
            for m in pat.finditer(page.get_text()):
                step = int(m.group(1))
                lst  = self.step_pages.setdefault(step, [])
                if page_no not in lst:
                    lst.append(page_no)
        print(f"[PDF] Index pasi: { {k: v for k, v in sorted(self.step_pages.items())} }")

    def _render(self, page_idx: int, target_w: int = MANUAL_W) -> "Image.Image | None":
        if self.doc is None:
            return None
        try:
            pg    = self.doc[page_idx]
            scale = target_w / pg.rect.width
            pix   = pg.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        except Exception as e:
            print(f"[PDF] Render pagina {page_idx}: {e}")
            return None

    def images_for_step(self, step: int) -> list[Image.Image]:
        imgs = []
        for p in self.step_pages.get(step, [])[:3]:
            img = self._render(p)
            if img:
                imgs.append(img)
        return imgs


# ═══════════════════════════════════════════════════════════════════════════════
#  Camera worker (runs in its own daemon thread)
# ═══════════════════════════════════════════════════════════════════════════════

class CameraWorker(threading.Thread):
    def __init__(self, out_q: queue.Queue, stop_ev: threading.Event):
        super().__init__(daemon=True, name="cam-worker")
        self.out_q   = out_q
        self.stop_ev = stop_ev

    def _push(self, img: Image.Image):
        if self.out_q.full():
            try:   self.out_q.get_nowait()
            except queue.Empty: pass
        try:   self.out_q.put_nowait(img)
        except queue.Full: pass

    def run(self):
        if not _HAS_CAM:
            return

        try:
            model = YOLO(app_config.MODEL_PATH)
            print(f"[CAM] Model YOLO incarcat: {app_config.MODEL_PATH}", flush=True)
        except Exception as e:
            print(f"[CAM] Eroare la incarcare model: {e}")
            return

        pipeline = None
        try:
            if not rs.context().devices:
                print("[CAM] Nicio camera RealSense - feed dezactivat.")
                return

            pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, CAM_W, CAM_H, rs.format.bgr8, 30)
            cfg.enable_stream(rs.stream.depth, CAM_W, CAM_H, rs.format.z16, 30)
            pipeline.start(cfg)

            sf = rs.spatial_filter()
            tf = rs.temporal_filter()
            print("[CAM] Camera pornita.", flush=True)

            while not self.stop_ev.is_set():
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                except RuntimeError:
                    continue

                c_frame = frames.get_color_frame()
                d_frame = frames.get_depth_frame()
                if not c_frame or not d_frame:
                    continue

                color  = np.asanyarray(c_frame.get_data())
                d_frame = sf.process(d_frame)
                d_frame = tf.process(d_frame)
                depth  = np.asanyarray(d_frame.get_data()) * 0.001

                h, w = color.shape[:2]
                dets = []

                for r in model(color, verbose=False):
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0])
                        if conf < app_config.CONF_TH:
                            continue
                        cls  = int(box.cls[0])
                        name = model.names[cls]

                        x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w, x2))
                        y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h, y2))

                        roi = depth[y1:y2, x1:x2]
                        rv  = roi[(roi > 0) & (~np.isnan(roi))]
                        if rv.size == 0:
                            continue
                        dist = float(np.median(rv))

                        dets.append({"name": name, "dist_m": dist, "conf": conf})

                        col = app_config.CLASS_COLORS.get(name, (0, 180, 255))
                        cv2.rectangle(color, (x1, y1), (x2, y2), col, 2)
                        cv2.putText(
                            color,
                            f"{name}  {dist:.2f} m  {conf:.0%}",
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2,
                        )

                dets.sort(key=lambda d: d["dist_m"])
                set_scene(dets)
                if dets:
                    set_focus(dets[0], "(gui)")

                self._push(Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB)))

        except Exception as e:
            print(f"[CAM] Eroare: {e}")
        finally:
            if pipeline:
                try: pipeline.stop()
                except: pass
            print("[CAM] Camera oprita.")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main GUI
# ═══════════════════════════════════════════════════════════════════════════════

class AssemblyGUI:

    BG     = "#1a1a2e"
    PANEL  = "#16213e"
    ACCENT = "#0f3460"
    FG     = "#ecf0f1"
    GREEN  = "#27ae60"
    RED    = "#e74c3c"
    GREY   = "#566573"
    ORANGE = "#f39c12"

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Asistent Asamblare - YOLOv8")
        self.root.configure(bg=self.BG)
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._cam_stop               = threading.Event()
        self._dlg_stop: threading.Event | None  = None
        self._tts_stop: threading.Event | None  = None
        self._cam_worker: CameraWorker | None   = None
        self._dlg_thread: threading.Thread | None = None
        self._running                = False

        self._frame_q: queue.Queue  = queue.Queue(maxsize=2)

        self._cam_photo: ImageTk.PhotoImage | None       = None
        self._manual_imgs: list[ImageTk.PhotoImage]      = []
        self._manual_idx                                 = 0

        self._client     = None
        self._last_step  = None
        self.manual      = ManualIndex()

        self._build_ui()
        self._install_tts_hook()
        self._start_camera()
        self._poll()

    # ══════════════════════════════════════════════════════════════════════
    #  UI construction
    # ══════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        r = self.root

        tk.Label(r,
                 text="Asistent Vocal Asamblare  •  YOLOv8",
                 font=("Helvetica", 15, "bold"),
                 bg=self.BG, fg=self.ACCENT,
                 ).pack(pady=(10, 4))

        row = tk.Frame(r, bg=self.BG)
        row.pack(fill=tk.BOTH, expand=True, padx=10)

        # ── camera panel (left) ───────────────────────────────────────────
        clf = tk.LabelFrame(
            row, text="  Camera / Detectie YOLO  ",
            font=("Helvetica", 10, "bold"),
            bg=self.PANEL, fg=self.FG, bd=2, relief=tk.GROOVE,
        )
        clf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        self._cam_canvas = tk.Canvas(
            clf, width=CAM_W, height=CAM_H, bg="#000", highlightthickness=0,
        )
        self._cam_canvas.pack(padx=4, pady=4)
        self._cam_canvas.create_text(
            CAM_W // 2, CAM_H // 2,
            text="Initializare camera...",
            fill=self.ORANGE, font=("Helvetica", 14),
        )

        self._cam_status = tk.Label(
            clf, text="Asteptare...",
            bg=self.PANEL, fg=self.ORANGE, font=("Helvetica", 9),
        )
        self._cam_status.pack(pady=(0, 4))

        # ── manual panel (right) ──────────────────────────────────────────
        mf = tk.LabelFrame(
            row, text="  Manual - Sectiunea curenta  ",
            font=("Helvetica", 10, "bold"),
            bg=self.PANEL, fg=self.FG, bd=2, relief=tk.GROOVE,
        )
        mf.pack(side=tk.LEFT, fill=tk.BOTH, padx=(6, 0))

        nav = tk.Frame(mf, bg=self.PANEL)
        nav.pack(fill=tk.X, padx=4, pady=(4, 0))

        self._prev_btn = tk.Button(
            nav, text="<  Inapoi", command=self._man_prev,
            bg=self.ACCENT, fg=self.FG, relief=tk.FLAT, padx=6,
            state=tk.DISABLED,
        )
        self._prev_btn.pack(side=tk.LEFT)

        self._pg_lbl = tk.Label(nav, text="", bg=self.PANEL, fg=self.FG,
                                font=("Helvetica", 9))
        self._pg_lbl.pack(side=tk.LEFT, expand=True)

        self._next_btn = tk.Button(
            nav, text="Inainte  >", command=self._man_next,
            bg=self.ACCENT, fg=self.FG, relief=tk.FLAT, padx=6,
            state=tk.DISABLED,
        )
        self._next_btn.pack(side=tk.RIGHT)

        # manual step selector row
        sel = tk.Frame(mf, bg=self.PANEL)
        sel.pack(fill=tk.X, padx=4, pady=(2, 0))

        tk.Label(sel, text="Pas:", bg=self.PANEL, fg=self.FG,
                 font=("Helvetica", 9)).pack(side=tk.LEFT)

        self._step_var = tk.StringVar()
        step_entry = tk.Entry(sel, textvariable=self._step_var, width=4,
                              bg=self.ACCENT, fg=self.FG, insertbackground=self.FG,
                              relief=tk.FLAT, font=("Helvetica", 10))
        step_entry.pack(side=tk.LEFT, padx=(2, 4))
        step_entry.bind("<Return>", lambda _e: self._on_show_step())

        tk.Button(sel, text="Afiseaza", command=self._on_show_step,
                  bg=self.ORANGE, fg="white", relief=tk.FLAT, padx=6,
                  ).pack(side=tk.LEFT)

        indexed = sorted(self.manual.step_pages.keys()) if self.manual.step_pages else []
        idx_text = (f"Pasi indexati: {indexed}" if indexed
                    else ("Nicio pagina indexata" if self.manual.doc else "Manual indisponibil"))
        self._man_status = tk.Label(
            sel, text=idx_text,
            bg=self.PANEL, fg=self.GREY, font=("Helvetica", 8),
        )
        self._man_status.pack(side=tk.LEFT, padx=(8, 0))

        self._man_canvas = tk.Canvas(
            mf, width=MANUAL_W, height=CAM_H,
            bg=self.PANEL, highlightthickness=0,
        )
        self._man_canvas.pack(padx=4, pady=4)
        self._man_canvas.create_text(
            MANUAL_W // 2, CAM_H // 2,
            text="Intreaba despre un pas de asamblare\nsau selecteaza manual un numar de pas.",
            fill=self.GREY, font=("Helvetica", 11),
            justify=tk.CENTER, width=MANUAL_W - 20,
        )

        # ── control bar ───────────────────────────────────────────────────
        ctrl = tk.Frame(r, bg=self.BG)
        ctrl.pack(fill=tk.X, padx=10, pady=6)

        self._start_btn = tk.Button(
            ctrl, text="START",
            font=("Helvetica", 12, "bold"),
            bg=self.GREEN, fg="white", activebackground="#1e8449",
            relief=tk.FLAT, padx=22, pady=8,
            command=self._on_start,
        )
        self._start_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._stop_btn = tk.Button(
            ctrl, text="STOP",
            font=("Helvetica", 12, "bold"),
            bg=self.RED, fg="white", activebackground="#c0392b",
            relief=tk.FLAT, padx=22, pady=8,
            command=self._on_stop, state=tk.DISABLED,
        )
        self._stop_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._close_btn = tk.Button(
            ctrl, text="INCHIDE",
            font=("Helvetica", 12, "bold"),
            bg=self.GREY, fg="white", relief=tk.FLAT,
            padx=22, pady=8,
            command=self._on_close,
        )
        self._close_btn.pack(side=tk.RIGHT)

        self._status = tk.StringVar(
            value="Gata. Apasa START pentru a porni asistentul vocal."
        )
        tk.Label(
            r, textvariable=self._status,
            bg=self.ACCENT, fg=self.FG,
            font=("Helvetica", 9), anchor=tk.W, padx=10, pady=3,
        ).pack(fill=tk.X, side=tk.BOTTOM)

    # ══════════════════════════════════════════════════════════════════════
    #  TTS intercept
    # ══════════════════════════════════════════════════════════════════════

    def _install_tts_hook(self):
        orig = _tts_mod.tts_queue.put

        def _hook(item, *args, **kwargs):
            orig(item, *args, **kwargs)
            if isinstance(item, str) and self.root.winfo_exists():
                self.root.after(0, lambda t=item: self._on_spoken(t))

        _tts_mod.tts_queue.put = _hook

    def _on_spoken(self, text: str):
        m = re.search(r"\bpasul?\s+(\d+)\b", text, re.IGNORECASE)
        if m:
            self._show_step(int(m.group(1)))

    # ══════════════════════════════════════════════════════════════════════
    #  Manual page display
    # ══════════════════════════════════════════════════════════════════════

    def _show_step(self, step: int):
        imgs = self.manual.images_for_step(step)
        if not imgs:
            return

        self._manual_imgs = []
        for img in imgs:
            ratio  = CAM_H / img.height
            new_w  = max(1, int(img.width * ratio))
            scaled = img.resize((new_w, CAM_H), Image.LANCZOS)
            self._manual_imgs.append(ImageTk.PhotoImage(scaled))

        self._manual_idx = 0
        self._render_manual()

    def _render_manual(self):
        if not self._manual_imgs:
            return
        total = len(self._manual_imgs)
        idx   = self._manual_idx
        photo = self._manual_imgs[idx]

        cw = self._man_canvas.winfo_width()
        ch = self._man_canvas.winfo_height()
        if cw <= 1: cw = MANUAL_W
        if ch <= 1: ch = CAM_H

        self._man_canvas.delete("all")
        self._man_canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=photo)

        self._pg_lbl.config(text=f"pagina {idx + 1} / {total}")
        self._prev_btn.config(state=tk.NORMAL if idx > 0         else tk.DISABLED)
        self._next_btn.config(state=tk.NORMAL if idx < total - 1 else tk.DISABLED)

    def _man_prev(self):
        if self._manual_idx > 0:
            self._manual_idx -= 1
            self._render_manual()

    def _man_next(self):
        if self._manual_idx < len(self._manual_imgs) - 1:
            self._manual_idx += 1
            self._render_manual()

    def _on_show_step(self):
        txt = self._step_var.get().strip()
        if not txt.isdigit():
            return
        step = int(txt)
        if self.manual.images_for_step(step):
            self._show_step(step)
            self._last_step = step
        else:
            self._man_canvas.delete("all")
            self._man_canvas.create_text(
                MANUAL_W // 2, CAM_H // 2,
                text=f"Pasul {step} nu a fost gasit in manual.",
                fill=self.ORANGE, font=("Helvetica", 11),
                justify=tk.CENTER, width=MANUAL_W - 20,
            )
            self._pg_lbl.config(text="")
            self._prev_btn.config(state=tk.DISABLED)
            self._next_btn.config(state=tk.DISABLED)

    # ══════════════════════════════════════════════════════════════════════
    #  Camera feed
    # ══════════════════════════════════════════════════════════════════════

    def _start_camera(self):
        self._cam_stop.clear()
        self._cam_worker = CameraWorker(self._frame_q, self._cam_stop)
        self._cam_worker.start()

    def _poll(self):
        if not self.root.winfo_exists():
            return

        try:
            frame: Image.Image = self._frame_q.get_nowait()
            self._cam_photo = ImageTk.PhotoImage(frame)
            self._cam_canvas.delete("all")
            self._cam_canvas.create_image(0, 0, anchor=tk.NW, image=self._cam_photo)
            self._cam_status.config(
                text="Camera activa - YOLOv8 ruleaza",
                fg=self.GREEN,
            )
        except queue.Empty:
            pass

        # Auto-update PDF panel when assembly step changes
        step = get_current_assembly_step()
        if step is not None and step != self._last_step:
            self._last_step = step
            self._show_step(step)

        self.root.after(POLL_MS, self._poll)

    # ══════════════════════════════════════════════════════════════════════
    #  Session control
    # ══════════════════════════════════════════════════════════════════════

    def _on_start(self):
        if self._running:
            return

        self._status.set("Se initializeaza clientul OpenAI...")
        self.root.update()

        try:
            self._client = make_openai_client()
        except Exception as e:
            self._status.set(f"Eroare OpenAI: {e}")
            return

        clear_tts_queue()

        self._tts_stop = threading.Event()
        start_tts(self._tts_stop)

        self._dlg_stop   = threading.Event()
        self._dlg_thread = threading.Thread(
            target=dialog_loop,
            args=(self._client, self._dlg_stop),
            daemon=True, name="dialog",
        )
        self._dlg_thread.start()

        self._running = True
        self._start_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        self._status.set("Asistent activ - microfon pornit, vorbeste dupa ce TTS termina.")

    def _on_stop(self):
        if not self._running:
            return

        if self._dlg_stop:
            self._dlg_stop.set()
        stop_tts()
        clear_tts_queue()

        self._running = False
        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._status.set("Sesiune oprita. Apasa START pentru a relua.")

    def _on_close(self):
        self._cam_stop.set()
        if self._running:
            self._on_stop()
        self.root.after(300, self.root.destroy)

    def run(self):
        self.root.mainloop()


# ─────────────────────────────────────────────────────────────────────────────

def main():
    AssemblyGUI().run()


if __name__ == "__main__":
    main()
