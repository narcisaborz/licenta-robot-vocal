import cv2
import numpy as np
import pyrealsense2 as rs
import pygame
from ultralytics import YOLO

from app_config import MODEL_PATH, CONF_TH, CLASS_COLORS, AUTO_DESCRIBE_NEW_OBJECTS
from agent_state import set_scene, get_focus, set_focus

_model = None


def _get_model():
    global _model
    if _model is None:
        print("[VISION] Se incarca modelul YOLO...", flush=True)
        _model = YOLO(MODEL_PATH)
    return _model


def _show_pygame(color_image, wait_ms=2000):
    try:
        pygame.display.init()
        screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption("YOLOv8 - Captura")
        rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        pygame.time.wait(wait_ms)
        pygame.display.quit()
    except Exception as e:
        print(f"[VISION] Nu pot afisa fereastra: {e}")


def capture_once(show_window: bool = True) -> list[dict]:
    """Single on-demand capture: open camera, grab one frame, run YOLO, close camera."""
    model = _get_model()

    pipeline = None
    try:
        ctx = rs.context()
        if len(ctx.devices) == 0:
            print("[VISION] Nicio camera conectata.")
            return []

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(config)

        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()
        depth_scale = 0.001

        # Skip frames so auto-exposure settles
        for _ in range(15):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return []

        color_image = np.asanyarray(color_frame.get_data())
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_image = np.asanyarray(depth_frame.get_data()) * depth_scale

        h, w, _ = color_image.shape
        results = model(color_image, verbose=False)
        dets = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = model.names[cls]
                if conf < CONF_TH:
                    continue

                x1 = max(0, min(w - 1, x1))
                x2 = max(0, min(w, x2))
                y1 = max(0, min(h - 1, y1))
                y2 = max(0, min(h, y2))

                roi = depth_image[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                roi_valid = roi[(roi > 0) & (~np.isnan(roi))]
                if roi_valid.size == 0:
                    continue

                dist_m = float(np.median(roi_valid))
                dets.append({"name": name, "dist_m": dist_m, "conf": conf})

                color = CLASS_COLORS.get(name, (0, 180, 255))
                cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 3)
                cv2.putText(color_image, f"{name} {dist_m:.2f} m",
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        dets.sort(key=lambda d: d["dist_m"])

        if show_window:
            _show_pygame(color_image, wait_ms=2000)

        return dets

    except Exception as e:
        print(f"[VISION] Eroare la capturare: {e}")
        return []
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass