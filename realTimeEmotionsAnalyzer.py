"""
Real-Time Facial Expression Detector  (optimised)
===================================================
Uses webcam + Google Cloud Vision API to detect emotions live.

Optimisations vs original
--------------------------
* API calls run in a background thread  →  camera stays smooth / truly real-time
* Frame is downscaled to 640 px wide before encoding  →  faster + cheaper API
* Display loop capped at ~30 fps via waitKey(33)  →  lower CPU load
* Semi-transparent dark panel behind emotion bars  →  readable on any background
* Status bar fixed (two texts no longer drawn on the same pixel)
* FPS counter in top-right corner
* "Scanning…" badge while the background thread is active

Requirements
------------
    I have tried it on python 3.12
    pip install opencv-python requests

Usage
-----
    python face_expression_detector.py --api-key YOUR_API_KEY
    python face_expression_detector.py --api-key YOUR_API_KEY --interval 2

Controls
--------
    Q  — quit
    S  — snapshot
    +  — longer interval   (up to 30 s)
    -  — shorter interval  (down to 1 s)
"""

import cv2
import requests
import base64
import argparse
import time
import threading
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────────
VISION_URL    = "https://vision.googleapis.com/v1/images:annotate"
SCAN_INTERVAL = 3.0          # seconds between API calls (runtime-adjustable)
JPEG_QUALITY  = 65           # encode quality for the *API* copy (not display)
API_MAX_WIDTH = 640          # downscale to this width before sending to API
MAX_FACES     = 5
TARGET_FPS    = 30           # display frame-rate cap

SCORE = {
    "VERY_UNLIKELY": 0,
    "UNLIKELY":      25,
    "POSSIBLE":      50,
    "LIKELY":        75,
    "VERY_LIKELY":   100,
    "UNKNOWN":       0,
}

EMOTIONS = ["joyLikelihood", "sorrowLikelihood", "angerLikelihood", "surpriseLikelihood"]
LABELS   = ["Joy", "Sorrow", "Anger", "Surprise"]
COLORS   = [
    ( 50, 200,  50),   # Joy     — green  (BGR)
    (180,  80,  50),   # Sorrow  — blue-ish
    ( 50,  50, 220),   # Anger   — red
    ( 50, 180, 220),   # Surprise— amber-ish
]


# ── Overlay helpers ─────────────────────────────────────────────────────────────

def alpha_rect(img, x1, y1, x2, y2, color, alpha=0.55):
    """Draw a filled rectangle with transparency (in-place)."""
    sub = img[y1:y2, x1:x2]
    if sub.size == 0:
        return
    overlay = sub.copy()
    cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), color, -1)
    cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0, sub)


def draw_bar(img, x, y, label, score, color, width=160, height=14):
    cv2.putText(img, label, (x, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    # track
    cv2.rectangle(img, (x, y), (x + width, y + height), (80, 80, 80), -1)
    # fill
    filled = int(width * score / 100)
    if filled > 0:
        cv2.rectangle(img, (x, y), (x + filled, y + height), color, -1)
    # percentage
    cv2.putText(img, f"{score}%", (x + width + 6, y + height - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)


def draw_face_overlay(img, face_data, face_index=0, scale=1.0):
    """Bounding box + emotion bars for one face.
    scale = display_width / api_width  (corrects for pre-send downscaling)."""
    def s(v):  # scale a single coordinate value
        return int(v * scale)

    verts = face_data.get("boundingPoly", {}).get("vertices", [])
    if len(verts) >= 3:
        x1 = s(verts[0].get("x", 0));  y1 = s(verts[0].get("y", 0))
        x2 = s(verts[2].get("x", verts[0].get("x", 0) + 100))
        y2 = s(verts[2].get("y", verts[0].get("y", 0) + 100))
        cv2.rectangle(img, (x1, y1), (x2, y2), (80, 220, 80), 2)
        cv2.putText(img, f"Face {face_index + 1}", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 1, cv2.LINE_AA)

    # semi-transparent panel behind bars
    PANEL_X, PAD = 8, 6
    bar_h    = 22
    n        = len(EMOTIONS)
    panel_y1 = 14 + face_index * (n * bar_h + 16)
    panel_y2 = panel_y1 + n * bar_h + PAD * 2
    alpha_rect(img, PANEL_X, panel_y1, PANEL_X + 205, panel_y2,
               (20, 20, 20), alpha=0.55)

    for i, (key, label, color) in enumerate(zip(EMOTIONS, LABELS, COLORS)):
        score = SCORE.get(face_data.get(key, "UNKNOWN"), 0)
        bar_y = panel_y1 + PAD + 14 + i * bar_h
        draw_bar(img, PANEL_X + PAD, bar_y, label, score, color)


def draw_status(img, status_text, scan_interval, scans_done, next_in, fps, scanning):
    """Bottom status bar — two clearly separated text regions."""
    h, w = img.shape[:2]
    # dark strip
    alpha_rect(img, 0, h - 34, w, h, (20, 20, 20), alpha=0.75)

    # left: last event
    color = (100, 220, 100) if "detected" in status_text else (180, 180, 180)
    cv2.putText(img, status_text, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    # right: counters
    badge = "  [scanning…]" if scanning else ""
    right = (f"FPS:{fps:4.1f}  |  Scans:{scans_done}  |  "
             f"Next:{next_in:.1f}s  |  Int:{scan_interval:.0f}s  |  "
             f"Q=quit  S=snap  +/-=interval{badge}")
    (tw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
    cv2.putText(img, right, (w - tw - 8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (130, 130, 130), 1, cv2.LINE_AA)


# ── Vision API (runs in a worker thread) ───────────────────────────────────────

def analyze_frame(frame, api_key):
    """Downscale, encode, POST to Vision API.
    Returns (face_annotations, scale) where scale maps API coords → display coords."""
    h, w = frame.shape[:2]
    scale = 1.0
    if w > API_MAX_WIDTH:
        scale = w / API_MAX_WIDTH
        frame = cv2.resize(frame, (API_MAX_WIDTH, int(h / scale)),
                           interpolation=cv2.INTER_AREA)

    _, buf = cv2.imencode(".jpg", frame,
                          [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    b64 = base64.b64encode(buf.tobytes()).decode()

    payload = {"requests": [{
        "image": {"content": b64},
        "features": [{"type": "FACE_DETECTION", "maxResults": MAX_FACES}]
    }]}
    resp = requests.post(VISION_URL,
                         params={"key": api_key},
                         json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(data["error"]["message"])
    r = data.get("responses", [{}])[0]
    if "error" in r:
        raise RuntimeError(r["error"]["message"])
    return r.get("faceAnnotations", []), scale


# ── Main loop ──────────────────────────────────────────────────────────────────

def main(api_key):
    global SCAN_INTERVAL

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # ask driver for a decent fps; may be ignored but worth trying
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Camera open. Q=quit  S=snapshot  +/-=interval")

    last_scan_time = 0.0
    last_faces: list = []
    last_scale: float = 1.0
    status_msg  = "Waiting for first scan…"
    scans_done  = 0
    scanning    = False            # background thread active?
    scan_lock   = threading.Lock()

    fps         = 0.0
    frame_times: list = []        # rolling window for FPS

    frame_ms  = int(1000 / TARGET_FPS)   # waitKey delay ≈ 33 ms @ 30 fps

    def run_scan(frame_copy):
        nonlocal last_faces, last_scale, status_msg, scans_done, scanning
        try:
            faces, scale = analyze_frame(frame_copy, api_key)
            ts = datetime.now().strftime("%H:%M:%S")
            with scan_lock:
                last_faces = faces
                last_scale = scale
                scans_done += 1
                status_msg = (f"[{ts}] {len(faces)} face(s) detected"
                              if faces else f"[{ts}] No faces detected")
        except Exception as e:
            with scan_lock:
                status_msg = f"API error: {e}"
            print(f"API error: {e}")
        finally:
            scanning = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame.")
            break

        now = time.time()

        # ── FPS calculation (rolling 30-frame window) ───────────────────────
        frame_times.append(now)
        if len(frame_times) > 30:
            frame_times.pop(0)
        if len(frame_times) > 1:
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])

        # ── Trigger async scan ──────────────────────────────────────────────
        if not scanning and (now - last_scan_time >= SCAN_INTERVAL):
            last_scan_time = now
            scanning       = True
            t = threading.Thread(target=run_scan,
                                 args=(frame.copy(),), daemon=True)
            t.start()

        # ── Build display frame ─────────────────────────────────────────────
        display  = frame.copy()
        next_in  = max(0.0, SCAN_INTERVAL - (now - last_scan_time))

        with scan_lock:
            faces_snapshot = list(last_faces)
            scale_snapshot = last_scale
            msg_snapshot   = status_msg

        for i, face in enumerate(faces_snapshot):
            draw_face_overlay(display, face, i, scale_snapshot)

        draw_status(display, msg_snapshot, SCAN_INTERVAL,
                    scans_done, next_in, fps, scanning)

        cv2.imshow("Face Expression Detector  —  Q to quit", display)

        # ── Key handling ────────────────────────────────────────────────────
        key = cv2.waitKey(frame_ms) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fname, display)
            print(f"Snapshot saved: {fname}")
            with scan_lock:
                status_msg = f"Snapshot saved: {fname}"
        elif key in (ord("+"), ord("=")):
            SCAN_INTERVAL = min(SCAN_INTERVAL + 1, 30)
            print(f"Interval → {SCAN_INTERVAL:.0f}s")
        elif key == ord("-"):
            SCAN_INTERVAL = max(SCAN_INTERVAL - 1, 1)
            print(f"Interval → {SCAN_INTERVAL:.0f}s")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time facial expression detector")
    parser.add_argument("--api-key",  required=True,
                        help="Google Cloud Vision API key")
    parser.add_argument("--interval", type=float, default=3.0,
                        help="Seconds between API scans (default: 3)")
    args = parser.parse_args()
    SCAN_INTERVAL = args.interval
    main(args.api_key)