#!/usr/bin/env python3
"""
BaluVision v8.1 — Performance‑tuned live detection
Model  : YOLOv8 (selectable) + Haar faces
"""
import os, sys
# suppress NNPACK "unsupported hardware" spam on old CPUs
os.environ["NNPACK_LOG_LEVEL"] = "0"
devnull = open(os.devnull, "w")
old_stderr_fd = os.dup(2)
os.dup2(devnull.fileno(), 2)

import subprocess, threading, time, queue, logging, tkinter as tk
from tkinter import ttk
import importlib.util

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("baluvision")
log.info("Python %s  |  pid %d", sys.version.split()[0], os.getpid())

# ── deps ─────────────────────────────────────────────────────────────────────
DEPS = [
    ("cv2",         "opencv-python"),
    ("numpy",       "numpy"),
    ("PIL",         "Pillow"),
    ("torch",       "torch --index-url https://download.pytorch.org/whl/cpu"),
    ("ultralytics", "ultralytics"),
]

def _ensure_deps():
    missing = [(i, p) for i, p in DEPS
               if not importlib.util.find_spec(i.split(".")[0])]
    if not missing:
        log.info("All deps present.")
        return
    log.warning("Installing: %s", [p for _, p in missing])
    for _, pkg in missing:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "--break-system-packages"] + pkg.split(),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    log.info("Deps installed.")

_ensure_deps()

import cv2
import numpy as np
from PIL import Image, ImageTk
import torch

# Set CPU threads to 2
torch.set_num_threads(2)
log.info("Torch CPU threads set to %d", torch.get_num_threads())

# ── COCO-80 labels (correct spelling) ───────────────────────────────────────
COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# neon palette — cycles per class index (RGB, converted to BGR when drawing)
_PALETTE_RGB = [
    (57, 255, 20), (255, 165, 0), (0, 229, 255), (255, 50, 50),
    (180, 0, 255), (255, 230, 0), (0, 255, 180), (255, 100, 200),
]
def _color(label_idx):           # returns BGR tuple for OpenCV
    r, g, b = _PALETTE_RGB[label_idx % len(_PALETTE_RGB)]
    return (b, g, r)

# ── constants ─────────────────────────────────────────────────────────────────
DISPLAY_W, DISPLAY_H = 480, 360   # smaller display = faster UI
INFER_INTERVAL       = 0.25       # seconds between detections (adjustable)
THRESHOLD            = 0.50       # confidence cutoff
MAX_CAM_IDX          = 6
FACE_CASCADE         = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
INFER_IMGSZ          = 160        # further downscaled for speed
FRAME_SKIP           = 2          # run inference every N frames (1 = every frame)

C = {
    "bg":    "#0a0c0a", "panel": "#0f130f", "border": "#1a2a1a",
    "green": "#39ff14", "dim":   "#3a5c3a", "text":   "#b8d4b8",
    "warn":  "#e8c84a", "face":  "#00e5ff", "bar_bg": "#141e14",
}

# Available YOLOv8 models
MODEL_OPTIONS = {
    "YOLOv8 Nano":  "yolov8n.pt",
    "YOLOv8 Small": "yolov8s.pt",
    "YOLOv8 Medium":"yolov8m.pt",
    "YOLOv8 Large": "yolov8l.pt",
}

# ── camera scan ───────────────────────────────────────────────────────────────
def scan_cameras():
    found = []
    log.info("Scanning /dev/video0–%d …", MAX_CAM_IDX - 1)
    for idx in range(MAX_CAM_IDX):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            lbl = f"/dev/video{idx}  ({w}x{h})"
            log.info("  [%d] %s", idx, lbl)
            found.append((idx, lbl))
            cap.release()
        else:
            log.debug("  /dev/video%d — absent", idx)
    if not found:
        log.warning("No cameras found.")
    return found


# ── model loader ──────────────────────────────────────────────────────────────
class ModelLoader(threading.Thread):
    def __init__(self, on_ready, on_status, model_path):
        super().__init__(daemon=True, name="ModelLoader")
        self.on_ready  = on_ready
        self.on_status = on_status
        self.model_path = model_path

    def run(self):
        try:
            self.on_status("Importing ultralytics…")
            t0 = time.perf_counter()
            from ultralytics import YOLO
            log.info("ultralytics imported in %.2fs", time.perf_counter() - t0)

            self.on_status(f"Loading {os.path.basename(self.model_path)} …")
            t1 = time.perf_counter()
            model = YOLO(self.model_path)
            log.info("%s ready in %.2fs", self.model_path, time.perf_counter() - t1)

            # Haar face cascade
            self.on_status("Loading face cascade…")
            cascade = cv2.CascadeClassifier(FACE_CASCADE)
            log.info("Face cascade: %s", "OK" if not cascade.empty() else "MISSING")

            self.on_status("Ready.")
            self.on_ready(model, cascade)

        except Exception:
            log.exception("ModelLoader failed")
            self.on_status("ERROR — see console")


# ── inference worker ──────────────────────────────────────────────────────────
class InferenceWorker(threading.Thread):
    def __init__(self, model, cascade):
        super().__init__(daemon=True, name="InferenceWorker")
        self.model   = model
        self.cascade = cascade
        self._q      = queue.Queue(maxsize=1)
        self._stop   = threading.Event()
        self.cb      = None
        # Lower thread priority if possible
        try:
            os.nice(10)
        except:
            pass

    def submit(self, frame_bgr, cb):
        self.cb = cb
        try:
            self._q.put_nowait(frame_bgr)
        except queue.Full:
            log.debug("Frame dropped — worker busy")

    def stop(self):
        self._stop.set()

    def run(self):
        log.info("InferenceWorker running.")
        while not self._stop.is_set():
            try:
                frame = self._q.get(timeout=0.4)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            try:
                result = self._infer(frame)
                result["elapsed"] = time.perf_counter() - t0
                log.debug("Inference %.3fs | detections=%d | faces=%d",
                         result["elapsed"],
                         len(result["detections"]),
                         len(result["faces"]))
                if self.cb:
                    self.cb(result)
            except Exception:
                log.exception("Inference error")
                if self.cb:
                    self.cb({"error": True, "detections": [],
                             "faces": [], "elapsed": 0})

    def _infer(self, frame_bgr):
        # YOLOv8 inference – model handles resizing internally
        results = self.model(frame_bgr, imgsz=INFER_IMGSZ, conf=THRESHOLD, verbose=False)
        dets = results[0].boxes

        detections = []
        if dets is not None:
            boxes = dets.xyxy.cpu().numpy()
            scores = dets.conf.cpu().numpy()
            classes = dets.cls.cpu().numpy().astype(int)

            for box, score, cls in zip(boxes, scores, classes):
                if score < THRESHOLD:
                    continue
                name = COCO[cls] if cls < len(COCO) else "?"
                x1, y1, x2, y2 = map(int, box)
                detections.append({
                    "label": name,
                    "score": float(score),
                    "box": (x1, y1, x2, y2),
                    "color": _color(cls),
                    "class_idx": cls,
                })

        # Haar faces – downscale even more for speed
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.resize(gray, (0, 0), fx=0.4, fy=0.4)  # 40% of original
        rects = self.cascade.detectMultiScale(
            small_gray,
            scaleFactor=1.2,       # faster scanning
            minNeighbors=3,
            minSize=(30, 30)
        )
        faces = [tuple(int(v / 0.4) for v in rect)
                 for rect in (rects if len(rects) else [])]

        return {"detections": detections, "faces": faces}


# ── application ───────────────────────────────────────────────────────────────
class App:
    def __init__(self, root):
        self.root      = root
        self.cap       = None
        self.worker    = None
        self._running  = False
        self._busy     = False
        self._last_inf = 0.0
        self._photo    = None
        self._result   = None
        self._cams     = []
        self._frame_after_id = None
        self._frame_counter = 0      # for frame skipping
        self._last_seen_hash = None  # to avoid redundant UI updates

        self._cam_var     = tk.StringVar(value="Scanning…")
        self._model_var   = tk.StringVar(value="YOLOv8 Small")
        self._status_var  = tk.StringVar(value="Loading model…")
        self._fps_var     = tk.StringVar(value="")
        self._det_var     = tk.StringVar(value="Objects: –")
        self._face_var    = tk.StringVar(value="Faces:   –")
        self._time_var    = tk.StringVar(value="Infer:   –")

        self._build_ui()
        log.info("Window open.")
        threading.Thread(target=lambda: self.root.after(
            0, lambda: self._populate_dropdown(scan_cameras())
        ), daemon=True).start()
        self._start_model_loader()

    def _start_model_loader(self):
        model_path = MODEL_OPTIONS[self._model_var.get()]
        ModelLoader(self._on_ready, self._on_status, model_path).start()

    # ── camera ────────────────────────────────────────────────────────────────
    def _populate_dropdown(self, cams):
        self._cams = cams
        menu = self._cam_menu["menu"]
        menu.delete(0, "end")
        if not cams:
            self._cam_var.set("No camera found")
            return
        for idx, lbl in cams:
            menu.add_command(label=lbl,
                             command=lambda i=idx, l=lbl: self._pick_cam(i, l))
        self._cam_var.set(cams[0][1])
        log.info("Dropdown: %d camera(s).", len(cams))

    def _pick_cam(self, idx, lbl):
        if not self.worker:
            return
        self._cam_menu.config(state="disabled")
        self._cam_var.set(lbl)
        log.info("Switching to camera %d", idx)
        self._close_cam()
        self.root.after(200, lambda: self._open_cam(idx))

    def _open_cam(self, idx):
        log.info("Opening /dev/video%d …", idx)
        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            log.error("Cannot open camera %d", idx)
            self._status_var.set(f"Cannot open /dev/video{idx}")
            self._cam_menu.config(state="normal")
            return
        # Request 640x480 but we'll display smaller
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS,          15)
        log.info("Camera open — %dx%d",
                 int(self.cap.get(3)), int(self.cap.get(4)))
        self._running  = True
        self._busy     = False
        self._last_inf = 0.0
        self._result   = None
        self._frame_counter = 0
        self._status_var.set(
            f"Live · /dev/video{idx} · {self._model_var.get()} · {INFER_IMGSZ}px"
        )
        self._dot.config(fg=C["green"])
        self._frame_loop()
        self._cam_menu.config(state="normal")

    def _close_cam(self):
        self._running = False
        if self._frame_after_id is not None:
            self.root.after_cancel(self._frame_after_id)
            self._frame_after_id = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self._dot.config(fg=C["dim"])
        log.info("Camera closed.")

    # ── model ─────────────────────────────────────────────────────────────────
    def _on_status(self, msg):
        log.info("[Loader] %s", msg)
        self.root.after(0, lambda: self._status_var.set(msg))

    def _on_ready(self, model, cascade):
        log.info("Model ready.")
        self.worker = InferenceWorker(model, cascade)
        self.worker.start()
        self.root.after(0, self._post_load)

    def _post_load(self):
        self._prog.stop()
        self._prog.pack_forget()
        if self._cams:
            self._open_cam(self._cams[0][0])
        else:
            self._status_var.set("Model ready — no camera found.")

    # ── frame loop ────────────────────────────────────────────────────────────
    def _frame_loop(self):
        if not self._running:
            return
        t0 = time.perf_counter()

        if self.cap is None or not self.cap.isOpened():
            log.warning("Camera not available, stopping loop.")
            self._running = False
            return

        ret, frame = self.cap.read()
        if not ret:
            log.warning("Frame read failed, retrying in 100ms")
            self._frame_after_id = self.root.after(100, self._frame_loop)
            return

        self._frame_counter += 1

        annotated = frame.copy()
        if self._result and not self._result.get("error"):
            self._draw(annotated, self._result)
        self._show(annotated)

        now = time.time()
        # Frame skipping + interval throttling
        should_infer = (self._frame_counter % FRAME_SKIP == 0 and
                        not self._busy and
                        (now - self._last_inf) >= INFER_INTERVAL)
        if should_infer:
            self._last_inf = now
            self._busy     = True
            self._dot.config(fg=C["warn"])
            self.worker.submit(frame, self._on_result)

        elapsed = time.perf_counter() - t0
        self._fps_var.set(f"{1/max(elapsed,1e-6):.0f} fps")
        # Aim for ~25 fps UI update
        self._frame_after_id = self.root.after(
            max(1, int((1/25 - elapsed) * 1000)), self._frame_loop
        )

    def _draw(self, frame, result):
        for det in result["detections"]:
            x1, y1, x2, y2 = det["box"]
            bgr   = det["color"]
            label = f"{det['label']} {det['score']*100:.0f}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            ty = max(y1 - 4, th + 4)
            cv2.rectangle(frame, (x1, ty - th - 3), (x1 + tw + 4, ty + 1), bgr, -1)
            cv2.putText(frame, label, (x1 + 2, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

        for (x, y, w, h) in result["faces"]:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 229, 0), 2)
            cv2.putText(frame, "face", (x, max(y - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 229, 0), 1, cv2.LINE_AA)

    def _show(self, frame):
        # Resize display frame to reduce CPU load
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize(
            (DISPLAY_W, DISPLAY_H), Image.BILINEAR)
        photo = ImageTk.PhotoImage(img)
        self._photo = photo
        self.canvas.create_image(0, 0, anchor="nw", image=photo)

    def _on_result(self, result):
        self._result = result
        self._busy   = False
        self.root.after(0, lambda: self._update_sidebar(result))
        self.root.after(0, lambda: self._dot.config(fg=C["green"]))

    def _update_sidebar(self, r):
        if r.get("error"):
            self._status_var.set("Inference error — check console")
            return

        dets  = r["detections"]
        faces = r["faces"]

        self._det_var.set(f"Objects: {len(dets)}")
        self._face_var.set(f"Faces:   {len(faces)}")
        self._time_var.set(f"Infer:   {r['elapsed']:.2f}s")

        # Only rebuild sidebar if detection list changed (simple hash)
        new_hash = hash(tuple((d["label"], d["score"]) for d in dets))
        if new_hash == self._last_seen_hash:
            return
        self._last_seen_hash = new_hash

        for w in self._seen_frame.winfo_children():
            w.destroy()
        seen = {}
        for d in dets:
            seen[d["label"]] = max(seen.get(d["label"], 0), d["score"])
        for label, score in sorted(seen.items(), key=lambda x: -x[1]):
            row = tk.Frame(self._seen_frame, bg=C["panel"])
            row.pack(fill="x", pady=1)
            cls_idx = next((d["class_idx"] for d in dets if d["label"] == label), 0)
            r_rgb = _PALETTE_RGB[cls_idx % len(_PALETTE_RGB)]
            hex_c = f"#{r_rgb[0]:02x}{r_rgb[1]:02x}{r_rgb[2]:02x}"
            tk.Label(row, text=f"  {label}",
                     font=("Courier", 9), fg=hex_c,
                     bg=C["panel"], anchor="w").pack(side="left", fill="x",
                                                      expand=True)
            tk.Label(row, text=f"{score*100:.0f}%",
                     font=("Courier", 9), fg=C["dim"],
                     bg=C["panel"]).pack(side="right")

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        r = self.root
        r.title("BaluVision v8.1")
        r.configure(bg=C["bg"])
        r.resizable(False, False)

        sty = ttk.Style(); sty.theme_use("clam")
        for name, fg in [("Bar", C["green"]), ("Load", C["green"])]:
            sty.configure(f"{name}.Horizontal.TProgressbar",
                           troughcolor=C["bar_bg"], background=fg,
                           bordercolor=C["border"],
                           lightcolor=fg, darkcolor=fg)
        sty.configure("Cam.TMenubutton",
                       background=C["panel"], foreground=C["text"],
                       relief="flat", font=("Courier", 9))
        sty.map("Cam.TMenubutton",
                 background=[("active", C["border"])],
                 foreground=[("active", C["green"])])

        # header
        hdr = tk.Frame(r, bg=C["bg"])
        hdr.pack(fill="x", padx=12, pady=(10, 2))
        tk.Label(hdr, text="BaluVision",
                 font=("Courier", 16, "bold"),
                 fg=C["green"], bg=C["bg"]).pack(side="left")
        tk.Label(hdr, text="  Optimized · CPU",
                 font=("Courier", 8), fg=C["dim"],
                 bg=C["bg"]).pack(side="left", pady=3)
        self._dot = tk.Label(hdr, text="[O]",
                              font=("Courier", 14), fg=C["dim"], bg=C["bg"])
        self._dot.pack(side="right")

        # camera selector
        cam_row = tk.Frame(r, bg=C["bg"])
        cam_row.pack(fill="x", padx=12, pady=(0, 4))
        tk.Label(cam_row, text="DEVICE",
                 font=("Courier", 8, "bold"), fg=C["dim"],
                 bg=C["bg"], width=6, anchor="w").pack(side="left")
        self._cam_menu = ttk.OptionMenu(
            cam_row, self._cam_var, "Scanning…", style="Cam.TMenubutton")
        self._cam_menu.pack(side="left", fill="x", expand=True)

        # model selector
        model_row = tk.Frame(r, bg=C["bg"])
        model_row.pack(fill="x", padx=12, pady=(0, 4))
        tk.Label(model_row, text="MODEL",
                 font=("Courier", 8, "bold"), fg=C["dim"],
                 bg=C["bg"], width=6, anchor="w").pack(side="left")
        model_menu = ttk.OptionMenu(
            model_row, self._model_var, "YOLOv8 Small",
            *MODEL_OPTIONS.keys(),
            command=self._on_model_change,
            style="Cam.TMenubutton")
        model_menu.pack(side="left", fill="x", expand=True)

        # body
        body = tk.Frame(r, bg=C["bg"])
        body.pack(padx=12, pady=2)

        border = tk.Frame(body, bg=C["border"], padx=1, pady=1)
        border.grid(row=0, column=0)
        self.canvas = tk.Canvas(border, width=DISPLAY_W, height=DISPLAY_H,
                                bg="#000", highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_text(
            DISPLAY_W//2, DISPLAY_H//2,
            text="Loading model…", fill=C["dim"], font=("Courier", 14))

        # sidebar
        side = tk.Frame(body, bg=C["panel"], width=200)
        side.grid(row=0, column=1, sticky="ns", padx=(8, 0))
        side.grid_propagate(False)

        tk.Label(side, text="DETECTIONS",
                 font=("Courier", 9, "bold"),
                 fg=C["green"], bg=C["panel"]).pack(
                     anchor="w", padx=10, pady=(10, 2))
        tk.Label(side,
                 text="YOLOv8 · 80 COCO",
                 font=("Courier", 7), fg=C["dim"],
                 bg=C["panel"], justify="left").pack(
                     anchor="w", padx=10, pady=(0, 8))

        for var, color in [
            (self._det_var,  C["warn"]),
            (self._face_var, C["face"]),
            (self._time_var, C["dim"]),
        ]:
            tk.Label(side, textvariable=var,
                     font=("Courier", 9), fg=color,
                     bg=C["panel"], anchor="w").pack(
                         fill="x", padx=10, pady=1)

        tk.Frame(side, bg=C["border"], height=1).pack(
            fill="x", padx=8, pady=8)

        tk.Label(side, text="SEEN",
                 font=("Courier", 9, "bold"),
                 fg=C["green"], bg=C["panel"]).pack(
                     anchor="w", padx=10, pady=(0, 4))

        self._seen_frame = tk.Frame(side, bg=C["panel"])
        self._seen_frame.pack(fill="x")

        tk.Label(side,
                 text=f"\nthreshold: {THRESHOLD*100:.0f}%\n"
                      f"imgsz:     {INFER_IMGSZ}px\n"
                      f"skip:      every {FRAME_SKIP} frames\n"
                      "haar:      optimized",
                 font=("Courier", 7), fg=C["dim"],
                 bg=C["panel"], justify="left").pack(
                     anchor="w", padx=10, pady=(8, 0))

        # progress bar
        self._prog = ttk.Progressbar(
            r, mode
