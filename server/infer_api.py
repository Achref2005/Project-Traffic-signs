"""Traffic-Sign inference service (FastAPI, port 8001).

POST /infer                    multipart `image`           -> IMAGE mode: detector only (YOLOv8 [+Transformer])
POST /infer?session=<id>       multipart `image` (frame)   -> VIDEO mode: detector -> ByteTrack -> LSTM refinement
DELETE /session/{id}           end a video stream (frees its tracker + LSTM state)
GET  /health
Weights are loaded once at startup. Each video stream (session id) owns its own tracker and LSTM states.
Env: DETECTOR_WEIGHTS, TEMPORAL_WEIGHTS to override the default weight search.
"""
import base64
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from traffic_signs.draw import draw_dets  # noqa: E402
from traffic_signs.pipeline import SignDetector, VideoSession  # noqa: E402
from traffic_signs.temporal import TemporalClassifier, load_temporal  # noqa: E402

IMG_DET_SIZE = 416
MAX_SESSIONS, SESSION_TTL = 8, 120.0     # idle streams are evicted -> bounded memory, no stale state reuse


def find_detector_weights():
    if os.environ.get("DETECTOR_WEIGHTS"):
        return Path(os.environ["DETECTOR_WEIGHTS"])
    def epochs(run_dir):          # legacy baseline: prefer the most-trained run, then the latest
        try:
            return sum(1 for _ in open(run_dir / "results.csv"))
        except OSError:
            return 0
    legacy = sorted((p for p in (ROOT / "runs").rglob("yolov8n_traffic_signs*/weights/best.pt")),
                    key=lambda p: (epochs(p.parent.parent), int(p.parent.parent.name.rsplit('-', 1)[-1]) if p.parent.parent.name[-1].isdigit() else 0),
                    reverse=True)
    cands = [ROOT / "weights" / "traffic_signs_transformer.pt", ROOT / "runs" / "train" / "yolov8n_transformer" / "weights" / "best.pt", *legacy]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError("no trained detector found - run train_detector.py (or set DETECTOR_WEIGHTS)")


class InferenceEngine:
    def __init__(self):
        self.det = None
        self.temporal = None
        self.temporal_trained = False
        self.sessions = {}
        self.error = None

    def load(self):
        t0 = time.time()
        w = find_detector_weights()
        self.det = SignDetector(w, imgsz=IMG_DET_SIZE)
        tw = Path(os.environ.get("TEMPORAL_WEIGHTS", ROOT / "weights" / "temporal_lstm.pt"))
        if tw.exists():
            self.temporal, _ = load_temporal(tw, self.det.device)
            self.temporal_trained = True
        else:      # zero-initialised residual head == exact pass-through of the detector until an LSTM is trained
            self.temporal = TemporalClassifier(self.det.emb_dim, self.det.nc).to(self.det.device).eval()
        print(f"loaded in {time.time() - t0:.1f}s | detector={w} (transformer={self.det.has_transformer}) "
              f"| lstm_trained={self.temporal_trained} | device={self.det.device}")

    def session(self, sid):
        now = time.time()
        for k in [k for k, s in self.sessions.items() if now - s.last_used > SESSION_TTL]:
            del self.sessions[k]
        if sid not in self.sessions:
            if len(self.sessions) >= MAX_SESSIONS:
                del self.sessions[min(self.sessions, key=lambda k: self.sessions[k].last_used)]
            self.sessions[sid] = VideoSession(self.det, self.temporal)
        s = self.sessions[sid]
        s.last_used = now
        return s

    def infer(self, img, conf, imgsz, session=None):
        H, W = img.shape[:2]
        if session:                                                  # VIDEO mode
            dets = self.session(session).update(img, conf=conf, imgsz=imgsz)
        else:                                                        # IMAGE mode (no tracker / LSTM)
            r = self.det.detect(img, conf=conf, imgsz=imgsz)
            dets = [{"yolo_class": self.det.names[int(c)], "yolo_conf": round(float(s), 3),
                     "box": [round(float(v), 1) for v in b]} for b, s, c in zip(r["boxes"], r["conf"], r["cls"])]
        ok, buf = cv2.imencode(".jpg", draw_dets(img, dets), [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("could not encode annotated image")
        return {"mode": "video" if session else "image", "width": W, "height": H, "detections": dets,
                "temporal_trained": self.temporal_trained if session else None,
                "annotated": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()}


engine = InferenceEngine()
app = FastAPI(title="Traffic Sign Recognition API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"detector_ready": engine.det is not None, "detector": engine.det.weights if engine.det else None,
            "transformer": engine.det.has_transformer if engine.det else None,
            "lstm_trained": engine.temporal_trained, "sessions": len(engine.sessions),
            "device": str(engine.det.device) if engine.det else None, "error": engine.error}


@app.post("/infer")
async def infer(image: UploadFile = File(...), conf: float = 0.30, imgsz: int = IMG_DET_SIZE, session: str = ""):
    if engine.det is None:
        raise HTTPException(503, engine.error or "model not loaded")
    imgsz = int(min(1600, max(256, imgsz)))
    img = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image (jpg/png expected)")
    try:
        return engine.infer(img, conf, imgsz, session[:64] or None)
    except Exception as e:
        raise HTTPException(500, f"inference failed: {e}") from e


@app.delete("/session/{sid}")
def end_session(sid: str):
    return {"closed": engine.sessions.pop(sid, None) is not None}


@app.on_event("startup")
def startup():
    try:
        engine.load()
    except Exception as e:
        engine.error = str(e)
        print("WARNING -", e, "\nUI will start, but /infer returns 503 until a detector is available.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
