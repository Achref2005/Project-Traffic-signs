"""
Traffic-Sign inference service
--- authors note: loads the two models trained by train.ipynb and serves one endpoint:
POST /infer  -> multipart image -> YOLOv8 detection + GTSRB CNN classification per crop
Run AFTER train.ipynb finished:   python server/infer_api.py     (port 8001)
"""
import base64
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RUNS_DIR = ROOT / "runs"
IMG_DET_SIZE = 416      # size YOLOv8 was trained with

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DET_CLASSES = ["Green Light", "Red Light", "Speed Limit 10", "Speed Limit 100",
               "Speed Limit 110", "Speed Limit 120", "Speed Limit 20", "Speed Limit 30",
               "Speed Limit 40", "Speed Limit 50", "Speed Limit 60", "Speed Limit 70",
               "Speed Limit 80", "Speed Limit 90", "Stop"]


# --- classifier architecture (mirrors TrafficSignCNN in train.ipynb) ---
class TrafficSignCNN(nn.Module):
    def __init__(self, n_classes=43, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Dropout(dropout))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class InferenceEngine:
    def __init__(self):
        self.det_model = None
        self.clf_model = None
        self.cnn_classes = []
        self.cnn_img_size = 48
        self.det_ready = False
        self.clf_ready = False
        self.detector_path = None

    def load(self):
        t0 = time.time()
        # --- YOLOv8 detector ---
        best = sorted(RUNS_DIR.rglob("yolov8n_traffic_signs*/weights/best.pt"),
                      key=lambda p: p.stat().st_mtime)
        if not best:
            raise FileNotFoundError(
                "no trained detector found (runs/yolov8n_traffic_signs*/weights/best.pt) — run train.ipynb first")
        self.detector_path = str(best[-1].resolve())
        from ultralytics import YOLO
        self.det_model = YOLO(self.detector_path).to(device)
        self.det_model.model.eval()
        self.det_ready = True

        # --- GTSRB CNN classifier ---
        self.cnn_classes = DET_CLASSES  # placeholder, replaced by meta below
        meta = MODELS_DIR / "gtsrb_cnn_scratch_meta.pt"
        if meta.exists():
            self.cnn_classes = torch.load(meta, map_location="cpu", weights_only=False)["classes"]
        state = torch.load(MODELS_DIR / "gtsrb_cnn_scratch.pth", map_location=device)
        self.cnn_img_size = 48
        self.clf_model = TrafficSignCNN(n_classes=len(self.cnn_classes)).to(device)
        self.clf_model.load_state_dict(state)
        self.clf_model.eval()
        self.clf_ready = True

        # transforms (same normalization as training: /2 then +0.5)
        from torchvision import transforms
        from PIL import Image
        self._tf = transforms.Compose([
            transforms.Resize((self.cnn_img_size, self.cnn_img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self._Image = Image
        print(f"models loaded in {time.time()-t0:.1f}s | detector={self.detector_path} | device={device}")

    def classify_crop(self, rgb_crop):
        t = self._tf(self._Image.fromarray(rgb_crop)).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = self.clf_model(t).softmax(1)[0]
        conf, idx = probs.max(0)
        return self.cnn_classes[int(idx)], float(conf)

    def infer(self, img_bgr, conf_th=0.30, imgsz=IMG_DET_SIZE):
        res = self.det_model.predict(
            source=img_bgr, conf=conf_th, imgsz=int(imgsz), verbose=False)[0]
        H, W = img_bgr.shape[:2]
        out = img_bgr.copy()
        detections = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)
            rgb_crop = cv2.cvtColor(out[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            det_name = DET_CLASSES[int(b.cls)]
            cnn_name, cnn_conf = "", 0.0
            if rgb_crop.size and min(x2 - x1, y2 - y1) > 8:
                cnn_name, cnn_conf = self.classify_crop(rgb_crop)
            detections.append({
                "yolo_class": det_name, "yolo_conf": round(float(b.conf), 3),
                "cnn_class": cnn_name, "cnn_conf": round(cnn_conf, 3),
                "box": [x1, y1, x2, y2]})
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(out, f"{det_name} {float(b.conf):.2f}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
            if cnn_name:
                cv2.putText(out, f"{cnn_name}", (x1, min(H - 5, y2 + 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 80), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("could not encode annotated image")
        annotated = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        return {"width": W, "height": H, "detections": detections, "annotated": annotated}


engine = InferenceEngine()

app = FastAPI(title="Traffic Sign Recognition API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"detector_ready": engine.det_ready, "classifier_ready": engine.clf_ready,
            "detector": engine.detector_path, "device": str(device)}


@app.post("/infer")
async def infer(image: UploadFile = File(...), conf: float = 0.30, imgsz: int = IMG_DET_SIZE):
    imgsz = int(min(1600, max(256, imgsz)))          # keep a sane range
    data = await image.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image (jpg/png expected)")
    try:
        result = engine.infer(img, conf_th=conf, imgsz=imgsz)
    except Exception as e:
        raise HTTPException(500, f"inference failed: {e}") from e
    return result


@app.on_event("startup")
def startup():
    try:
        engine.load()
    except FileNotFoundError as e:
        print("WARNING —", e)
        print("UI will start, but inference returns 503 until train.ipynb has been run.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
