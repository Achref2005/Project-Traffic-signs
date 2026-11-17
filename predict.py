"""Offline inference.  python predict.py --weights W --source img.jpg|video.mp4 [--temporal weights/temporal_lstm.pt]"""
import argparse
from pathlib import Path

import cv2

from traffic_signs.draw import draw_dets
from traffic_signs.pipeline import SignDetector, VideoSession
from traffic_signs.temporal import TemporalClassifier, load_temporal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--temporal", default=None, help="trained LSTM checkpoint (untrained pass-through if omitted)")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--out", default="outputs/predictions")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    det = SignDetector(a.weights, imgsz=a.imgsz)
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(a.source)
    if src.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:          # image mode: detector only
        img = cv2.imread(str(src)); r = det.detect(img, conf=a.conf)
        dets = [{"yolo_class": det.names[int(c)], "yolo_conf": round(float(s), 3), "box": [float(v) for v in b]}
                for b, s, c in zip(r["boxes"], r["conf"], r["cls"])]
        cv2.imwrite(str(out_dir / src.name), draw_dets(img, dets)); print(dets)
        return
    tm = load_temporal(a.temporal, det.device)[0] if a.temporal else TemporalClassifier(det.emb_dim, det.nc).to(det.device).eval()
    sess = VideoSession(det, tm)
    cap = cv2.VideoCapture(str(src)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(3)), int(cap.get(4))
    vw = cv2.VideoWriter(str(out_dir / (src.stem + "_tracked.mp4")), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while True:
        ok, f = cap.read()
        if not ok or (a.max_frames and n >= a.max_frames):
            break
        vw.write(draw_dets(f, sess.update(f, conf=a.conf))); n += 1
    vw.release(); print(f"{n} frames -> {out_dir}")


if __name__ == "__main__":
    main()
