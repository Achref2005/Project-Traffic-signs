"""Compare detector variants on the TEST split (+ video temporal-consistency proxies). Writes outputs/eval_results.csv.
   python evaluate.py --models baseline=path/best.pt transformer=path/best.pt [--temporal W.pt --video database/video.mp4]"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import traffic_signs  # noqa: F401
from traffic_signs.data import resolved_data_yaml
from traffic_signs.pipeline import SignDetector, VideoSession
from traffic_signs.temporal import TemporalClassifier, load_temporal


def speed(det, imgs, n=60):
    ts = []
    for im in imgs[:n]:
        t = time.perf_counter(); det.detect(im, conf=0.25); ts.append(time.perf_counter() - t)
    ts = np.array(ts[5:]) * 1000
    return float(ts.mean()), 1000 / float(ts.mean())


def video_metrics(det, tm, video, max_frames):
    s = VideoSession(det, tm); cap = cv2.VideoCapture(str(video)); n = 0
    seq_d, seq_r, t0 = {}, {}, time.perf_counter()
    while n < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        for d in s.update(f):
            seq_d.setdefault(d["track_id"], []).append(d["yolo_class"]); seq_r.setdefault(d["track_id"], []).append(d["refined_class"])
        n += 1
    fps = n / (time.perf_counter() - t0)
    flip = lambda seqs: (sum(a != b for q in seqs.values() for a, b in zip(q, q[1:])) / max(1, sum(len(q) - 1 for q in seqs.values())))
    return {"video_fps": fps, "tracks": len(seq_d), "mean_track_len": float(np.mean([len(q) for q in seq_d.values()])) if seq_d else 0,
            "class_flip_rate_detector": flip(seq_d), "class_flip_rate_lstm": flip(seq_r)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=weights.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--temporal", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--out", default="outputs/eval_results.csv")
    a = ap.parse_args()
    data = resolved_data_yaml()
    root = Path(__file__).parent / "database" / "car" / a.split / "images"
    imgs = [cv2.imread(str(p)) for p in sorted(root.glob("*"))[:65]]
    rows = []
    for spec in a.models:
        name, w = spec.split("=", 1)
        r = YOLO(w).val(data=data, split=a.split, imgsz=a.imgsz, device=a.device, plots=False, verbose=False, workers=0)
        p, rc = float(r.box.mp), float(r.box.mr)
        det = SignDetector(w, device=a.device, imgsz=a.imgsz)
        lat, fps = speed(det, imgs)
        row = {"model": name, "weights": w, "split": a.split, "precision": p, "recall": rc, "f1": 2 * p * rc / max(p + rc, 1e-9),
               "mAP50": float(r.box.map50), "mAP50-95": float(r.box.map), "latency_ms": lat, "fps": fps,
               "size_MB": Path(w).stat().st_size / 1e6, "params_M": sum(q.numel() for q in det.model.parameters()) / 1e6}
        if a.video:
            tm = load_temporal(a.temporal, det.device)[0] if a.temporal else TemporalClassifier(det.emb_dim, det.nc).to(det.device).eval()
            row.update({"lstm_" + k if not k.startswith(("class", "video")) else k: v for k, v in video_metrics(det, tm, a.video, a.max_frames).items()})
            row["lstm_trained"] = bool(a.temporal)
        rows.append(row); print(row)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, keys); wr.writeheader(); wr.writerows(rows)


if __name__ == "__main__":
    main()
