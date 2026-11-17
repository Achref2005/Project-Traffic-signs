"""Build LSTM training sequences from VIDEOS (detector -> ByteTrack -> per-track feature sequences).

Static images carry no temporal information, so the current dataset (images only) cannot train the LSTM.
Needed: videos with consistent identities. Labels per track come from, in order of preference:
  1. --labels CSV with columns  video,track_id,class_name   (ground truth; get track ids from the *_tracks.csv written here)
  2. otherwise PSEUDO-labels = confidence-weighted majority vote of the detector over the whole track
     (weak supervision: teaches temporal consistency, cannot fix a class the detector systematically gets wrong).
Split is done per VIDEO (no leakage). With a single video a temporal split is used: first 70% of frames train,
last 20% val, tracks crossing a boundary are dropped.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from traffic_signs.pipeline import SignDetector, VideoSession


def extract(det, video, conf, min_len, every):
    cap = cv2.VideoCapture(str(video)); n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sess = VideoSession(det, None, collect=True)
    f = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if f % every == 0:
            sess.update(img, conf=conf)
        f += 1
    tracks = []
    for tid, rec in sess.records.items():
        if len(rec["feats"]) < min_len:
            continue
        feats = torch.stack(rec["feats"]).cpu()
        sc = feats[:, det.emb_dim:det.emb_dim + det.nc]
        tracks.append({"video": video.name, "track_id": tid, "feats": feats, "frames": rec["frames"],
                       "det_cls": sc.argmax(1), "conf": sc.max(1).values})
    return tracks, n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", default="data/sequences")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--min-len", type=int, default=5)
    ap.add_argument("--every", type=int, default=1, help="use every n-th frame")
    ap.add_argument("--val-videos", nargs="*", default=[])
    a = ap.parse_args()
    det = SignDetector(a.weights)
    gt = {}
    if a.labels:
        names = {v: k for k, v in det.names.items()}
        for r in csv.DictReader(open(a.labels)):
            gt[(r["video"], int(r["track_id"]))] = names[r["class_name"]]

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    train, val = [], []
    for v in map(Path, a.videos):
        tracks, n_total = extract(det, v, a.conf, a.min_len, a.every)
        for t in tracks:
            key = (t["video"], t["track_id"])
            if key in gt:
                t["label"], t["pseudo"] = gt[key], False
            elif gt:
                continue                                    # labels given but this track is unlabeled -> skip
            else:
                votes = torch.zeros(det.nc).index_add_(0, t["det_cls"], t["conf"])
                t["label"], t["pseudo"] = int(votes.argmax()), True
        with open(out / f"{v.stem}_tracks.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["video", "track_id", "first_frame", "last_frame", "n", "label_class", "pseudo"])
            for t in tracks:
                if "label" in t:
                    w.writerow([t["video"], t["track_id"], t["frames"][0], t["frames"][-1], len(t["frames"]),
                                det.names[t["label"]], t["pseudo"]])
        tracks = [t for t in tracks if "label" in t]
        if v.name in a.val_videos:
            val += tracks
        elif len(a.videos) == 1 and not a.val_videos:            # temporal split of the single video
            lo, hi = 0.7 * n_total, 0.8 * n_total
            train += [t for t in tracks if t["frames"][-1] < lo]
            val += [t for t in tracks if t["frames"][0] > hi]
        else:
            train += tracks
        print(f"{v.name}: {len(tracks)} labelled tracks")
    meta = {"emb_dim": det.emb_dim, "nc": det.nc, "names": det.names, "detector": det.weights}
    torch.save({"seqs": train, **meta}, out / "train.pt"); torch.save({"seqs": val, **meta}, out / "val.pt")
    print(f"train {len(train)} | val {len(val)} sequences -> {out}")


if __name__ == "__main__":
    main()
