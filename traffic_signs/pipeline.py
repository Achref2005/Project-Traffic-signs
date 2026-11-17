"""Inference pipelines.  Image mode: detector only.  Video mode: detector -> ByteTrack -> LSTM refinement."""
from pathlib import Path

import numpy as np
import torch
import torchvision
from ultralytics.data.augment import LetterBox
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.ops import scale_boxes

from .temporal import TemporalClassifier, TrackMemory, build_features, load_temporal


class SignDetector:
    """YOLOv8 (optionally + Transformer blocks). Returns boxes, class scores and P4 ROI embeddings."""

    def __init__(self, weights, device=None, imgsz=416):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model, _ = load_checkpoint(str(weights), device=self.device)
        self.model = model.float().eval()
        self.names = self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))
        self.nc = len(self.names)
        self.imgsz = imgsz
        self.weights = str(weights)
        self.has_transformer = any(type(m).__name__ == "TransformerEncoder2D" for m in self.model.modules())
        detect = self.model.model[-1]
        self._feat = None
        self.model.model[detect.f[1]].register_forward_hook(lambda m, i, o: setattr(self, "_feat", o))  # P4 neck
        self.emb_dim = self.model.model[detect.f[1]].cv2.conv.out_channels if hasattr(self.model.model[detect.f[1]], "cv2") \
            else self._probe_emb_dim()
        self._lb = LetterBox(new_shape=(imgsz, imgsz), auto=True, stride=32)

    def _probe_emb_dim(self):
        with torch.no_grad():
            self.model(torch.zeros(1, 3, 64, 64, device=self.device))
        return self._feat.shape[1]

    @torch.no_grad()
    def detect(self, img_bgr, conf=0.3, iou=0.5, imgsz=None, with_embeddings=False, max_det=100):
        if imgsz and imgsz != self.imgsz:
            self.imgsz = int(imgsz); self._lb = LetterBox(new_shape=(self.imgsz, self.imgsz), auto=True, stride=32)
        h0, w0 = img_bgr.shape[:2]
        lb = self._lb(image=img_bgr)
        ph, pw = lb.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(lb[..., ::-1].transpose(2, 0, 1))).to(self.device).float()[None] / 255
        out = self.model(x)
        pred = (out[0] if isinstance(out, (list, tuple)) else out)[0].T        # (A, 4+nc)
        xywh, scores = pred[:, :4], pred[:, 4:]
        mx, cls = scores.max(1)
        keep = mx > conf
        xywh, scores, mx, cls = xywh[keep], scores[keep], mx[keep], cls[keep]
        xyxy = torch.cat([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
        idx = torchvision.ops.batched_nms(xyxy, mx, cls, iou)[:max_det]
        xyxy, scores, mx, cls = xyxy[idx], scores[idx], mx[idx], cls[idx]
        emb = None
        if with_embeddings:
            fm = self._feat
            emb = (torchvision.ops.roi_align(fm, [xyxy], 2, spatial_scale=fm.shape[-1] / pw, aligned=True).mean((2, 3))
                   if len(xyxy) else fm.new_zeros((0, fm.shape[1])))
        boxes = scale_boxes((ph, pw), xyxy.clone(), (h0, w0)) if len(xyxy) else xyxy
        return {"boxes": boxes, "conf": mx, "cls": cls, "scores": scores, "emb": emb, "size": (w0, h0)}


class _Dets:
    """Minimal NumPy detection container accepted by ultralytics BYTETracker (conf/cls/xywh + bool indexing)."""

    def __init__(self, conf, cls, xywh):
        self.conf, self.cls, self.xywh = conf, cls, xywh

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, i):
        return _Dets(self.conf[i], self.cls[i], self.xywh[i])


def _tracker_args(track_buffer=None):
    import ultralytics
    cfg = YAML.load(Path(ultralytics.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml")
    if track_buffer:
        cfg["track_buffer"] = track_buffer
    return IterableSimpleNamespace(**cfg)


class VideoSession:
    """State for ONE video stream: its own ByteTrack instance and LSTM states (create a new one per stream)."""

    def __init__(self, detector, temporal=None, collect=False):
        from ultralytics.trackers.byte_tracker import BYTETracker
        self.det, self.temporal = detector, temporal
        self.args = _tracker_args()
        self.tracker = BYTETracker(self.args)
        self.memory = TrackMemory(temporal, detector.device, max_age=self.args.track_buffer) if temporal else None
        self.collect = collect          # record per-track feature sequences (LSTM data preparation)
        self.records = {}
        self.frame = 0

    def reset(self):
        self.tracker.reset()
        if self.memory:
            self.memory.reset()
        self.frame = 0
        self.records = {}

    @torch.no_grad()
    def update(self, img_bgr, conf=0.3, imgsz=None):
        d = self.det.detect(img_bgr, conf=conf, imgsz=imgsz, with_embeddings=self.memory is not None or self.collect)
        self.frame += 1
        boxes = d["boxes"].cpu().numpy()
        xywh = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2,
                         boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], 1) if len(boxes) else np.zeros((0, 4))
        tr = self.tracker.update(_Dets(d["conf"].cpu().numpy(), d["cls"].cpu().numpy().astype(np.float32), xywh), img_bgr)
        rows = [r for r in tr if int(r[7]) < len(boxes)]
        ids = [int(r[4]) for r in rows]
        di = [int(r[7]) for r in rows]
        probs = None
        feats = None
        if rows and (self.memory is not None or self.collect):
            sel = torch.as_tensor(di, device=d["boxes"].device)
            feats = build_features(d["emb"][sel], d["scores"][sel], d["boxes"][sel], d["conf"][sel], *d["size"])
        if self.collect and feats is not None:
            for tid, ft in zip(ids, feats):
                r = self.records.setdefault(tid, {"feats": [], "frames": []})
                r["feats"].append(ft.detach().cpu()); r["frames"].append(self.frame)
        if self.memory is not None and feats is not None:
            probs = self.memory.step(self.frame, ids, feats).cpu().numpy()
        elif self.memory is not None:
            self.memory.step(self.frame, [], d["scores"].new_zeros((0, 1)))
        out = []
        for k, (tid, i) in enumerate(zip(ids, di)):
            c = int(d["cls"][i]); item = {
                "track_id": tid, "box": [round(float(v), 1) for v in boxes[i]],
                "yolo_class": self.det.names[c], "yolo_conf": round(float(d["conf"][i]), 3)}
            if probs is not None:
                rc = int(probs[k].argmax())
                item.update(refined_class=self.det.names[rc], refined_conf=round(float(probs[k][rc]), 3),
                            track_age=self.memory.steps[tid])
            out.append(item)
        return out
