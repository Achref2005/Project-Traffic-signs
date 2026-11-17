"""Train / resume the detector.   --variant baseline : stock YOLOv8n
                                   --variant transformer : YOLOv8n + Transformer blocks at P5 (SPPF) and P4 (neck)
Pretrained yolov8n.pt weights are transferred in both cases. Same hyper-parameters => fair comparison."""
import argparse

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

import traffic_signs  # noqa: F401  (registers the Transformer block)
from traffic_signs.data import ROOT, resolved_data_yaml
from traffic_signs.model_builder import build_model


class TransformerTrainer(DetectionTrainer):
    """Builds the YOLOv8+Transformer graph and transfers yolov8n.pt weights (layer indices remapped)."""
    pretrained = "yolov8n.pt"
    heads = 4

    def get_model(self, cfg=None, weights=None, verbose=True):
        return build_model(self.data["nc"], pretrained=self.pretrained, heads=self.heads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "transformer"], default="transformer")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=75)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of the train split (quick runs)")
    ap.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--fliplr", type=float, default=0.0, help="0: flips corrupt left/right + digit semantics")
    ap.add_argument("--name", default=None)
    ap.add_argument("--resume", default=None, help="path to last.pt of an interrupted run")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    name = a.name or f"yolov8n_{a.variant}"
    trainer = TransformerTrainer if a.variant == "transformer" else None
    TransformerTrainer.pretrained = a.weights
    if a.resume:
        YOLO(a.resume).train(resume=True, **({"trainer": trainer} if trainer else {}))
        return
    kw = dict(data=resolved_data_yaml(), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
              workers=a.workers, fraction=a.fraction, fliplr=a.fliplr, seed=a.seed, cos_lr=True, patience=25,
              close_mosaic=15, project=str(ROOT / "runs" / "train"), name=name, exist_ok=True, save_period=10, plots=False)
    if trainer:
        YOLO(a.weights).train(trainer=trainer, **kw)
    else:
        YOLO(a.weights).train(**kw)


if __name__ == "__main__":
    main()
