# Traffic-sign detection — YOLOv8 + Transformer (+ ByteTrack + LSTM for video)

```
image : YOLOv8 backbone/neck + Transformer(P5,P4) -> Detect head -> boxes / classes / scores
video : ... -> Detect -> ByteTrack ids -> P4 ROI embedding per track -> LSTM (per-track state) -> refined class
```
- `traffic_signs/attention.py` multi-head self-attention encoder, pre-norm + residual, zero-init gates (identity at init => pretrained YOLO features preserved).
- `traffic_signs/model_builder.py` inserts the block after SPPF (P5) and the neck P4 output of stock YOLOv8, remaps and transfers `yolov8n.pt` weights.
- `traffic_signs/temporal.py` LSTM residual refiner + per-stream `TrackMemory` (state per track id, pruned when a track vanishes).
- `traffic_signs/pipeline.py` `SignDetector` (image mode) and `VideoSession` (one per video stream).

## Install / run
```
pip install -r requirements.txt
python train_detector.py --variant transformer --epochs 75        # or --variant baseline ; --resume runs/train/<name>/weights/last.pt
python prepare_sequences.py --weights <best.pt> --videos database/video.mp4   # LSTM data (needs VIDEOS, see below)
python train_temporal.py                                            # -> weights/temporal_lstm.pt
python evaluate.py --models baseline=<a.pt> transformer=<b.pt> --video database/video.mp4
python predict.py --weights <best.pt> --source database/video.mp4
python server/infer_api.py                                          # port 8001
cd ui && npm install && npm run dev                                 # http://localhost:3000
```
Weights are resolved by `server/infer_api.py`: `weights/traffic_signs_transformer.pt` -> `runs/train/*transformer*/weights/best.pt` -> older baseline runs (override: `DETECTOR_WEIGHTS`, `TEMPORAL_WEIGHTS`).

## LSTM data
The image dataset has no temporal information. The LSTM needs videos with consistent track ids and per-track labels
(`--labels video,track_id,class_name` CSV, ids from the generated `*_tracks.csv`). Without labels `prepare_sequences.py`
uses detector majority-vote pseudo-labels (weak supervision). Until `weights/temporal_lstm.pt` exists the LSTM is an exact pass-through of the detector.
