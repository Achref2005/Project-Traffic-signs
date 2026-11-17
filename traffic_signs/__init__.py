"""Traffic-sign detection: YOLOv8 + Transformer encoder (+ tracker + LSTM for video)."""
from .attention import TransformerEncoder2D, register_modules

register_modules()
