"""Build YOLOv8 + Transformer models from the stock YOLOv8 config and transfer pretrained weights."""
import copy
import math
from pathlib import Path

import torch

from . import register_modules  # noqa: F401  (ensures registration)

register_modules()

# Original yolov8 layer indices after which a Transformer block is inserted:
# 9 = SPPF (P5/32 backbone output), 18 = neck P4/16 output (feeds Detect).
DEFAULT_INSERT_AFTER = (9, 18)


def _scaled_channels(c, cfg):
    d = cfg["scales"][cfg.get("scale", "n")] if "scales" in cfg else (1, 1, 1024)
    width, max_ch = d[1], d[2]
    return int(math.ceil(min(c, max_ch) * width / 8) * 8)


def build_cfg(base="yolov8n.yaml", insert_after=DEFAULT_INSERT_AFTER, heads=4, mlp_ratio=2.0):
    """Return (cfg_dict, new2old) where new2old maps new layer idx -> original idx (None = inserted)."""
    from ultralytics.nn.tasks import yaml_model_load
    cfg = copy.deepcopy(yaml_model_load(base))
    layers = cfg["backbone"] + cfg["head"]
    n_backbone = len(cfg["backbone"])
    insert_after = set(insert_after)
    # output channels of the layers a block may follow (Conv / C2f / SPPF, args[0] is the nominal width)
    out_ch = {i: (_scaled_channels(l[3][0], cfg) if l[2] in ("Conv", "C2f", "SPPF") else None)
              for i, l in enumerate(layers)}

    new_layers, new2old, old2ref = [], [], {}
    for i, (f, n, m, a) in enumerate(layers):
        old2ref[i] = len(new_layers)
        new_layers.append([f, n, m, a]); new2old.append(i)
        if i in insert_after:
            c = out_ch[i]
            assert c and c % heads == 0, f"cannot place block after layer {i} (c={c})"
            new_layers.append([-1, 1, "TransformerEncoder2D", [c, heads, mlp_ratio]]); new2old.append(None)
            old2ref[i] = len(new_layers) - 1       # later absolute references use the attended features

    def remap(f):
        if isinstance(f, int):
            return f if f < 0 else old2ref[f]
        return [x if x < 0 else old2ref[x] for x in f]

    for j, l in enumerate(new_layers):
        if new2old[j] is not None:
            l[0] = remap(l[0])
    cfg["backbone"], cfg["head"] = new_layers[:old2ref_idx(new2old, n_backbone)], new_layers[old2ref_idx(new2old, n_backbone):]
    cfg["_new2old"] = new2old
    return cfg, new2old


def old2ref_idx(new2old, n_backbone):
    """index in the new list where the original head starts (after any block attached to the last backbone layer)."""
    last_bb = n_backbone - 1
    k = new2old.index(last_bb)
    return k + 2 if (k + 1 < len(new2old) and new2old[k + 1] is None) else k + 1


def transfer_pretrained(model, weights, new2old):
    """Copy matching tensors from a stock YOLOv8 checkpoint into `model`, remapping layer indices."""
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    src = (ckpt.get("ema") or ckpt["model"]).float().state_dict()
    old2new = {o: n for n, o in enumerate(new2old) if o is not None}
    dst, loaded = model.state_dict(), {}
    for k, v in src.items():
        parts = k.split(".")
        if parts[0] == "model" and parts[1].isdigit() and int(parts[1]) in old2new:
            nk = ".".join(["model", str(old2new[int(parts[1])])] + parts[2:])
            if nk in dst and dst[nk].shape == v.shape:
                loaded[nk] = v
    model.load_state_dict(loaded, strict=False)
    return len(loaded), len(dst)


def build_model(nc, base="yolov8n.yaml", pretrained="yolov8n.pt", insert_after=DEFAULT_INSERT_AFTER,
                heads=4, verbose=False):
    from ultralytics.nn.tasks import DetectionModel
    cfg, new2old = build_cfg(base, insert_after, heads)
    model = DetectionModel(cfg, nc=nc, ch=3, verbose=verbose)
    if pretrained and Path(pretrained).exists():
        n, total = transfer_pretrained(model, pretrained, new2old)
        print(f"transferred {n}/{total} tensors from {pretrained}")
    return model
