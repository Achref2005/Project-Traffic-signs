"""LSTM temporal refinement of per-track traffic-sign classification.

Feature layout per tracked detection (see build_features):  [ROI embedding (E) | class scores (nc) | geometry (5)]
The LSTM predicts a *residual* on top of the detector's own log-scores, with a zero-initialised head, so an
untrained module is an exact pass-through of the detector and training can only add a temporal correction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

GEOM_DIM = 5   # cx, cy, w, h (normalised by image size) and detector confidence


def build_features(emb, scores, boxes_xyxy, conf, img_w, img_h):
    """emb (N,E), scores (N,nc), boxes (N,4) in pixels, conf (N,) -> (N, E+nc+5) float tensor."""
    x1, y1, x2, y2 = boxes_xyxy.unbind(1)
    geom = torch.stack([(x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h, conf], 1)
    return torch.cat([emb, scores, geom], 1)


class TemporalClassifier(nn.Module):
    def __init__(self, emb_dim, n_classes, hidden=128, layers=1, dropout=0.1):
        super().__init__()
        self.emb_dim, self.nc, self.hidden, self.layers = emb_dim, n_classes, hidden, layers
        self.in_dim = emb_dim + n_classes + GEOM_DIM
        self.proj = nn.Sequential(nn.LayerNorm(self.in_dim), nn.Linear(self.in_dim, hidden), nn.GELU())
        self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_classes)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def base_logp(self, x):
        s = x[..., self.emb_dim:self.emb_dim + self.nc].clamp_min(0) + 1e-6
        return (s / s.sum(-1, keepdim=True)).log()

    def forward(self, x, lengths=None, state=None):
        """x (B,T,D) padded; lengths (B,) true lengths (variable-length sequences). Returns refined logits (B,T,nc), state."""
        h = self.proj(x)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, state = self.lstm(packed, state)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        else:
            out, state = self.lstm(h, state)
        return self.base_logp(x) + self.head(out), state


class TrackMemory:
    """Per-video-stream LSTM state for every live track id. One instance per stream => no cross-stream leakage."""

    def __init__(self, model, device, max_age=30):
        self.model, self.device, self.max_age = model, device, max_age
        self.reset()

    def reset(self):
        self.state = {}      # tid -> (h, c) each (layers, 1, H)
        self.last_seen = {}  # tid -> frame idx
        self.steps = {}      # tid -> number of LSTM steps taken

    @torch.no_grad()
    def step(self, frame_idx, track_ids, feats):
        """One LSTM step for all tracks visible in this frame. feats (N,D) -> refined probs (N,nc)."""
        if not len(track_ids):
            self.prune(frame_idx)
            return feats.new_zeros((0, self.model.nc))
        L, H = self.model.layers, self.model.hidden
        z = feats.new_zeros((L, 1, H))
        hs, cs = zip(*[self.state.get(t, (z, z)) for t in track_ids])
        logits, (h, c) = self.model(feats[:, None, :], state=(torch.cat(hs, 1), torch.cat(cs, 1)))
        for i, t in enumerate(track_ids):
            self.state[t] = (h[:, i:i + 1].clone(), c[:, i:i + 1].clone())
            self.last_seen[t] = frame_idx
            self.steps[t] = self.steps.get(t, 0) + 1
        self.prune(frame_idx)
        return F.softmax(logits[:, 0], -1)

    def prune(self, frame_idx):
        """Drop state of tracks that vanished (not seen for max_age frames)."""
        for t in [t for t, f in self.last_seen.items() if frame_idx - f > self.max_age]:
            self.state.pop(t); self.last_seen.pop(t); self.steps.pop(t, None)


def save_temporal(model, path, extra=None):
    torch.save({"state_dict": model.state_dict(), "emb_dim": model.emb_dim, "nc": model.nc,
                "hidden": model.hidden, "layers": model.layers, **(extra or {})}, path)


def load_temporal(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = TemporalClassifier(ck["emb_dim"], ck["nc"], ck["hidden"], ck["layers"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck
