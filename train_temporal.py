"""Train the LSTM temporal refiner on sequences from prepare_sequences.py (variable-length, per-step loss)."""
import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from traffic_signs.temporal import TemporalClassifier, save_temporal


def batchify(seqs, train, device):
    xs, ys, lens = [], [], []
    for s in seqs:
        x = s["feats"]
        if train:                                           # random sub-sequence + dropped frames (missing detections)
            T = len(x); a = random.randint(0, max(0, T - 3)); b = random.randint(a + 3, T) if T - a >= 3 else T
            keep = [i for i in range(a, b) if random.random() > 0.15] or list(range(a, b))
            x = x[keep]
        xs.append(x); lens.append(len(x)); ys.append(s["label"])
    X = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True).to(device)
    return X, torch.tensor(ys, device=device), torch.tensor(lens)


def run_eval(model, seqs, nc, device):
    model.eval(); n = c_ref = c_det = flips_ref = flips_det = steps = 0
    with torch.no_grad():
        for s in seqs:
            x = s["feats"][None].to(device)
            ref = model(x)[0][0].argmax(-1).cpu(); det = s["det_cls"]
            c_ref += (ref == s["label"]).sum().item(); c_det += (det == s["label"]).sum().item(); n += len(ref)
            flips_ref += (ref[1:] != ref[:-1]).sum().item(); flips_det += (det[1:] != det[:-1]).sum().item(); steps += len(ref) - 1
    return {"acc_lstm": c_ref / max(n, 1), "acc_detector": c_det / max(n, 1),
            "flip_lstm": flips_ref / max(steps, 1), "flip_detector": flips_det / max(steps, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sequences")
    ap.add_argument("--out", default="weights/temporal_lstm.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    tr, va = torch.load(f"{a.data}/train.pt", weights_only=False), torch.load(f"{a.data}/val.pt", weights_only=False)
    assert tr["seqs"], "no training sequences - run prepare_sequences.py on videos first"
    model = TemporalClassifier(tr["emb_dim"], tr["nc"], a.hidden).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    best = float("-inf"); Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, a.epochs + 1):
        model.train(); random.shuffle(tr["seqs"]); tot = 0.0
        for i in range(0, len(tr["seqs"]), a.batch):
            X, y, L = batchify(tr["seqs"][i:i + a.batch], True, a.device)
            logits, _ = model(X, lengths=L)
            mask = torch.arange(X.shape[1], device=a.device)[None] < L.to(a.device)[:, None]
            loss = F.cross_entropy(logits[mask], y[:, None].expand(-1, X.shape[1])[mask])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item()
        m = run_eval(model, va["seqs"], tr["nc"], a.device) if va["seqs"] else {}
        score = m.get("acc_lstm", -tot)
        if score >= best:
            best = score; save_temporal(model, a.out, {"val": m, "pseudo_labels": any(s.get("pseudo") for s in tr["seqs"])})
        if ep % 5 == 0 or ep == 1:
            print(f"ep {ep:3d} loss {tot:.3f} val {({k: round(v, 3) for k, v in m.items()})}")
    print("saved best ->", a.out)


if __name__ == "__main__":
    main()
