"""Train + evaluate the PlantVillage CNN.

Split: 70% train / 15% validation / 15% test (stratified, seed 42) – the
paper reports test accuracy on a 15% hold-out.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import PlantCNN

MEAN = torch.tensor([0.466, 0.489, 0.411]).view(1, 3, 1, 1)
STD = torch.tensor([0.177, 0.152, 0.193]).view(1, 3, 1, 1)


def stratified_split(y, seed=42, val=0.15, test=0.15):
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_te, n_va = int(round(len(idx) * test)), int(round(len(idx) * val))
        te += list(idx[:n_te]); va += list(idx[n_te:n_te + n_va]); tr += list(idx[n_te + n_va:])
    return np.array(tr), np.array(va), np.array(te)


def to_tensor(xb: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(np.array(xb, copy=True)).permute(0, 3, 1, 2).float().div_(255)
    return ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)


def augment(x: torch.Tensor) -> torch.Tensor:
    """Augmentation from the paper: flips, rotations, brightness/contrast."""
    if torch.rand(1) < 0.5:
        x = x.flip(3)
    if torch.rand(1) < 0.5:
        x = x.flip(2)
    k = int(torch.randint(0, 4, (1,)))
    x = torch.rot90(x, k, (2, 3))
    b = x.shape[0]
    x = x * (1 + 0.2 * (torch.rand(b, 1, 1, 1) - 0.5)) + 0.2 * (torch.rand(b, 1, 1, 1) - 0.5)
    return x.contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def predict(model, X, bs=512):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(F.softmax(model(to_tensor(X[i:i + bs])), 1))
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=int, help="debug: use only N images")
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    cache, out = Path(a.cache), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    X = np.load(cache / f"X_{a.size}.npy", mmap_mode="r")
    y = np.load(cache / "y.npy").astype(np.int64)
    classes = json.loads((cache / "classes.json").read_text())
    if a.limit:
        sel = np.random.default_rng(0).choice(len(y), a.limit, replace=False)
        X, y = X[sel], y[sel]
    X = np.ascontiguousarray(X)
    tr, va, te = stratified_split(y)
    np.save(out / "test_indices.npy", te)
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}  classes {len(classes)}")

    model = PlantCNN(len(classes), img_size=a.size).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.2)
    hist, best = [], 0.0
    for ep in range(a.epochs):
        model.train()
        perm = np.random.default_rng(ep).permutation(tr)
        t0, tl, tc, n = time.time(), 0.0, 0, 0
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            idx = np.sort(perm[i:i + a.bs])
            xb, yb = augment(to_tensor(X[idx])), torch.from_numpy(y[idx])
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(idx); tc += (logits.argmax(1) == yb).sum().item(); n += len(idx)
        pv = predict(model, X[va])
        vacc = float((pv.argmax(1) == y[va]).mean())
        vloss = float(-np.log(pv[np.arange(len(va)), y[va]] + 1e-9).mean())
        hist.append({"epoch": ep + 1, "train_loss": tl / n, "train_acc": tc / n, "val_loss": vloss,
                     "val_acc": vacc, "secs": round(time.time() - t0, 1)})
        print(json.dumps(hist[-1]), flush=True)
        if vacc > best:
            best = vacc
            torch.save({"state_dict": model.state_dict(), "classes": classes, "img_size": a.size}, out / "plant_cnn.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=1))

    ck = torch.load(out / "plant_cnn.pt")
    model.load_state_dict(ck["state_dict"])
    pt = predict(model, X[te])
    np.save(out / "test_probs.npy", pt.astype(np.float16))
    acc = float((pt.argmax(1) == y[te]).mean())
    print(f"TEST accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
