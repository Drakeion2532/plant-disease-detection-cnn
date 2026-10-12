"""Predict the disease in one or more leaf photos.

    python -m plantdoc.predict path/to/leaf.jpg [more.jpg ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .model import PlantCNN
from .train import to_tensor
from .advice import advice_for, pretty

ROOT = Path(__file__).resolve().parents[1]


class Predictor:
    def __init__(self, weights=ROOT / "results" / "plant_cnn.pt"):
        ck = torch.load(weights, map_location="cpu")
        self.classes, self.size = ck["classes"], ck["img_size"]
        self.model = PlantCNN(len(self.classes), img_size=self.size)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def __call__(self, img: Image.Image, top=3):
        arr = np.asarray(img.convert("RGB").resize((self.size, self.size), Image.BILINEAR), np.uint8)[None]
        # test-time augmentation: average over the 4 rotations
        x = to_tensor(arr)
        p = sum(torch.softmax(self.model(torch.rot90(x, k, (2, 3)).contiguous()), 1) for k in range(4)) / 4
        p = p[0].numpy()
        idx = np.argsort(-p)[:top]
        out = []
        for i in idx:
            plant, disease = pretty(self.classes[i])
            out.append({"class": self.classes[i], "plant": plant, "disease": disease,
                        "confidence": round(float(p[i]), 4), "advice": advice_for(self.classes[i])})
        return out


if __name__ == "__main__":
    pred = Predictor()
    for f in sys.argv[1:]:
        r = pred(Image.open(f))[0]
        print(f"{f}: {r['plant']} – {r['disease']} ({r['confidence']:.1%})\n  → {r['advice']}")
