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


# ---------------------------------------------------------------- photo check
# The CNN only knows PlantVillage leaves. Shown anything else (a screenshot, a chart, a plain
# green wall) it still picks one of its 38 classes, sometimes with high confidence: a graph
# drawing came out as "Tomato late blight, 89%, destroy infected plants". So before giving
# advice we check that the photo plausibly shows a leaf, and say so when we are not sure.
# Thresholds were set on the 76 sample leaves (all accepted) and 24 non-leaf images
# (screenshots, charts, noise, plain colours; all rejected).
MIN_CONTRAST = 6.0          # grey-level std on a 128x128 copy; below this the image is blank
MIN_PLANT_FRACTION = 0.05   # share of pixels with leaf-like colour (green / yellow / brown)
MIN_CONFIDENCE = 0.50       # below this the top class is not trusted

MESSAGES = {
    "blank": "The photo looks blank or completely out of focus. Retake it in daylight with the leaf filling most of the frame.",
    "no_leaf": "No leaf found in this photo. Photograph a single leaf on a plain background.",
    "uncertain": "Not sure about this one: the photo may be a crop this model doesn't cover, or too dark or blurry. "
                 "Retake it, and if the problem continues, show the plant to a local agriculture officer (Krishi Vigyan Kendra).",
}


def photo_check(img: Image.Image) -> tuple[str | None, dict]:
    small = img.convert("RGB").resize((128, 128))
    hsv = np.asarray(small.convert("HSV")).astype(float)
    h, s, v = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    plant = float(((h >= 15) & (h <= 190) & (s >= 0.08) & (v >= 0.12)).mean())
    contrast = float(np.asarray(small.convert("L")).astype(float).std())
    stats = {"plant_fraction": round(plant, 3), "contrast": round(contrast, 1)}
    if contrast < MIN_CONTRAST:
        return "blank", stats
    if plant < MIN_PLANT_FRACTION:
        return "no_leaf", stats
    return None, stats


def diagnose(predictor: "Predictor", img: Image.Image) -> dict:
    """Photo check + prediction. status is "ok", "blank", "no_leaf" or "uncertain"."""
    problem, stats = photo_check(img)
    if problem:
        return {"status": problem, "message": MESSAGES[problem], "predictions": [], "photo": stats}
    preds = predictor(img)
    if preds[0]["confidence"] < MIN_CONFIDENCE:
        return {"status": "uncertain", "message": MESSAGES["uncertain"], "predictions": preds, "photo": stats}
    return {"status": "ok", "message": preds[0]["advice"], "predictions": preds, "photo": stats}


if __name__ == "__main__":
    pred = Predictor()
    for f in sys.argv[1:]:
        d = diagnose(pred, Image.open(f))
        if d["status"] == "ok":
            r = d["predictions"][0]
            print(f"{f}: {r['plant']} – {r['disease']} ({r['confidence']:.1%})\n  → {r['advice']}")
        else:
            print(f"{f}: {d['status'].replace('_', ' ').upper()}\n  → {d['message']}")
