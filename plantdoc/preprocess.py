"""Resize every PlantVillage image once and cache as a uint8 numpy array so
training does not re-decode 54k JPEGs every epoch."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def _load(args):
    path, size = args
    im = Image.open(path)
    im.draft("RGB", (size * 2, size * 2))  # fast JPEG downscale
    return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)


def build_cache(src: Path, out: Path, size: int = 96, workers: int = 2):
    classes = sorted(p.name for p in src.iterdir() if p.is_dir())
    files, labels = [], []
    for i, c in enumerate(classes):
        imgs = sorted(f for f in (src / c).iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        files += imgs
        labels += [i] * len(imgs)
    X = np.zeros((len(files), size, size, 3), np.uint8)
    with ProcessPoolExecutor(workers) as ex:
        for k, arr in enumerate(ex.map(_load, [(f, size) for f in files], chunksize=256)):
            X[k] = arr
            if k % 5000 == 0:
                print(f"{k}/{len(files)}")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"X_{size}.npy", X)
    np.save(out / "y.npy", np.array(labels, np.int16))
    (out / "classes.json").write_text(json.dumps(classes, indent=1))
    print(f"cached {len(files)} images, {len(classes)} classes -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/plantvillage/raw/color")
    ap.add_argument("--out", default="data/cache")
    ap.add_argument("--size", type=int, default=96)
    a = ap.parse_args()
    build_cache(Path(a.src), Path(a.out), a.size)
