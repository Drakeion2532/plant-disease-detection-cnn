import numpy as np
import torch

from plantdoc.model import PlantCNN
from plantdoc.train import stratified_split, augment, to_tensor
from plantdoc.advice import advice_for, pretty


def test_model_output_shape():
    m = PlantCNN(38, img_size=64)
    assert m(torch.zeros(2, 3, 64, 64)).shape == (2, 38)


def test_stratified_split_is_disjoint_and_balanced():
    y = np.repeat(np.arange(5), 100)
    tr, va, te = stratified_split(y)
    assert not (set(tr) & set(te)) and not (set(tr) & set(va))
    assert all((y[te] == c).sum() == 15 for c in range(5))


def test_augment_preserves_shape():
    x = to_tensor(np.random.randint(0, 255, (4, 64, 64, 3), dtype=np.uint8))
    assert augment(x).shape == x.shape


def test_advice_lookup():
    assert "URGENT" in advice_for("Tomato___Late_blight")
    assert "No disease" in advice_for("Apple___healthy")
    assert pretty("Corn_(maize)___Common_rust_") == ("Corn (maize)", "Common rust")


# ---------------------------------------------------------------- photo check (don't diagnose non-leaves)
import glob

import numpy as np
from PIL import Image

from plantdoc.predict import Predictor, diagnose, photo_check

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


def test_blank_and_non_leaf_images_are_not_diagnosed():
    pred = Predictor()
    assert diagnose(pred, Image.new("RGB", (256, 256), (40, 140, 50)))["status"] == "blank"   # was "healthy corn 99%"
    assert diagnose(pred, Image.new("RGB", (256, 256), (128, 128, 128)))["status"] == "blank"
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8))
    assert diagnose(pred, noise)["status"] in ("no_leaf", "uncertain")
    # a screenshot of the app (which itself shows a small leaf photo) must not get a diagnosis
    screenshot = diagnose(pred, Image.open(ROOT / "docs" / "app_late_blight.png"))
    assert screenshot["status"] in ("no_leaf", "uncertain") and screenshot["message"]


def test_every_sample_leaf_passes_the_photo_check():
    files = sorted(glob.glob(str(ROOT / "data" / "sample" / "*" / "*")))
    assert len(files) == 76
    assert all(photo_check(Image.open(f))[0] is None for f in files)


def test_web_app_returns_status_and_message():
    from fastapi.testclient import TestClient
    from app.app import app
    c = TestClient(app)
    leaf = sorted(glob.glob(str(ROOT / "data" / "sample" / "Tomato___Late_blight" / "*")))[0]
    with open(leaf, "rb") as f:
        r = c.post("/predict", files={"file": ("leaf.jpg", f, "image/jpeg")}).json()
    assert r["status"] == "ok" and r["predictions"][0]["disease"] == "Late blight"
    with open(ROOT / "docs" / "app_late_blight.png", "rb") as f:
        assert c.post("/predict", files={"file": ("s.png", f, "image/png")}).json()["status"] != "ok"
