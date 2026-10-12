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
