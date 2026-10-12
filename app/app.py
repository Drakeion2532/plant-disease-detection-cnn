"""Web app: upload a leaf photo, get the disease + what to do.

    uvicorn app.app:app --port 8004   ->  http://127.0.0.1:8004
"""
import io
import sys
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plantdoc.predict import Predictor  # noqa: E402

app = FastAPI(title="Plant Disease Detector")
predictor = Predictor()
PAGE = (Path(__file__).parent / "index.html").read_text()


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "image too large (max 10 MB)")
    try:
        img = Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise HTTPException(415, "not an image")
    return {"predictions": predictor(img)}
