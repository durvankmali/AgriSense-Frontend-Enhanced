"""AgriSense inference API and optional same-origin exhibition frontend."""

import base64
import io
import logging
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from app.schemas import PredictionResponse
from app.assistant import AssistantRequest, AssistantResponse, explain
from src.inference.pipeline import AgriSensePipeline

log = logging.getLogger("agrisense")
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
PROJECT_ROOT = Path(__file__).resolve().parents[1]

app = FastAPI(
    title="AgriSense API",
    description="Crop disease classification, localization and Grad-CAM explainability.",
    version="1.3.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500", "http://localhost:5500",
        "http://127.0.0.1:8000", "http://localhost:8000",
        "https://agrisense-rcwy.onrender.com",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Load once; requests run serially to avoid duplicated inference graphs/memory spikes.
pipeline = AgriSensePipeline()
inference_lock = threading.Lock()


def array_to_base64_png(array):
    array = (array * 255).clip(0, 255).astype("uint8")
    image = Image.fromarray(array, mode="L")
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


@app.get("/api")
def api_info():
    return {"message": "AgriSense API is running", "status": "ok", "version": app.version}


@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": True,
            "device": str(pipeline.device), "num_classes": len(pipeline.class_names)}


def predict_serial(image):
    with inference_lock:
        return pipeline.predict_from_image(image)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Please upload a JPEG or PNG image.")

    # Limit memory allocation before reading the entire request into RAM.
    payload = await file.read(MAX_FILE_SIZE + 1)
    await file.close()
    if len(payload) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")

    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise ValueError("Unsupported image format")
            image = ImageOps.exif_transpose(source)
            image.load()
            image = image.convert("RGB")
    except (UnidentifiedImageError, ValueError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Cannot read this image. Try a different JPG or PNG.")

    start = time.perf_counter()
    try:
        result = await run_in_threadpool(predict_serial, image)
        log.info("Prediction completed in %.2fs", time.perf_counter() - start)
        return {
            "disease": result["disease"],
            "confidence": result["confidence"],
            "class_index": result["class_index"],
            "uncertain": result["uncertain"],
            "mask_coverage": result["mask_coverage"],
            "segmentation_mask": array_to_base64_png(result["segmentation_mask"].squeeze()),
            "gradcam": array_to_base64_png(result["gradcam"]),
        }
    except Exception:
        log.exception("Prediction pipeline failed")
        raise HTTPException(status_code=500, detail="Analysis failed on the server. Please retry.")


@app.post("/api/assistant", response_model=AssistantResponse)
async def answer_visitor_question(payload: AssistantRequest):
    """Explain a *previous* model result without running inference again.

    No image or model weights are sent to the external language model.
    """
    if payload.disease not in pipeline.class_names:
        raise HTTPException(status_code=422, detail="Unknown AgriSense prediction label.")
    return await run_in_threadpool(explain, payload)


# Mount last, so /predict and /health always take precedence.
frontend_dir = PROJECT_ROOT / "frontend"
if frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
