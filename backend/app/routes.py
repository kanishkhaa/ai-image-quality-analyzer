from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io

from app.analysis import analyze_image

router = APIRouter()


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/webp"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, PNG and WEBP images are supported."
        )

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image."
        )

    try:
        features = analyze_image(contents)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Unable to analyze image."
        )

    return {
        "filename": file.filename,
        "features": features
    }