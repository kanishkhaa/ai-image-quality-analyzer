import os
import sys


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# Imports
# ============================================================

import json
import shutil
import uuid

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Depends
)

from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy.orm import Session

from .database import (
    Base,
    engine,
    get_db
)

from .models import Analysis

from ml.image_analyzer import analyze_image


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="AI Image Quality & Defect Detection",
    description=(
        "AI-powered image quality assessment and "
        "defect detection system."
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# Database
# ============================================================

Base.metadata.create_all(
    bind=engine
)


# ============================================================
# Upload directory
# ============================================================

UPLOAD_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "uploads"
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# ============================================================
# Allowed formats
# ============================================================

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


# ============================================================
# Health check
# ============================================================

@app.get("/api/health")
def health_check():

    return {
        "status": "healthy",
        "service": "AI Image Quality Analyzer"
    }


# ============================================================
# Analyze image
# ============================================================

@app.post("/api/analyze")
async def analyze_uploaded_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )


    # --------------------------------------------------------
    # Validate extension
    # --------------------------------------------------------

    extension = os.path.splitext(
        file.filename
    )[1].lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Allowed formats: JPG, JPEG, PNG, WEBP, BMP."
            )
        )


    # --------------------------------------------------------
    # Generate safe filename
    # --------------------------------------------------------

    safe_filename = (
        uuid.uuid4().hex +
        extension
    )

    image_path = os.path.join(
        UPLOAD_DIR,
        safe_filename
    )


    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    try:

        with open(
            image_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to save image: {str(e)}"
        )


    # --------------------------------------------------------
    # Run AI analysis
    # --------------------------------------------------------

    try:

        result = analyze_image(
            image_path
        )

    except Exception as e:

        if os.path.exists(image_path):
            os.remove(image_path)

        raise HTTPException(
            status_code=500,
            detail=f"Image analysis failed: {str(e)}"
        )


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    try:

        analysis = Analysis(

            filename=file.filename,

            quality_score=float(
                result["quality_score"]
            ),

            quality_label=result[
                "quality_label"
            ],

            issues=json.dumps(
                result["issues"]
            ),

            statistics=json.dumps(
                result["statistics"]
            ),

            explanation=json.dumps(
                result["explanation"]
            )
        )

        db.add(analysis)

        db.commit()

        db.refresh(analysis)

    except Exception as e:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )


    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return {

        "id": analysis.id,

        "filename": analysis.filename,

        "quality_score": analysis.quality_score,

        "quality_label": analysis.quality_label,

        "issues": result["issues"],

        "statistics": result["statistics"],

        "explanation": result["explanation"],

        "created_at": analysis.created_at
    }


# ============================================================
# Analysis history
# ============================================================

@app.get("/api/analyses")
def get_analyses(
    db: Session = Depends(get_db)
):

    analyses = (
        db.query(Analysis)
        .order_by(
            Analysis.created_at.desc()
        )
        .all()
    )

    results = []

    for analysis in analyses:

        results.append({

            "id": analysis.id,

            "filename": analysis.filename,

            "quality_score": analysis.quality_score,

            "quality_label": analysis.quality_label,

            "issues": json.loads(
                analysis.issues
            ),

            "statistics": json.loads(
                analysis.statistics
            ),

            "explanation": json.loads(
                analysis.explanation
            ),

            "created_at": analysis.created_at
        })

    return results


# ============================================================
# Single analysis
# ============================================================

@app.get("/api/analyses/{analysis_id}")
def get_analysis(
    analysis_id: int,
    db: Session = Depends(get_db)
):

    analysis = (
        db.query(Analysis)
        .filter(
            Analysis.id == analysis_id
        )
        .first()
    )

    if analysis is None:

        raise HTTPException(
            status_code=404,
            detail="Analysis not found."
        )


    return {

        "id": analysis.id,

        "filename": analysis.filename,

        "quality_score": analysis.quality_score,

        "quality_label": analysis.quality_label,

        "issues": json.loads(
            analysis.issues
        ),

        "statistics": json.loads(
            analysis.statistics
        ),

        "explanation": json.loads(
            analysis.explanation
        ),

        "created_at": analysis.created_at
    }