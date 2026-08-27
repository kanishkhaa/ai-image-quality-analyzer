from fastapi import FastAPI
from app.routes import router

app = FastAPI(
    title="AI Image Quality Analyzer",
    description="AI-powered image quality and defect detection API",
    version="1.0.0"
)

app.include_router(router, prefix="/api")


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }