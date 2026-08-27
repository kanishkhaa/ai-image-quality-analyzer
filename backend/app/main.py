from fastapi import FastAPI

app = FastAPI(
    title="AI Image Quality Analyzer",
    description="AI-powered image quality and defect detection API",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }