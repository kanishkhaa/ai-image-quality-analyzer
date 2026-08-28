FROM python:3.11-slim

# Prevent Python from creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Prevent Python output buffering
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# System dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY backend/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install CPU-only PyTorch
RUN pip install --no-cache-dir \
    torch==2.13.0+cpu \
    torchvision==0.28.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Copy backend
COPY backend/app /app/backend/app

# Copy ML code
COPY ml /app/ml

# Copy trained model
COPY models /app/models

# Create upload directory
RUN mkdir -p /app/backend/uploads

# Python import paths
ENV PYTHONPATH=/app:/app/backend:/app/ml

# FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]