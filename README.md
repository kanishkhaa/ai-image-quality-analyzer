# AI Image Quality Analyzer

An AI-powered image quality assessment and defect detection system that analyzes uploaded images and provides a quality score, quality classification, detected image issues, statistical measurements, and an explanation of the results.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20App-success?style=for-the-badge)](https://ai-image-quality-analyzer.onrender.com)
[![Backend API](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge)](https://ai-image-quality-backend.onrender.com/docs)
[![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB?style=for-the-badge)](https://react.dev/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge)](https://www.docker.com/)

---

## Live Demo

### Web Application

**https://ai-image-quality-analyzer.onrender.com**

### Backend API Documentation

**https://ai-image-quality-backend.onrender.com/docs**

The frontend is deployed as a Render Static Site, while the FastAPI backend is deployed separately as a Docker-based web service.

---

## Overview

Image quality plays an important role in computer vision, digital media, e-commerce, content moderation, photography, and automated inspection systems.

Poor-quality images can contain problems such as:

* Blur
* Underexposure
* Overexposure
* Low contrast
* Excessive noise
* Overall poor perceptual quality

The **AI Image Quality Analyzer** automatically evaluates an uploaded image and produces a structured quality report.

The system combines:

1. **Deep learning-based image quality prediction**
2. **Computer vision-based image quality measurements**
3. **Rule-based defect detection**
4. **REST API backend**
5. **Interactive React frontend**
6. **Database-backed analysis history**

---

## Key Features

### AI-Based Quality Prediction

A trained PyTorch image-quality model predicts a perceptual quality score for the uploaded image.

The model:

* Resizes images to `224 × 224`
* Applies ImageNet normalization
* Performs inference using PyTorch
* Produces a normalized quality score between `0` and `1`

---

### Image Defect Detection

The analyzer evaluates multiple image characteristics.

#### Blur Detection

Sharpness is estimated using the **variance of the Laplacian**.

The system categorizes blur severity based on the calculated sharpness value.

#### Exposure Analysis

The analyzer checks:

* Average brightness
* Dark pixel ratio
* Bright pixel ratio

This allows it to identify:

* Underexposure
* Overexposure

#### Contrast Analysis

Image contrast is estimated using the standard deviation of grayscale pixel intensities.

Low-contrast images are identified and assigned severity levels.

#### Noise Detection

Image noise is estimated by comparing the original grayscale image with a Gaussian-blurred version.

The resulting residual is used to estimate noise intensity.

---

### Structured Analysis Results

Each analysis returns information including:

* Image filename
* Quality score
* Quality label
* Detected issues
* Issue severity
* Issue confidence
* Image statistics
* Explanation
* Analysis timestamp

---

### Analysis History

The backend stores completed analyses in a database.

Users can retrieve:

* All previous analyses
* A specific analysis by ID

This makes the application useful not only as a one-time analyzer but also as a foundation for an image-quality monitoring system.

---

## How It Works

```text
                    USER
                      │
                      ▼
             ┌─────────────────┐
             │ React Frontend  │
             │   Vite + CSS    │
             └────────┬────────┘
                      │
                      │ HTTP / REST API
                      ▼
             ┌─────────────────┐
             │ FastAPI Backend │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Image Validation│
             │ & File Storage  │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ AI / CV Analysis│
             ├─────────────────┤
             │ PyTorch Model   │
             │ OpenCV          │
             │ NumPy           │
             │ PIL             │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Quality Report  │
             └────────┬────────┘
                      │
              ┌───────┴────────┐
              ▼                ▼
       Database Storage     React UI
```

---

## Architecture

The project follows a separated frontend/backend architecture.

### Frontend

The frontend is built using:

* React
* Vite
* Tailwind CSS
* Axios

It provides the user interface for uploading images and displaying analysis results.

### Backend

The backend is built using:

* Python
* FastAPI
* SQLAlchemy
* SQLite/database layer
* PyTorch
* OpenCV
* NumPy
* Pillow

The backend handles:

* Image uploads
* Validation
* AI inference
* Image-quality analysis
* Database operations
* REST API responses

### Machine Learning Layer

The `ml/` directory contains the machine-learning and computer-vision components.

```text
ml/
├── dataset.py
├── evaluate.py
├── features.py
├── image_analyzer.py
├── inspect_data.py
├── model.py
├── prepare_data.py
├── test_dataset.py
├── test_model.py
└── train.py
```

The trained model is stored separately under:

```text
models/
```

---

## Technology Stack

| Layer               | Technology                |
| ------------------- | ------------------------- |
| Frontend            | React                     |
| Build Tool          | Vite                      |
| Styling             | Tailwind CSS              |
| API Client          | Axios                     |
| Backend             | FastAPI                   |
| Language            | Python                    |
| ORM                 | SQLAlchemy                |
| Machine Learning    | PyTorch                   |
| Computer Vision     | OpenCV                    |
| Image Processing    | Pillow                    |
| Numerical Computing | NumPy                     |
| Database            | SQLite / SQLAlchemy       |
| Containerization    | Docker                    |
| Frontend Deployment | Render Static Site        |
| Backend Deployment  | Render Docker Web Service |

---

## Project Structure

```text
ai-image-quality-analyzer/
│
├── backend/
│   ├── app/
│   │   ├── analysis.py
│   │   ├── database.py
│   │   ├── main.py
│   │   ├── models.py
│   │   └── routes.py
│   │
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── assets/
│   │   ├── App.css
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   │
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   └── index.html
│
├── ml/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── features.py
│   ├── image_analyzer.py
│   ├── inspect_data.py
│   ├── model.py
│   ├── prepare_data.py
│   ├── test_dataset.py
│   ├── test_model.py
│   └── train.py
│
├── models/
│   └── best_iqa_model.pth
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── .gitignore
```

---

## Supported Image Formats

The backend currently accepts:

```text
JPG
JPEG
PNG
WEBP
BMP
```

Unsupported formats are rejected before analysis.

---

# API Documentation

The backend exposes a REST API using FastAPI.

Interactive Swagger documentation is available at:

**https://ai-image-quality-backend.onrender.com/docs**

---

## Health Check

### `GET /api/health`

Checks whether the backend is running.

Example response:

```json
{
  "status": "healthy",
  "service": "AI Image Quality Analyzer"
}
```

---

## Analyze Image

### `POST /api/analyze`

Uploads an image and performs the complete quality analysis pipeline.

### Request

```text
Content-Type: multipart/form-data
```

Parameter:

```text
file
```

Example using cURL:

```bash
curl -X POST \
  https://ai-image-quality-backend.onrender.com/api/analyze \
  -F "file=@image.jpg"
```

### Response

The API returns a structured result containing:

```json
{
  "id": 1,
  "filename": "image.jpg",
  "quality_score": 0.82,
  "quality_label": "Good",
  "issues": [],
  "statistics": {},
  "explanation": {},
  "created_at": "..."
}
```

---

## Get Analysis History

### `GET /api/analyses`

Returns previously stored image analyses.

Example:

```bash
curl https://ai-image-quality-backend.onrender.com/api/analyses
```

---

## Get Single Analysis

### `GET /api/analyses/{analysis_id}`

Returns one analysis using its database ID.

Example:

```bash
curl https://ai-image-quality-backend.onrender.com/api/analyses/1
```

---

# Machine Learning Pipeline

The image-quality model performs regression-based quality prediction.

The inference pipeline is:

```text
Input Image
     │
     ▼
Image Validation
     │
     ▼
RGB Conversion
     │
     ▼
Resize to 224 × 224
     │
     ▼
Normalize using ImageNet statistics
     │
     ▼
PyTorch Model
     │
     ▼
Quality Score
     │
     ▼
Clamp to [0, 1]
```

In parallel, classical computer-vision analysis calculates:

```text
Image
 │
 ├── Sharpness
 │      └── Blur detection
 │
 ├── Brightness
 │      ├── Underexposure
 │      └── Overexposure
 │
 ├── Contrast
 │      └── Low contrast detection
 │
 └── Noise
        └── Noise detection
```

The results from these components are combined into the final analysis response.

---

# Local Development

## Prerequisites

Make sure the following are installed:

* Python 3.11+
* Node.js
* npm
* Git
* Docker Desktop (optional but recommended)

---

## 1. Clone the Repository

```bash
git clone https://github.com/kanishkhaa/ai-image-quality-analyzer.git

cd ai-image-quality-analyzer
```

---

# Running the Backend Locally

Navigate to the backend:

```bash
cd backend
```

Create a virtual environment:

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server from the project root:

```bash
uvicorn backend.app.main:app --reload
```

The backend will be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

Health check:

```text
http://localhost:8000/api/health
```

---

# Running the Frontend Locally

Open another terminal.

Navigate to:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Create a `.env` file inside `frontend/`:

```env
VITE_API_URL=http://localhost:8000
```

Start the development server:

```bash
npm run dev
```

The frontend will normally be available at:

```text
http://localhost:5173
```

---

# Environment Variables

The frontend uses:

```env
VITE_API_URL=http://localhost:8000
```

For production:

```env
VITE_API_URL=https://ai-image-quality-backend.onrender.com
```

> Do not commit sensitive secrets to GitHub. Environment variables containing API keys, credentials, tokens, or other secrets should be configured through the deployment platform instead.

---

# Docker

The project also includes Docker configuration for containerized deployment.

Build and run using Docker Compose:

```bash
docker compose up --build
```

To run in detached mode:

```bash
docker compose up --build -d
```

Stop the containers:

```bash
docker compose down
```

---

# Deployment

The project is deployed using a separated frontend/backend architecture.

## Frontend

The React/Vite frontend is deployed on Render as a **Static Site**.

Configuration:

```text
Root Directory: frontend
Build Command: npm run build
Publish Directory: dist
```

Live application:

**https://ai-image-quality-analyzer.onrender.com**

Render supports React/Vite frontends as static sites and publishes the generated build directory through its CDN.

---

## Backend

The FastAPI backend is deployed separately as a Docker-based Render Web Service.

Backend:

**https://ai-image-quality-backend.onrender.com**

API documentation:

**https://ai-image-quality-backend.onrender.com/docs**

---

# Deployment Architecture

```text
                         Internet
                            │
                            ▼
              ┌──────────────────────────┐
              │      Render Static Site  │
              │                          │
              │    React + Vite          │
              │                          │
              │ ai-image-quality-        │
              │ analyzer.onrender.com    │
              └────────────┬─────────────┘
                           │
                           │ HTTPS REST API
                           ▼
              ┌──────────────────────────┐
              │   Render Web Service     │
              │                          │
              │      FastAPI             │
              │                          │
              │ ai-image-quality-        │
              │ backend.onrender.com     │
              └────────────┬─────────────┘
                           │
              ┌────────────┼─────────────┐
              │            │             │
              ▼            ▼             ▼
          PyTorch       OpenCV        Database
           Model       Analysis       Storage
```

Render automatically deploys linked Git repositories when changes are pushed to the configured branch, making the deployment suitable for continuous updates.

---

# Security Considerations

The application includes basic validation for uploaded images.

The backend:

* Validates file extensions
* Generates unique filenames for uploaded images
* Prevents the original filename from being used directly as the stored filename
* Removes uploaded files when analysis fails
* Validates that uploaded files can be decoded as images

For a production-grade deployment, additional protections could include:

* File-size limits
* MIME-type validation
* Authentication and authorization
* Rate limiting
* Malware scanning
* Object storage such as S3-compatible storage
* Restricted CORS origins
* HTTPS-only API communication
* Database authentication and managed database infrastructure

---

# Testing

The machine-learning directory includes test and evaluation utilities:

```text
ml/
├── evaluate.py
├── test_dataset.py
└── test_model.py
```

Run frontend linting:

```bash
cd frontend
npm run lint
```

Run the ML tests according to the project's Python environment and test configuration.

---

# Future Improvements

Potential improvements include:

* [ ] Add user authentication
* [ ] Add image-size and file-size validation
* [ ] Add batch image analysis
* [ ] Add downloadable PDF analysis reports
* [ ] Add image-quality comparison
* [ ] Add visual defect heatmaps
* [ ] Add model confidence visualization
* [ ] Add model performance metrics to the UI
* [ ] Add cloud object storage for uploaded images
* [ ] Add production-grade PostgreSQL support
* [ ] Add API rate limiting
* [ ] Add automated CI/CD testing
* [ ] Add monitoring and application logging
* [ ] Improve model accuracy with larger and more diverse datasets
* [ ] Add additional image-quality metrics
* [ ] Add support for video-frame quality analysis

---

# Use Cases

This system can serve as a foundation for applications such as:

### E-Commerce

Automatically identify poor-quality product images before they are published.

### Content Management

Validate uploaded images before accepting them into a media library.

### Computer Vision Pipelines

Filter low-quality images before sending them to downstream computer-vision models.

### Photography

Provide automated feedback about image quality.

### Digital Asset Management

Automatically categorize and flag problematic media assets.

### Automated Inspection

Use image-quality measurements as an initial quality-control stage before further visual inspection.

---

# Why This Project?

This project combines several areas of software engineering and artificial intelligence:

* Machine Learning
* Computer Vision
* Deep Learning
* REST API Development
* Full-Stack Development
* Database Integration
* Docker
* Cloud Deployment
* Frontend Engineering

It demonstrates how a machine-learning model can be integrated into a complete production-style web application rather than being used only as an isolated Python script.

---

# Author

**Kanishkhaa M S**

Computer Science & Engineering Student

GitHub:
https://github.com/kanishkhaa

---

# License

This project is intended for educational and portfolio purposes.

If you plan to use the project commercially, review the licenses of the datasets, pretrained components, libraries, and other third-party dependencies used by the project.

---

## Acknowledgements

Built using open-source technologies including:

* React
* Vite
* Tailwind CSS
* FastAPI
* PyTorch
* OpenCV
* NumPy
* Pillow
* SQLAlchemy
* Docker
* Render
