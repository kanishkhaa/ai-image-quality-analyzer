import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image


# ============================================================
# Make ml/ imports work when this file is imported elsewhere
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from model import IQAModel


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = os.path.abspath(
    os.path.join(
        CURRENT_DIR,
        "..",
        "models",
        "best_iqa_model.pth"
    )
)

# ImageNet normalization used during training
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Load model
# ============================================================

_model = None


def load_model():
    """
    Load the trained IQA model only once.
    """

    global _model

    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"IQA model not found at: {MODEL_PATH}"
        )

    print(
        f"Loading IQA model from: {MODEL_PATH}"
    )

    model = IQAModel()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(
            checkpoint
        )

    model = model.to(DEVICE)

    model.eval()

    _model = model

    print(
        f"IQA model loaded successfully on {DEVICE}"
    )

    return _model


# ============================================================
# Image Loading
# ============================================================

def load_image(image_path):
    """
    Safely load an image.

    Returns:
        PIL RGB image
        OpenCV BGR image
    """

    if not os.path.exists(image_path):
        raise ValueError(
            f"Image file does not exist: {image_path}"
        )

    try:

        pil_image = Image.open(
            image_path
        )

        pil_image.load()

        pil_image = pil_image.convert(
            "RGB"
        )

    except Exception as e:

        raise ValueError(
            f"Image is corrupted or unreadable: {str(e)}"
        )

    cv_image = cv2.imread(
        image_path,
        cv2.IMREAD_COLOR
    )

    if cv_image is None:
        raise ValueError(
            "OpenCV could not decode the image."
        )

    return pil_image, cv_image


# ============================================================
# Deep Learning Quality Prediction
# ============================================================

def predict_quality(pil_image):
    """
    Predict perceptual image quality.

    Returns:
        Quality score between 0 and 1.
    """

    model = load_model()

    # --------------------------------------------------------
    # IMPORTANT:
    # Must match training preprocessing exactly.
    # --------------------------------------------------------

    image = pil_image.resize(
        (224, 224)
    )

    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    # HWC -> CHW
    image = np.transpose(
        image,
        (2, 0, 1)
    )

    mean = np.asarray(
        IMAGE_MEAN,
        dtype=np.float32
    ).reshape(3, 1, 1)

    std = np.asarray(
        IMAGE_STD,
        dtype=np.float32
    ).reshape(3, 1, 1)

    image = (
        image - mean
    ) / std

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        prediction = model(
            tensor
        )

    score = float(
        prediction.squeeze().cpu().item()
    )

    # Model is a regression model.
    # Keep final output inside valid MOS range.
    score = max(
        0.0,
        min(1.0, score)
    )

    return score


# ============================================================
# Sharpness
# ============================================================

def calculate_sharpness(gray):
    """
    Estimate image sharpness using
    variance of Laplacian.
    """

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    return float(
        laplacian.var()
    )


def detect_blur(sharpness):
    """
    Conservative blur detection.

    IMPORTANT:
    These thresholds are intentionally less aggressive
    than the previous version.
    """

    if sharpness < 25:

        return {
            "type": "blur",
            "severity": "high",
            "confidence": 0.90
        }

    elif sharpness < 50:

        return {
            "type": "blur",
            "severity": "medium",
            "confidence": 0.80
        }

    elif sharpness < 80:

        return {
            "type": "blur",
            "severity": "low",
            "confidence": 0.65
        }

    return None


# ============================================================
# Brightness / Exposure
# ============================================================

def calculate_brightness(gray):

    return float(
        np.mean(gray)
    )


def calculate_dark_pixel_ratio(gray):

    dark_pixels = np.sum(
        gray < 30
    )

    return float(
        dark_pixels / gray.size
    )


def calculate_bright_pixel_ratio(gray):

    bright_pixels = np.sum(
        gray > 225
    )

    return float(
        bright_pixels / gray.size
    )


def detect_exposure(
    brightness,
    dark_ratio,
    bright_ratio
):
    """
    Conservative exposure detection.

    Exposure problems are reported only when the
    image is clearly under/over exposed.
    """

    issues = []

    # --------------------------------------------------------
    # Underexposure
    # --------------------------------------------------------

    if (
        brightness < 45
        and dark_ratio > 0.30
    ):

        issues.append(
            {
                "type": "underexposure",
                "severity": "high",
                "confidence": 0.90
            }
        )

    elif (
        brightness < 60
        and dark_ratio > 0.20
    ):

        issues.append(
            {
                "type": "underexposure",
                "severity": "medium",
                "confidence": 0.78
            }
        )

    # --------------------------------------------------------
    # Overexposure
    # --------------------------------------------------------

    if (
        brightness > 220
        and bright_ratio > 0.30
    ):

        issues.append(
            {
                "type": "overexposure",
                "severity": "high",
                "confidence": 0.90
            }
        )

    elif (
        brightness > 200
        and bright_ratio > 0.20
    ):

        issues.append(
            {
                "type": "overexposure",
                "severity": "medium",
                "confidence": 0.78
            }
        )

    return issues


# ============================================================
# Contrast
# ============================================================

def calculate_contrast(gray):

    return float(
        np.std(gray)
    )


def detect_low_contrast(contrast):
    """
    Conservative contrast detection.
    """

    if contrast < 15:

        return {
            "type": "low_contrast",
            "severity": "high",
            "confidence": 0.88
        }

    elif contrast < 25:

        return {
            "type": "low_contrast",
            "severity": "medium",
            "confidence": 0.75
        }

    elif contrast < 32:

        return {
            "type": "low_contrast",
            "severity": "low",
            "confidence": 0.60
        }

    return None


# ============================================================
# Noise
# ============================================================

def estimate_noise(gray):

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    residual = (
        gray.astype(np.float32)
        - blurred.astype(np.float32)
    )

    return float(
        np.std(residual)
    )


def detect_noise(noise):
    """
    Conservative noise detection.

    Natural textures can produce high-frequency
    variation, so thresholds are intentionally high.
    """

    if noise > 30:

        return {
            "type": "noise",
            "severity": "high",
            "confidence": 0.88
        }

    elif noise > 22:

        return {
            "type": "noise",
            "severity": "medium",
            "confidence": 0.75
        }

    elif noise > 15:

        return {
            "type": "noise",
            "severity": "low",
            "confidence": 0.60
        }

    return None


# ============================================================
# Entropy
# ============================================================

def calculate_entropy(gray):

    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    histogram = histogram.flatten()

    total = histogram.sum()

    if total == 0:
        return 0.0

    probabilities = histogram / total

    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        probabilities
        * np.log2(probabilities)
    )

    return float(entropy)


# ============================================================
# Quality Label
# ============================================================

def quality_label(score):
    """
    Convert model quality score into a human-readable label.

    ML model is the PRIMARY source of the final quality score.
    """

    if score >= 75:

        return "GOOD"

    elif score >= 55:

        return "ACCEPTABLE"

    elif score >= 35:

        return "DEGRADED"

    else:

        return "DEFECTIVE"


# ============================================================
# Final Quality Decision
# ============================================================

def calculate_overall_label(
    quality_score,
    issues
):
    """
    Combine ML quality score with CV checks.

    IMPORTANT:
    The ML model remains the primary decision maker.

    CV checks are used only to provide additional context
    and to downgrade an image when there is strong evidence
    of a serious quality problem.

    A single minor CV finding cannot make a good image
    defective.
    """

    label = quality_label(
        quality_score
    )

    severe_issues = [
        issue
        for issue in issues
        if issue["severity"] == "high"
    ]

    medium_issues = [
        issue
        for issue in issues
        if issue["severity"] == "medium"
    ]

    # --------------------------------------------------------
    # VERY HIGH ML SCORE
    # --------------------------------------------------------
    # If the model is highly confident that the image is good,
    # do not downgrade it because of minor CV measurements.

    if quality_score >= 75:

        # Only downgrade if there are at least
        # TWO independent severe quality indicators.
        if len(severe_issues) >= 2:

            label = "ACCEPTABLE"

        else:

            label = "GOOD"

        return label

    # --------------------------------------------------------
    # ACCEPTABLE ML SCORE
    # --------------------------------------------------------

    if quality_score >= 55:

        if len(severe_issues) >= 2:

            return "DEGRADED"

        if len(severe_issues) == 1:

            return "ACCEPTABLE"

        return "ACCEPTABLE"

    # --------------------------------------------------------
    # DEGRADED ML SCORE
    # --------------------------------------------------------

    if quality_score >= 35:

        if len(severe_issues) >= 1:
            return "DEGRADED"

        if len(medium_issues) >= 2:
            return "DEGRADED"

        return "DEGRADED"

    # --------------------------------------------------------
    # LOW ML SCORE
    # --------------------------------------------------------

    return "DEFECTIVE"


# ============================================================
# Main Analyzer
# ============================================================

def analyze_image(image_path):
    """
    Analyze image quality.

    Returns:
        quality_score
        quality_label
        issues
        statistics
        explanation
    """

    # ========================================================
    # Load Image
    # ========================================================

    try:

        pil_image, cv_image = load_image(
            image_path
        )

    except ValueError as e:

        return {
            "quality_score": 0,
            "quality_label": "DEFECTIVE",
            "issues": [
                {
                    "type": "corruption",
                    "severity": "high",
                    "confidence": 0.99
                }
            ],
            "statistics": {},
            "explanation": [
                str(e)
            ]
        }

    # ========================================================
    # Grayscale
    # ========================================================

    gray = cv2.cvtColor(
        cv_image,
        cv2.COLOR_BGR2GRAY
    )

    # ========================================================
    # Calculate Image Statistics
    # ========================================================

    sharpness = calculate_sharpness(
        gray
    )

    brightness = calculate_brightness(
        gray
    )

    dark_ratio = calculate_dark_pixel_ratio(
        gray
    )

    bright_ratio = calculate_bright_pixel_ratio(
        gray
    )

    contrast = calculate_contrast(
        gray
    )

    noise = estimate_noise(
        gray
    )

    entropy = calculate_entropy(
        gray
    )

    # ========================================================
    # Detect Quality Issues
    # ========================================================

    issues = []

    # --------------------------------------------------------
    # Blur
    # --------------------------------------------------------

    blur_issue = detect_blur(
        sharpness
    )

    if blur_issue is not None:

        issues.append(
            blur_issue
        )

    # --------------------------------------------------------
    # Exposure
    # --------------------------------------------------------

    exposure_issues = detect_exposure(
        brightness,
        dark_ratio,
        bright_ratio
    )

    issues.extend(
        exposure_issues
    )

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    contrast_issue = detect_low_contrast(
        contrast
    )

    if contrast_issue is not None:

        issues.append(
            contrast_issue
        )

    # --------------------------------------------------------
    # Noise
    # --------------------------------------------------------

    noise_issue = detect_noise(
        noise
    )

    if noise_issue is not None:

        issues.append(
            noise_issue
        )

    # ========================================================
    # ML Quality Prediction
    # ========================================================

    quality_prediction = predict_quality(
        pil_image
    )

    quality_score = round(
        quality_prediction * 100,
        2
    )

    # ========================================================
    # Final Label
    # ========================================================

    label = calculate_overall_label(
        quality_score,
        issues
    )

    # ========================================================
    # Explainability
    # ========================================================

    explanation = []

    # --------------------------------------------------------
    # ML Explanation
    # --------------------------------------------------------

    if quality_score >= 75:

        explanation.append(
            f"The AI model estimates a high-quality image "
            f"with a quality score of {quality_score:.1f}/100."
        )

    elif quality_score >= 55:

        explanation.append(
            f"The AI model estimates acceptable image quality "
            f"with a quality score of {quality_score:.1f}/100."
        )

    elif quality_score >= 35:

        explanation.append(
            f"The AI model estimates degraded image quality "
            f"with a quality score of {quality_score:.1f}/100."
        )

    else:

        explanation.append(
            f"The AI model estimates poor image quality "
            f"with a quality score of {quality_score:.1f}/100."
        )

    # --------------------------------------------------------
    # Blur Explanation
    # --------------------------------------------------------

    if blur_issue:

        explanation.append(
            f"Low sharpness was detected "
            f"(sharpness={sharpness:.2f}), "
            f"which may indicate blur."
        )

    # --------------------------------------------------------
    # Underexposure
    # --------------------------------------------------------

    if any(
        issue["type"] == "underexposure"
        for issue in issues
    ):

        explanation.append(
            f"The image contains a relatively high "
            f"amount of dark content "
            f"({dark_ratio * 100:.1f}%), "
            f"suggesting possible underexposure."
        )

    # --------------------------------------------------------
    # Overexposure
    # --------------------------------------------------------

    if any(
        issue["type"] == "overexposure"
        for issue in issues
    ):

        explanation.append(
            f"The image contains a relatively high "
            f"amount of very bright content "
            f"({bright_ratio * 100:.1f}%), "
            f"suggesting possible overexposure."
        )

    # --------------------------------------------------------
    # Noise
    # --------------------------------------------------------

    if noise_issue:

        explanation.append(
            f"High-frequency variation was detected "
            f"(noise estimate={noise:.2f}), "
            f"which may indicate image noise."
        )

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    if contrast_issue:

        explanation.append(
            f"Low contrast was detected "
            f"(contrast={contrast:.2f})."
        )

    # --------------------------------------------------------
    # No Issues
    # --------------------------------------------------------

    if not issues:

        explanation.append(
            "No significant quality issues were detected "
            "by the additional computer-vision checks."
        )

    # --------------------------------------------------------
    # Model Information
    # --------------------------------------------------------

    explanation.append(
        "The final quality assessment primarily uses "
        "the trained ResNet-18 image-quality model, "
        "while computer-vision checks provide supporting "
        "quality information."
    )

    # ========================================================
    # Final Response
    # ========================================================

    result = {

        "quality_score": quality_score,

        "quality_label": label,

        "issues": issues,

        "statistics": {

            "sharpness": round(
                sharpness,
                2
            ),

            "brightness": round(
                brightness,
                2
            ),

            "contrast": round(
                contrast,
                2
            ),

            "noise": round(
                noise,
                2
            ),

            "entropy": round(
                entropy,
                2
            ),

            "dark_pixel_ratio": round(
                dark_ratio,
                4
            ),

            "bright_pixel_ratio": round(
                bright_ratio,
                4
            )
        },

        "explanation": explanation
    }

    return result


# ============================================================
# Command-Line Test
# ============================================================

if __name__ == "__main__":

    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Analyze image quality"
    )

    parser.add_argument(
        "image",
        help="Path to image"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("IMAGE QUALITY ANALYZER")
    print("=" * 60)

    try:

        result = analyze_image(
            args.image
        )

        print(
            json.dumps(
                result,
                indent=4
            )
        )

    except Exception as e:

        print("\nERROR:")
        print(str(e))
