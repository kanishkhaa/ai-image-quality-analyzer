import cv2
import numpy as np


def calculate_image_features(image):

    if image is None:
        raise ValueError("Invalid image")


    # =====================================================
    # Grayscale
    # =====================================================

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # =====================================================
    # Brightness
    # =====================================================

    brightness = float(
        np.mean(gray)
    )


    # =====================================================
    # Contrast
    # =====================================================

    contrast = float(
        np.std(gray)
    )


    # =====================================================
    # Sharpness
    # =====================================================

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    sharpness = float(
        laplacian.var()
    )


    # =====================================================
    # Exposure
    # =====================================================

    underexposed_percentage = float(
        np.mean(gray < 30) * 100
    )

    overexposed_percentage = float(
        np.mean(gray > 225) * 100
    )


    # =====================================================
    # Entropy
    # =====================================================

    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    histogram = histogram.flatten()

    histogram = histogram / (
        histogram.sum() + 1e-8
    )

    entropy = float(
        -np.sum(
            histogram *
            np.log2(histogram + 1e-8)
        )
    )


    # =====================================================
    # Saturation
    # =====================================================

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    saturation = float(
        np.mean(
            hsv[:, :, 1]
        )
    )


    # =====================================================
    # Dimensions
    # =====================================================

    height, width = image.shape[:2]


    return {

        "brightness":
            round(brightness, 3),

        "contrast":
            round(contrast, 3),

        "sharpness":
            round(sharpness, 3),

        "underexposed_percentage":
            round(
                underexposed_percentage,
                3
            ),

        "overexposed_percentage":
            round(
                overexposed_percentage,
                3
            ),

        "entropy":
            round(entropy, 3),

        "saturation":
            round(saturation, 3),

        "width":
            width,

        "height":
            height
    }