import cv2
import numpy as np


def analyze_image(image_bytes: bytes):
    # Convert bytes to NumPy array
    image_array = np.frombuffer(image_bytes, np.uint8)

    # Decode image
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Unable to decode image")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Basic image information
    height, width = gray.shape

    # 1. Sharpness
    # Higher variance = sharper image
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2. Brightness
    brightness = float(np.mean(gray))

    # 3. Contrast
    contrast = float(np.std(gray))

    # 4. Noise estimation
    noise = float(np.std(gray - cv2.GaussianBlur(gray, (5, 5), 0)))

    # 5. Saturation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = float(np.mean(hsv[:, :, 1]))

    return {
        "width": width,
        "height": height,
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "noise": round(noise, 2),
        "saturation": round(saturation, 2)
    }