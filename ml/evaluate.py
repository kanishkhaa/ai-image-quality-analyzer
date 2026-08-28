import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import numpy as np

from dataset import IQADataset
from model import IQAModel


# ============================================================
# Configuration
# ============================================================

TEST_CSV = "data/processed/test.csv"
IMAGE_DIR = "data/BIQ2021/archive"
MODEL_PATH = "models/best_iqa_model.pth"

BATCH_SIZE = 32


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("IMAGE QUALITY ASSESSMENT - TEST EVALUATION")
print("=" * 60)

print("Device:", device)


# ============================================================
# Dataset
# ============================================================

test_dataset = IQADataset(
    csv_file=TEST_CSV,
    image_dir=IMAGE_DIR,
    train=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test samples:", len(test_dataset))


# ============================================================
# Model
# ============================================================

model = IQAModel()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# Your training script saved a dictionary
if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model = model.to(device)
model.eval()


# ============================================================
# Prediction
# ============================================================

all_predictions = []
all_targets = []


print("\nRunning inference...\n")

with torch.no_grad():

    for images, mos in test_loader:

        images = images.to(device)

        predictions = model(
            images
        ).squeeze(1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_targets.extend(
            mos.numpy()
        )


# ============================================================
# Convert to NumPy
# ============================================================

predictions = np.array(
    all_predictions
)

targets = np.array(
    all_targets
)


# ============================================================
# Metrics
# ============================================================

mae = mean_absolute_error(
    targets,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        targets,
        predictions
    )
)

plcc, _ = pearsonr(
    targets,
    predictions
)

srcc, _ = spearmanr(
    targets,
    predictions
)


# ============================================================
# Results
# ============================================================

print("=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"PLCC : {plcc:.4f}")
print(f"SRCC : {srcc:.4f}")

print("=" * 60)


# ============================================================
# Prediction statistics
# ============================================================

print("\nPrediction statistics:")

print(
    f"Actual MOS range      : "
    f"{targets.min():.3f} - {targets.max():.3f}"
)

print(
    f"Predicted score range : "
    f"{predictions.min():.3f} - {predictions.max():.3f}"
)

print(
    f"Actual mean           : "
    f"{targets.mean():.3f}"
)

print(
    f"Predicted mean        : "
    f"{predictions.mean():.3f}"
)