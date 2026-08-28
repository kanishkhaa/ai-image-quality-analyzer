import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import IQADataset
from model import IQAModel


# ============================================================
# Configuration
# ============================================================

TRAIN_CSV = "data/processed/train.csv"
VAL_CSV = "data/processed/validation.csv"
IMAGE_DIR = "data/BIQ2021/archive"

# CPU-friendly batch size
BATCH_SIZE = 32

# Number of epochs
EPOCHS = 10

# Learning rate for regression head
LEARNING_RATE = 1e-3

# Early stopping patience
PATIENCE = 3

# Model output directory
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("IMAGE QUALITY ASSESSMENT TRAINING")
print("=" * 60)

print(f"Using device: {device}")


# ============================================================
# Dataset
# ============================================================

print("\nLoading datasets...")

train_dataset = IQADataset(
    csv_file=TRAIN_CSV,
    image_dir=IMAGE_DIR,
    train=True
)

val_dataset = IQADataset(
    csv_file=VAL_CSV,
    image_dir=IMAGE_DIR,
    train=False
)

print(f"Training samples:   {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")


# ============================================================
# DataLoader
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


print(f"Training batches per epoch:   {len(train_loader)}")
print(f"Validation batches per epoch: {len(val_loader)}")


# ============================================================
# Model
# ============================================================

print("\nLoading model...")

model = IQAModel()
model = model.to(device)


# ============================================================
# Show trainable parameters
# ============================================================

total_parameters = sum(
    p.numel() for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"Total parameters:      {total_parameters:,}")
print(f"Trainable parameters:  {trainable_parameters:,}")
print(
    f"Frozen parameters:     "
    f"{total_parameters - trainable_parameters:,}"
)


# ============================================================
# Loss Function
# ============================================================

# SmoothL1 is more robust to noisy MOS labels than MSE
criterion = nn.SmoothL1Loss()


# ============================================================
# Optimizer
# ============================================================
optimizer = torch.optim.AdamW(
    [
        {
            "params": model.backbone.layer3.parameters(),
            "lr": 1e-4
        },
        {
            "params": model.backbone.layer4.parameters(),
            "lr": 1e-4
        },
        {
            "params": model.backbone.fc.parameters(),
            "lr": 1e-3
        }
    ],
    weight_decay=1e-4
)

# ============================================================
# Learning Rate Scheduler
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=1
)


# ============================================================
# Mixed Precision
# ============================================================

use_amp = device.type == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# ============================================================
# Training Variables
# ============================================================

best_val_loss = float("inf")
epochs_without_improvement = 0

best_model_path = os.path.join(
    MODEL_DIR,
    "best_iqa_model.pth"
)


# ============================================================
# Training Loop
# ============================================================

print("\nStarting training...\n")

training_start = time.time()


for epoch in range(EPOCHS):

    epoch_start = time.time()

    # ========================================================
    # Training
    # ========================================================

    model.train()

    train_loss = 0.0

    train_progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} - Training",
        leave=True
    )

    for images, mos in train_progress:

        images = images.to(
            device,
            non_blocking=True
        )

        mos = mos.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp
        ):

            predictions = model(images).squeeze(1)

            loss = criterion(
                predictions,
                mos
            )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        if use_amp:

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

        else:

            loss.backward()

            optimizer.step()

        # ----------------------------------------------------
        # Track loss
        # ----------------------------------------------------

        train_loss += (
            loss.item() * images.size(0)
        )

        train_progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss /= len(train_dataset)


    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    val_loss = 0.0

    val_progress = tqdm(
        val_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} - Validation",
        leave=True
    )

    with torch.no_grad():

        for images, mos in val_progress:

            images = images.to(
                device,
                non_blocking=True
            )

            mos = mos.to(
                device,
                non_blocking=True
            )

            with torch.autocast(
                device_type=device.type,
                enabled=use_amp
            ):

                predictions = model(
                    images
                ).squeeze(1)

                loss = criterion(
                    predictions,
                    mos
                )

            val_loss += (
                loss.item() * images.size(0)
            )

            val_progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

    val_loss /= len(val_dataset)


    # ========================================================
    # Learning Rate Scheduler
    # ========================================================

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]


    # ========================================================
    # Epoch Results
    # ========================================================

    epoch_time = time.time() - epoch_start

    print("\n" + "-" * 60)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Val Loss:   {val_loss:.4f}"
    )

    print(
        f"Learning Rate: {current_lr:.6f}"
    )

    print(
        f"Epoch Time: {epoch_time:.1f} seconds"
    )

    print("-" * 60)


    # ========================================================
    # Save Best Model
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "epoch": epoch + 1
            },
            best_model_path
        )

        print(
            f"✓ Best model saved to: "
            f"{best_model_path}"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )


    # ========================================================
    # Early Stopping
    # ========================================================

    if epochs_without_improvement >= PATIENCE:

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# Training Complete
# ============================================================

total_time = time.time() - training_start

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Total training time: "
    f"{total_time / 60:.2f} minutes"
)

print(
    f"Best model: "
    f"{best_model_path}"
)

print("=" * 60)