import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# =========================
# Paths
# =========================

DATA_DIR = Path("data/BIQ2021/archive")
CSV_PATH = DATA_DIR / "BIQ2021.csv"

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Load dataset
# =========================

df = pd.read_csv(CSV_PATH)

print(f"Total images: {len(df)}")


# =========================
# Verify images exist
# =========================

def image_exists(filename):
    return (DATA_DIR / filename).exists()


df["image_exists"] = df["Images"].apply(image_exists)

missing_images = df[~df["image_exists"]]

if len(missing_images) > 0:
    print(f"Missing images: {len(missing_images)}")
    print(missing_images.head())
    raise FileNotFoundError("Some images are missing.")

df = df.drop(columns=["image_exists"])

print("All images verified.")


# =========================
# Train / Test split
# =========================

train_df, test_df = train_test_split(
    df,
    test_size=2000,
    random_state=42
)


# =========================
# Train / Validation split
# =========================

train_df, val_df = train_test_split(
    train_df,
    test_size=2000,
    random_state=42
)


# =========================
# Save splits
# =========================

train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

val_df.to_csv(
    OUTPUT_DIR / "validation.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)


# =========================
# Print results
# =========================

print("\n===== DATA SPLIT =====")
print(f"Training   : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Testing    : {len(test_df)}")

print("\nFiles created:")
print(OUTPUT_DIR / "train.csv")
print(OUTPUT_DIR / "validation.csv")
print(OUTPUT_DIR / "test.csv")