import pandas as pd
from pathlib import Path

# Dataset paths
DATA_DIR = Path("data/BIQ2021/archive")
CSV_PATH = DATA_DIR / "BIQ2021.csv"

# Read CSV
df = pd.read_csv(CSV_PATH)

print("\n===== DATASET INFO =====")
print("Rows:", len(df))
print("Columns:", list(df.columns))

print("\n===== FIRST 10 ROWS =====")
print(df.head(10))

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== MOS STATISTICS =====")
print(df["MOS"].describe())

print("\n===== IMAGE COUNT =====")
image_count = len(list(DATA_DIR.glob("*.jpg")))
print("Images found:", image_count)