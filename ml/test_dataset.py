from dataset import IQADataset
from torch.utils.data import DataLoader


TRAIN_CSV = "data/processed/train.csv"
IMAGE_DIR = "data/BIQ2021/archive"

dataset = IQADataset(
    csv_file=TRAIN_CSV,
    image_dir=IMAGE_DIR,
    train=True
)

print("Dataset size:", len(dataset))

image, mos = dataset[0]

print("Image shape:", image.shape)
print("MOS:", mos)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

images, mos_values = next(iter(loader))

print("\n===== BATCH TEST =====")
print("Batch image shape:", images.shape)
print("Batch MOS shape:", mos_values.shape)