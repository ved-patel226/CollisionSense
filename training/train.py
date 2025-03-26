from ultralytics import YOLO
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import os

from common import print_divider


FORMATTED_PATH = Path("./formatted_data")

os.makedirs(FORMATTED_PATH / "val" / "images", exist_ok=True)
os.makedirs(FORMATTED_PATH / "train" / "images", exist_ok=True)

VAL_IMGS, TRAIN_IMGS = (
    os.listdir(FORMATTED_PATH / "val" / "images"),
    os.listdir(FORMATTED_PATH / "train" / "images"),
)

VAL_LABELS, TRAIN_LABELS = (
    os.listdir(FORMATTED_PATH / "val" / "labels"),
    os.listdir(FORMATTED_PATH / "train" / "labels"),
)

# Create pairs of corresponding image and label filenames.
train = list(zip(TRAIN_IMGS, TRAIN_LABELS))
val = list(zip(VAL_IMGS, VAL_LABELS))

print("File Samples:")
print(f"Train: {train[0]}, {train[-1]}")
print(f"Val: {val[0]}, {val[-1]}")
print_divider()

print("Train vs Val")
print(f"# of train: {len(train)}")
print(f"# of val : {len(val)}")
print_divider()

with open(FORMATTED_PATH / "classes.json", "r") as f:
    classes = json.load(f)

print("Classes:")
print(classes)
print_divider()

model = YOLO("yolo11n.pt")

train_res = model.train(
    data=FORMATTED_PATH / "dataset.yaml",
    epochs=50,
)
model.save("car.pt")
