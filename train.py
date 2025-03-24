from ultralytics import YOLO
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import os

from common import print_divider
import shutil

RAW_PATH = Path("./raw_data")
FORMATTED_PATH = Path("./formatted_data")

IMG_PATH = RAW_PATH / "data_object_image_2" / "training" / "image_2"
IMG_FILES = os.listdir(IMG_PATH)

LABEL_PATH = RAW_PATH / "labels"
LABEL_FILES = os.listdir(LABEL_PATH)

# Sort files because sometimes theres disturbances in files for some reason
LABEL_FILES.sort()
IMG_FILES.sort()

# Create pairs of corresponding image and label filenames.
PAIRS = list(zip(IMG_FILES, LABEL_FILES))

# Split the paired data into train and validation sets (80% train, 20% val).
train, val = train_test_split(PAIRS, test_size=0.2, random_state=42)


print("File Samples:")
print(f"Labels: {LABEL_FILES[0]}, {LABEL_FILES[-1]}")
print(f"Images: {IMG_FILES[0]}, {IMG_FILES[-1]}")
print(f"Paired: {PAIRS[0]}, {PAIRS[-1]}")
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
    data=FORMATTED_PATH / "kitti.yaml",
    epochs=100,
)
model.save("car.pt")
