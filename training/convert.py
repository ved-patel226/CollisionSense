import os
import json
import shutil
from pathlib import Path


def process_dataset(input_labels_path, input_img_path, output_path):
    """Process a dataset by copying images, converting labels, and formatting to YOLO format."""
    # Create the output directory and subdirectories
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create imgs and labels subdirectories
    imgs_dir = output_path / "imgs"
    labels_dir = output_path / "labels"
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    missing = 0
    total = 0

    # Load JSON data
    with open(input_labels_path, "r") as file:
        data = json.load(file)

    # Copy images and their corresponding JSON files
    for d in data:
        filename = d["name"]
        total += 1

        # Search for the file recursively in subdirs
        src_file = None
        for f in input_img_path.rglob(filename):
            if f.is_file():
                src_file = f
                break

        if src_file:
            # Save image to imgs subdirectory
            shutil.copy2(src_file, imgs_dir / filename)

            # Save temporary JSON to output directory
            json_dst = output_path / f"{os.path.splitext(filename)[0]}.json"
            with open(json_dst, "w") as json_file:
                json.dump(d, json_file)
        else:
            missing += 1
            print(
                f"Missing: {input_img_path / filename} | Total Count: {missing}/{total}"
            )

    # Filter labels to keep only relevant categories
    files = [f for f in os.listdir(output_path) if f.endswith(".json")]
    good_labels = ["car", "person", "bus", "truck", "bike", "train"]

    for file in files:
        with open(os.path.join(output_path, file), "r") as f:
            data = json.load(f)

        # Remove labels whose category is not in good_labels
        data["labels"] = [
            dat for dat in data["labels"] if dat["category"] in good_labels
        ]

        with open(os.path.join(output_path, file), "w") as f:
            json.dump(data, f, indent=4)

    # Convert JSON labels to YOLO format text files
    target_files = [file for file in os.listdir(output_path) if file.endswith(".json")]
    num_to_class = {
        "car": 1,
        "person": 2,
        "bus": 3,
        "truck": 4,
        "bike": 5,
        "train": 6,
    }

    for file in target_files:
        with open(output_path / file) as f:
            data = json.load(f)

        output = []

        for label in data["labels"]:
            num = num_to_class[label["category"]]
            label = label["box2d"]

            x1, y1, x2, y2 = label["x1"], label["y1"], label["x2"], label["y2"]

            height = y2 - y1
            width = x2 - x1
            middle_x = x1 + width / 2
            middle_y = y1 + height / 2

            output.append(f"{num} {middle_x} {middle_y} {width} {height}")

        # Save txt label file to labels subdirectory
        txt_filename = file.replace(".json", ".txt")
        with open(labels_dir / txt_filename, "w") as f:
            f.write("\n".join(output))

        # Remove temporary JSON file
        os.remove(output_path / file)

    print(f"Processed dataset at {output_path}")


# Process validation data
val_label_path = Path(
    "bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
)
val_img_path = Path("bdd100k/images/100k/val")
val_output_path = Path("formatted_data/val")
process_dataset(val_label_path, val_img_path, val_output_path)

# Process training data
train_label_path = Path(
    "bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
)
train_img_path = Path("bdd100k/images/100k/train")
train_output_path = Path("formatted_data/train")
process_dataset(train_label_path, train_img_path, train_output_path)
