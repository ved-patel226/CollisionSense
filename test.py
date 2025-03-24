import os
from PIL import Image


def print_jpg_dimensions(folder):
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg")):
            file_path = os.path.join(folder, filename)
            try:
                with Image.open(file_path) as img:
                    print(f"{filename}: {img.width} x {img.height}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    folder = "formatted_data/val/images"
    print_jpg_dimensions(folder)
