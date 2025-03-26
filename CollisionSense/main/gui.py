import cv2
import tkinter as tk
from PIL import Image, ImageTk
import queue
import numpy as np
from CollisionSense.logic import get_relative_coordinates, get_velocity
import os


def is_debug():
    return os.environ["COLLISION_SENSE_DEBUG"] == "true"


def normalize_with_range(max_possible, min_possible, target_min, target_max, num):
    normalized = (num - min_possible) / (max_possible - min_possible)
    scaled = normalized * (target_max - target_min) + target_min
    return scaled


def show_frame(cap, lbl, bbox_queue, bbox_info_label):
    ret, frame = cap.read()
    if ret:
        # Convert the frame (BGR to RGB)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Try to get bbox data from queue
        try:
            bbox_data = bbox_queue.get_nowait()

            # Draw bounding boxes on the frame
            for obj in bbox_data:
                x1, y1, x2, y2 = obj["bbox"]
                conf = obj["confidence"]

                img_height, img_width, _ = cv2image.shape

                # Adjust beta based on confidence (lower confidence results in a lower beta)
                beta = normalize_with_range(0.75, 1.0, 0.0, 75.0, conf)
                roi = cv2image[y1:y2, x1:x2]
                bright_roi = cv2.convertScaleAbs(roi, alpha=1.0, beta=beta)

                # Create a mask with rounded edges
                mask = np.zeros_like(roi, dtype=np.uint8)
                h, w = roi.shape[:2]
                radius = 20  # adjust for desired curvature
                # Fill rectangular areas
                cv2.rectangle(mask, (radius, 0), (w - radius, h), (255, 255, 255), -1)
                cv2.rectangle(mask, (0, radius), (w, h - radius), (255, 255, 255), -1)
                # Draw circles for rounded corners
                cv2.circle(mask, (radius, radius), radius, (255, 255, 255), -1)
                cv2.circle(mask, (w - radius, radius), radius, (255, 255, 255), -1)
                cv2.circle(mask, (radius, h - radius), radius, (255, 255, 255), -1)
                cv2.circle(mask, (w - radius, h - radius), radius, (255, 255, 255), -1)

                # Blend the brightened ROI with the original ROI using the mask
                mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask_norm = mask_gray.astype(float) / 255.0
                mask_norm = mask_norm[..., None]  # align dimensions for broadcasting
                roi_out = (bright_roi * mask_norm + roi * (1 - mask_norm)).astype(
                    np.uint8
                )

                cv2image[y1:y2, x1:x2] = roi_out

                if is_debug():
                    # Put object ID on the frame
                    text_position = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10)

                    relative_coords = get_relative_coordinates(
                        obj["bbox"], img_width, img_height, focal_length=1000
                    )

                    if obj["old_bbox"] and obj["prev_time"]:
                        old_relative_coords = get_relative_coordinates(
                            obj["old_bbox"], img_width, img_height, focal_length=1000
                        )

                        velocity = get_velocity(
                            old_relative_coords,
                            relative_coords,
                            obj["prev_time"],
                        )
                        obj["prev_velocity"] = velocity

                    elif "prev_velocity" in obj and obj["prev_velocity"]:
                        velocity = obj["prev_velocity"]
                    else:
                        velocity = (0, 0, 0)

                    # Add black background rectangle for better text visibility
                    (text_width, text_height), baseline = cv2.getTextSize(
                        f"ID: {obj['id']}| V: {velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f}",
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        1,
                    )
                    top_left = (text_position[0], text_position[1] - text_height - 5)
                    bottom_right = (
                        text_position[0] + text_width,
                        text_position[1] + baseline,
                    )
                    cv2.rectangle(cv2image, top_left, bottom_right, (0, 0, 0), -1)
                    cv2.putText(
                        cv2image,
                        f"ID: {obj['id']}| V: {velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f}",
                        text_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

            # Update info label with object count and details
            info_text = f"Objects detected: {len(bbox_data)}\n"
            for obj in bbox_data[:10]:  # Show details for first 3 objects

                info_text += f'{obj["id"]}\n'
            if len(bbox_data) > 10:
                info_text += f"...and {len(bbox_data)-3} more"
            bbox_info_label.config(text=info_text)

        except queue.Empty:
            # No new bbox data available
            pass

        # Display the frame
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl.imgtk = imgtk  # keep a reference
        lbl.configure(image=imgtk)

    lbl.after(10, lambda: show_frame(cap, lbl, bbox_queue, bbox_info_label))


# NOTE - Meant to be run in the MAIN THREAD
def show_gui(bbox_queue):
    root = tk.Tk()
    root.title("CollisionSense")

    # Create frame for video
    video_frame = tk.Frame(root)
    video_frame.pack(side=tk.LEFT, padx=10, pady=10)

    # Create frame for bbox info
    info_frame = tk.Frame(root)
    info_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

    # Label for video
    lbl = tk.Label(video_frame)
    lbl.pack()

    # Label for bbox information
    bbox_info_label = tk.Label(
        info_frame,
        text="Waiting for detections...",
        justify=tk.LEFT,
        font=("Arial", 12),
        bg="#f0f0f0",
        width=30,
        height=10,
        anchor="nw",
    )
    bbox_info_label.pack(fill=tk.BOTH, expand=True)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        exit()

    show_frame(cap, lbl, bbox_queue, bbox_info_label)

    def on_closing():
        cap.release()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
