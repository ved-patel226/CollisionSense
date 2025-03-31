import cv2
import tkinter as tk
from PIL import Image, ImageTk
import queue
import numpy as np
from CollisionSense.logic import (
    get_relative_coordinates,
    get_velocity,
    calculate_risk_level,
)
import os
from time import time


class CollisionSenseGUI:
    def __init__(self, bbox_queue):
        self.bbox_queue = bbox_queue
        self.root = None
        self.cap = None
        self.lbl = None
        self.bbox_info_label = None

        self.label_to_width = {"car": 1.8, "person": 0.15}

    @staticmethod
    def is_debug():
        return os.environ.get("COLLISION_SENSE_DEBUG") == "true"

    @staticmethod
    def normalize_with_range(max_possible, min_possible, target_min, target_max, num):
        normalized = (num - min_possible) / (max_possible - min_possible)
        scaled = normalized * (target_max - target_min) + target_min
        return scaled

    def setup_gui(self):
        """Initialize the GUI components"""
        self.root = tk.Tk()
        self.root.title("CollisionSense")

        # Make it fullscreen
        self.root.attributes("-fullscreen", True)
        # Set background color to black
        self.root.configure(background="black")

        # Create frame for video without padding
        video_frame = tk.Frame(self.root, bg="black")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create frame for bbox info without padding
        info_frame = tk.Frame(self.root, bg="black")
        info_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Label for video
        self.lbl = tk.Label(video_frame, bg="black")
        self.lbl.pack(fill=tk.BOTH, expand=True)

        # Label for bbox information
        self.bbox_info_label = tk.Label(info_frame, bg="black", fg="white")
        self.bbox_info_label.pack(fill=tk.BOTH, expand=True)

        # Add key binding to exit fullscreen with Escape key
        self.root.bind("<Escape>", lambda e: self.on_closing())

        # Setup video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot open webcam")
            exit()

        # Set up window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def show_frame(self):
        """Process and display a single frame with bounding boxes"""
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame (BGR to RGB)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Try to get bbox data from queue
            try:
                bbox_data = self.bbox_queue.get_nowait()
                # Draw bounding boxes on the frame
                self.process_bounding_boxes(cv2image, bbox_data)
            except queue.Empty:
                # No new bbox data available
                pass

            # Get current dimensions of the label
            label_width = self.lbl.winfo_width()
            label_height = self.lbl.winfo_height()

            # Ensure we have valid dimensions (on first run they may be 1)
            if label_width > 1 and label_height > 1:
                # Resize frame to fit label while maintaining aspect ratio
                img_height, img_width = cv2image.shape[:2]
                ratio = min(label_width / img_width, label_height / img_height)
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
                cv2image = cv2.resize(
                    cv2image, (new_width, new_height), interpolation=cv2.INTER_AREA
                )

            # Display the frame
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.lbl.imgtk = imgtk  # keep a reference
            self.lbl.configure(image=imgtk)

        self.lbl.after(10, self.show_frame)

    def process_bounding_boxes(self, cv2image, bbox_data):
        """Process and draw bounding boxes on the image"""
        img_height, img_width, _ = cv2image.shape

        for obj in bbox_data:
            x1, y1, x2, y2 = obj["bbox"]
            conf = obj["confidence"]

            # Get width based on object label with fallback to default value if label not found
            width = self.label_to_width.get(obj["label"], 1.8)

            relative_coords = get_relative_coordinates(
                obj["bbox"], img_width, img_height, focal_length=1000, known_width=width
            )

            # Adjust beta based on confidence (lower confidence results in a lower beta)
            beta = self.normalize_with_range(0.75, 1.0, 0.0, 75.0, conf)
            roi = cv2image[y1:y2, x1:x2]
            bright_roi = cv2.convertScaleAbs(roi, alpha=1.0, beta=beta)

            # Create a mask with rounded edges
            mask = self.create_rounded_mask(roi)

            # Blend the brightened ROI with the original ROI using the mask
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask_norm = mask_gray.astype(float) / 255.0
            mask_norm = mask_norm[..., None]  # align dimensions for broadcasting
            roi_out = (bright_roi * mask_norm + roi * (1 - mask_norm)).astype(np.uint8)

            # Determine if blue tint should be applied
            car_in_lane = relative_coords[0] < 1.5 and relative_coords[0] > -1.5

            if car_in_lane:
                pass  # process warning system

            velocity = self.calculate_velocity(
                obj, relative_coords, img_width, img_height
            )

            risk = calculate_risk_level(
                (relative_coords[0], relative_coords[2]), (velocity[0], velocity[2])
            )

            # Prepare the ROI to be applied
            roi_to_apply = self.apply_tint_if_needed(roi_out, car_in_lane, risk)

            # Use the same mask to apply the ROI to the original image
            mask_norm_3ch = np.repeat(mask_norm, 3, axis=2)  # Convert to 3-channel mask
            cv2image[y1:y2, x1:x2] = roi_to_apply * mask_norm_3ch + cv2image[
                y1:y2, x1:x2
            ] * (1 - mask_norm_3ch)

            # Add debug info if needed
            if self.is_debug():
                self.add_debug_info(
                    cv2image, obj, relative_coords, x1, y1, velocity, risk, obj["label"]
                )

    @staticmethod
    def create_rounded_mask(roi):
        """Create a mask with rounded edges for the ROI"""
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

        return mask

    @staticmethod
    def apply_tint_if_needed(roi_out, car_in_lane, risk=0):
        """Apply color tint to the ROI based on risk level and lane position"""
        # Create a tint color
        tint = np.zeros_like(roi_out)

        if car_in_lane:
            # Calculate color intensity based on risk level (0-100)
            red_intensity = min(255, int(2.55 * risk))  # More red with higher risk
            blue_intensity = max(0, 255 - red_intensity)  # Less blue with higher risk

            # Apply color to tint
            tint[:, :, 0] = red_intensity  # Red channel
            tint[:, :, 2] = blue_intensity  # Blue channel
        else:
            # Even if car is not in lane, apply red tint
            red_intensity = min(255, int(2.55 * (risk // 2)))
            tint[:, :, 0] = red_intensity

        alpha = min(0.5, 0.2 + (risk / 200))

        return cv2.addWeighted(roi_out, 1 - alpha, tint, alpha, 0)

    def add_debug_info(
        self, cv2image, obj, relative_coords, x1, y1, velocity, risk, label
    ):
        """Add debugging information to the image"""

        # Put object ID on the frame
        text_position = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10)

        # Add black background rectangle for better text visibility
        text = f"Label: {label} | V: {velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f} | POS: {relative_coords[0]:.2f}, {relative_coords[1]:.2f}, {relative_coords[2]:.2f} | Risk: {risk}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        top_left = (text_position[0], text_position[1] - text_height - 5)
        bottom_right = (text_position[0] + text_width, text_position[1] + baseline)
        cv2.rectangle(cv2image, top_left, bottom_right, (0, 0, 0), -1)
        cv2.putText(
            cv2image,
            text,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def calculate_velocity(obj, relative_coords, img_width, img_height):
        """Calculate velocity of an object"""
        if obj["old_bbox"] and obj["prev_time"]:
            old_relative_coords = get_relative_coordinates(
                obj["old_bbox"], img_width, img_height, focal_length=1000
            )

            velocity = get_velocity(
                old_relative_coords,
                relative_coords,
                time() - obj["prev_time"],
            )
            obj["prev_velocity"] = velocity
            return velocity
        elif "prev_velocity" in obj and obj["prev_velocity"]:
            return obj["prev_velocity"]
        else:
            return (0, 0, 0)

    def on_closing(self):
        """Clean up resources and close the application"""
        if self.cap:
            self.cap.release()
        if self.root:
            self.root.destroy()

    def start(self):
        """Start the GUI application"""
        self.setup_gui()
        self.show_frame()
        self.root.mainloop()


# NOTE - Meant to be run in the MAIN THREAD
def show_gui(bbox_queue):
    app = CollisionSenseGUI(bbox_queue)
    app.start()
