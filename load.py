import cv2
from ultralytics import YOLO
import time

# Load the YOLO model
model = YOLO("runs/detect/train5/weights/best.pt")

# Open the video file
video_path = "sample.mp4"
cap = cv2.VideoCapture(video_path)

# Constants for distance calculation
# Average width of a car in meters (adjust this based on the cars in your video)
KNOWN_WIDTH = 1.8  # meters

# Focal length calculation - needs to be calibrated for accurate results
# To calibrate: place a car at a known distance, detect it and measure its width in pixels
# Then: focal_length = (width_in_pixels * known_distance) / KNOWN_WIDTH
# Example: if car is 200 pixels wide at 10m: focal_length = (200 * 10) / 1.8 = 1111.11
FOCAL_LENGTH = 1000  # placeholder - calibrate for your camera


def calculate_distance(bbox_width):
    """Calculate distance using similar triangles"""
    if bbox_width <= 0:
        return 0

    # Distance = (Known Width × Focal Length) / Width in pixels
    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width
    return distance


# Loop through the video frames
while cap.isOpened():
    time.sleep(1)
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Process each detection
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Calculate bounding box width
            bbox_width = x2 - x1

            # Calculate distance using similar triangles
            distance = calculate_distance(bbox_width)

            # Add distance text to the annotated frame
            cv2.putText(
                annotated_frame,
                f"{distance:.2f}m",
                (x1 - 30, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                2,
            )

        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
