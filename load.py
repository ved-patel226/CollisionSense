import cv2
from ultralytics import YOLO
import time  # Add this at the top of your file if not already imported

# Load the YOLO model
model = YOLO("runs/detect/train5/weights/best.pt")

# Open the video file
video_path = "test/sample5.mp4"
cap = cv2.VideoCapture(video_path)

KNOWN_WIDTH = 1.8

FOCAL_LENGTH = 1000


def calculate_distance(bbox_width):
    """Calculate distance using similar triangles"""
    if bbox_width <= 0:
        return 0

    # Distance = (Known Width × Focal Length) / Width in pixels
    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width
    return distance


# Loop through the video frames

while cap.isOpened():
    start_time = time.time()  # Start timer for FPS limiting

    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame, conf=0.85)
        annotated_frame = results[0].plot()

        # Process each detection filtering by confidence > 0.5
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confs):
            # Visualize the results on the frame

            # Get box coordinates
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Calculate bounding box width
            bbox_width = x2 - x1

            # Calculate distance using similar triangles
            distance = calculate_distance(bbox_width)

            # Add distance text to the annotated frame
            text = f"{distance:.2f}m"
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2
            )
            roi_x = x1
            roi_y = y1 - 50

            # Draw black rectangle as background for the text
            cv2.rectangle(
                annotated_frame,
                (roi_x, roi_y - text_height - baseline),
                (roi_x + text_width, roi_y + baseline),
                (0, 0, 0),
                cv2.FILLED,
            )
            # Overlay the text on the black rectangle
            cv2.putText(
                annotated_frame,
                text,
                (roi_x, roi_y),
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

    # Limit the loop to ~30 frames per second
    elapsed = time.time() - start_time
    delay = max(1 / 30 - elapsed, 0)
    time.sleep(delay)

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
