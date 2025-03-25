import cv2
import pyvirtualcam
from ultralytics import YOLO


# NOTE -  Function MEANT to be threaded...
def stream_to_virtual_cam(stop_event):
    while not stop_event.is_set():

        # Load the YOLO model
        model = YOLO("training/runs/detect/train12/weights/best.pt")

        # Open the video file
        video_path = "training/test/sample5.mp4"
        cap = cv2.VideoCapture(video_path)

        KNOWN_WIDTH = 1.8
        FOCAL_LENGTH = 1000

        def calculate_distance(bbox_width):
            """Calculate distance using similar triangles"""
            if bbox_width <= 0:
                return 0
            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width
            return distance

        # Get frame properties for the virtual camera
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read a frame from the video.")
        height, width, _ = frame.shape
        fps = (
            cap.get(cv2.CAP_PROP_FPS) or 30
        )  # default to 30 if fps cannot be determined

        # Initialize the virtual camera
        with pyvirtualcam.Camera(
            width=width, height=height, fps=fps, print_fps=True
        ) as cam:
            # Rewind the capture to the first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # Run YOLO inference on the frame
                results = model.track(frame, persist=True, conf=0.85, verbose=False)
                annotated_frame = results[0].plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Process detections and add distance annotations
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()

                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = map(int, box)
                    bbox_width = x2 - x1
                    distance = calculate_distance(bbox_width)
                    text = f"{distance:.2f}m"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2
                    )
                    roi_x, roi_y = x1, y1 - 50
                    cv2.rectangle(
                        annotated_frame,
                        (roi_x, roi_y - text_height - baseline),
                        (roi_x + text_width, roi_y + baseline),
                        (0, 0, 0),
                        cv2.FILLED,
                    )
                    cv2.putText(
                        annotated_frame,
                        text,
                        (roi_x, roi_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        2,
                    )

                # Send the annotated frame to the virtual camera
                cam.send(annotated_frame)
                cam.sleep_until_next_frame()

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()
