import queue
import cv2
import pyvirtualcam
from ultralytics import YOLO


# NOTE -  Function MEANT to be threaded...
def stream_to_virtual_cam(stop_event, bbox_queue):
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
        with pyvirtualcam.Camera(width=width, height=height, fps=fps) as cam:
            # Rewind the capture to the first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while cap.isOpened():
                success, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if not success:
                    break

                # Run YOLO inference on the frame
                results = model.track(frame, persist=True, conf=0.75, verbose=False)

                # Process detections and add distance annotations
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                ids = results[0].boxes.id.int().cpu().tolist()

                # Create a list to store bbox info
                bbox_data = []

                for box, conf, id in zip(boxes, confs, ids):
                    x1, y1, x2, y2 = map(int, box)
                    distance = calculate_distance(x2 - x1)

                    bbox_data.append(
                        {
                            "id": id,
                            "bbox": (x1, y1, x2, y2),
                            "distance": distance,
                            "confidence": float(conf),
                        }
                    )

                # Send bbox data to queue (non-blocking)
                try:
                    bbox_queue.put(bbox_data, block=False)
                except queue.Full:
                    # If queue is full, get rid of the oldest item
                    try:
                        bbox_queue.get_nowait()
                        bbox_queue.put(bbox_data, block=False)
                    except:
                        pass

                # Send the annotated frame to the virtual camera
                cam.send(frame)
                cam.sleep_until_next_frame()

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()
