import queue
import cv2
import pyvirtualcam
from ultralytics import YOLO
from time import time


# NOTE -  Function MEANT to be threaded...
def stream_to_virtual_cam(stop_event, bbox_queue):
    # Track object history across frames
    object_history = {}

    while not stop_event.is_set():
        # Load the YOLO model
        import torch

        model = (
            YOLO("models/best.pt")
            if torch.cuda.is_available()
            else YOLO("models/best.onnx")
        )

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
                ids = (
                    results[0].boxes.id.int().cpu().tolist()
                    if results[0].boxes.id is not None
                    else []
                )

                # Create a list to store bbox info
                bbox_data = []
                current_time = time()

                # Get class indices from detection results
                cls_indices = (
                    results[0].boxes.cls.cpu().numpy()
                    if results[0].boxes.cls is not None
                    else []
                )
                class_names = results[
                    0
                ].names  # Dictionary mapping indices to class names

                for box, conf, id, cls_idx in zip(boxes, confs, ids, cls_indices):
                    x1, y1, x2, y2 = map(int, box)
                    distance = calculate_distance(x2 - x1)
                    label = class_names[int(cls_idx)]  # Convert index to label name

                    # Initialize with no history
                    bbox_info = {
                        "id": id,
                        "bbox": (x1, y1, x2, y2),
                        "old_bbox": None,
                        "distance": distance,
                        "confidence": float(conf),
                        "time": current_time,
                        "prev_time": None,
                        "label": label,  # Add label to bbox info
                    }

                    # Check if we have history for this object
                    if id in object_history:
                        previous = object_history[id]
                        bbox_info["old_bbox"] = previous["bbox"]
                        bbox_info["prev_time"] = previous["time"]

                    # Add to current frame's data
                    bbox_data.append(bbox_info)

                    # Update history for next frame
                    object_history[id] = bbox_info

                # Clean up history (remove objects not seen in this frame)
                current_ids = set(ids)
                object_history = {
                    id: data for id, data in object_history.items() if id in current_ids
                }
                # Send bbox data to queue (non-blocking)
                try:
                    # Empty the queue first to avoid backlog
                    while not bbox_queue.empty():
                        bbox_queue.get_nowait()
                    # Put the new data
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
