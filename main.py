import os

os.environ["COLLISION_SENSE_DEBUG"] = "true"

from CollisionSense.main import stream_to_virtual_cam, show_gui
import time
import threading
import queue

# FIXME - Sometimes, doesn't work unless you do modprobe v4l2loopback..

stop_event = threading.Event()
bbox_queue = queue.Queue(maxsize=10)  # Limit queue size to avoid memory issues

virtual_cam_thread = threading.Thread(
    target=stream_to_virtual_cam, args=(stop_event, bbox_queue), daemon=True
)

try:
    virtual_cam_thread.start()
    print(f"Virtual Camera Thread ID: {virtual_cam_thread.ident}")

    time.sleep(1)

    show_gui(bbox_queue)

except KeyboardInterrupt:
    print("\nKeyboardInterrupt detected. Stopping thread...")
    stop_event.set()
except Exception:
    stop_event.set()
finally:
    virtual_cam_thread.join(timeout=2)
    print("Thread stopped.")
