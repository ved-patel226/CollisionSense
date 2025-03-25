from load import stream_to_virtual_cam
import threading
import time


virtual_cam_thread = threading.Thread(target=stream_to_virtual_cam, daemon=True)
virtual_cam_thread.start()

time.sleep(5)

virtual_cam_thread.stop()
