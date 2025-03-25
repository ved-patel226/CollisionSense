from load import stream_to_virtual_cam
import threading
import time

stop_event = threading.Event()

virtual_cam_thread = threading.Thread(
    target=stream_to_virtual_cam, args=(stop_event,), daemon=True
)

try:
    virtual_cam_thread.start()
    print(f"Virtual Camera Thread ID: {virtual_cam_thread.ident}")

    time.sleep(5)
except KeyboardInterrupt:
    print("\nKeyboardInterrupt detected. Stopping thread...")
    stop_event.set()
except Exception:
    stop_event.set()
finally:
    virtual_cam_thread.join(timeout=2)
    print("Thread stopped.")
