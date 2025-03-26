import numpy as np


def get_relative_coordinates(
    bbox, image_width, image_height, focal_length, known_width=1.8
):
    """
    Calculate the relative 3D coordinates (x, y, z) of an object from the camera.

    Args:
        bbox: Tuple (x1, y1, x2, y2) of bounding box coordinates
        image_width: Width of the camera frame in pixels
        image_height: Height of the camera frame in pixels
        focal_length: Focal length of the camera in pixels
        known_width: Known width of the object in meters (default: 1.8m for average car)

    Returns:
        Tuple (x, y, z) where:
        - x: Lateral distance from camera center (negative = left, positive = right)
        - y: Vertical distance from camera center (negative = up, positive = down)
        - z: Distance from camera (depth)
    """
    x1, y1, x2, y2 = bbox

    # Calculate center of the bounding box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Calculate distance (z-coordinate) using similar triangles
    bbox_width = x2 - x1
    z = (known_width * focal_length) / bbox_width

    # Calculate x coordinate (lateral position)
    # Convert from pixel coordinates to real-world coordinates
    # (0,0) is at the center of the image
    x = ((center_x - image_width / 2) * z) / focal_length

    # Calculate y coordinate (vertical position)
    y = ((center_y - image_height / 2) * z) / focal_length

    return (x, y, z)


def get_velocity(initial_pos, new_pos, time_elapsed):
    """
    Calculate the velocity of an object.

    Args:
        initial_pos: Tuple (x, y, z) representing the initial position in meters
        new_pos: Tuple (x, y, z) representing the new position in meters
        time_elapsed: Time elapsed between the two positions in seconds

    Returns:
        velocity: Tuple (vx, vy, vz) representing the velocity in m/s
    """
    vx = (new_pos[0] - initial_pos[0]) / time_elapsed
    vy = (new_pos[1] - initial_pos[1]) / time_elapsed
    vz = (new_pos[2] - initial_pos[2]) / time_elapsed

    return (vx, vy, vz)


def calculate_angle_to_object(x, z):
    """
    Calculate the angle to the object from the camera's forward direction.

    Args:
        x: Lateral distance
        z: Forward distance (depth)

    Returns:
        angle_degrees: Angle in degrees (negative = left, positive = right)
    """
    # Calculate angle in radians and convert to degrees
    angle_radians = np.arctan2(x, z)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def calculate_time_to_collision(distance, relative_velocity):
    """
    Calculate the time to collision.

    Args:
        distance: Distance to object in meters
        relative_velocity: Relative velocity in m/s (negative means approaching)

    Returns:
        ttc: Time to collision in seconds (None if not on collision course)
    """
    if relative_velocity >= 0:
        # Objects moving away or stationary have no collision
        return None

    # Time = Distance / Speed
    ttc = abs(distance / relative_velocity)
    return ttc
