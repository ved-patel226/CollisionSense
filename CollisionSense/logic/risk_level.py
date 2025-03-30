import math


def calculate_distance(pos):
    """Calculate Euclidean distance from origin (0,0) to position (x, z)"""
    x, z = pos
    return math.sqrt(x**2 + z**2)


def calculate_velocity_magnitude(velocity):
    """Calculate the magnitude of a velocity vector (vx, vz)"""
    vx, vz = velocity
    return math.sqrt(vx**2 + vz**2)


def calculate_risk_level(car_pos, car_vel, max_deceleration=7):
    """
    Calculate risk level between ego car (at origin) and another car based on relative position and velocities.

    Parameters:
    - car_pos: (x, z) position of other car relative to ego car at (0,0)
    - car_vel: (vx, vz) velocity of other car relative to ego car
    - max_deceleration: maximum deceleration rate in m/s²

    Returns:
    - risk_level: A value from 0 to 100
    """
    # Calculate current distance between cars (other car to origin)
    distance = calculate_distance(car_pos)

    # Since we're in the ego car's reference frame, its velocity is (0,0)
    # and car_vel is already the relative velocity
    rel_vx, rel_vz = car_vel
    rel_velocity = calculate_velocity_magnitude(car_vel)

    # Check if cars are moving toward each other
    x, z = car_pos
    dot_product = x * rel_vx + z * rel_vz
    approaching = dot_product < 0

    # Calculate minimum future distance (closest approach) and time to closest approach
    # For an object moving at constant velocity, the time to closest approach is:
    # t = -(x*vx + z*vz)/(vx²+vz²)
    if rel_velocity > 0.001:  # Avoid division by zero
        t_closest = -dot_product / (rel_vx**2 + rel_vz**2)

        # If t_closest is negative, we're already at the closest approach
        t_closest = max(0, t_closest)

        # Calculate position at closest approach
        future_x = x + rel_vx * t_closest
        future_z = z + rel_vz * t_closest
        min_distance = math.sqrt(future_x**2 + future_z**2)
    else:
        # Object is nearly stationary relative to ego
        min_distance = distance
        t_closest = float("inf")

    # Safety threshold for minimum passing distance (m)
    # Increase this value for pedestrians if needed
    safe_passing_distance = 3.0

    # Time to collision (TTC) if trajectories actually intersect
    if min_distance < safe_passing_distance and approaching and rel_velocity > 0:
        # Adjust TTC based on actual intersection
        ttc = distance / rel_velocity  # Time in seconds
    else:
        ttc = float("inf")  # Not on collision course

    # Calculate stopping distances
    # Ego car velocity in its own reference frame is 0
    v1 = 0
    # Other car velocity magnitude is the relative velocity
    v2 = rel_velocity

    stopping_dist2 = v2**2 / (2 * max_deceleration)

    # Calculate combined stopping distance
    combined_stopping_dist = stopping_dist2  # Since v1 = 0

    # Risk factors

    # Distance factor: Higher risk when closer
    safe_distance = max(combined_stopping_dist * 1.5, 5)  # Adjusted safety buffer
    distance_factor = max(0, min(100, 100 * (1 - distance / safe_distance)))

    # TTC factor: Higher risk with lower TTC
    if ttc == float("inf"):
        ttc_factor = 0
    else:
        ttc_factor = max(
            0, min(100, 100 * (1 - ttc / 10))
        )  # 10 seconds as safe threshold

    # Sudden stop factor: Risk if stopping distances exceed current distance
    sudden_stop_factor = max(
        0, min(100, 100 * combined_stopping_dist / max(distance, 1))
    )

    # Perpendicular motion factor: Reduce risk for objects moving perpendicular
    # Calculate angle between position and velocity vectors
    if rel_velocity > 0.001:
        cos_angle = dot_product / (distance * rel_velocity)
        # -1 means directly approaching, 0 means perpendicular, 1 means moving away
        perpendicular_factor = abs(cos_angle)  # 0 for perpendicular, 1 for parallel
    else:
        perpendicular_factor = 0

    # Minimum distance factor: Low risk if minimum future distance is large
    min_distance_factor = max(
        0, min(100, 100 * (1 - min_distance / safe_passing_distance))
    )
    min_distance_factor = (
        min_distance_factor if min_distance < safe_passing_distance * 3 else 0
    )

    # Combined risk level (weighted average of factors)
    risk_level = (
        0.3 * distance_factor
        + 0.3 * ttc_factor
        + 0.15 * sudden_stop_factor
        + 0.25 * min_distance_factor
    )

    # Reduce risk for perpendicular motion
    if perpendicular_factor < 0.3:  # More perpendicular than parallel
        risk_level *= perpendicular_factor * 2  # Reduce risk significantly

    return min(100, max(0, round(risk_level)))
