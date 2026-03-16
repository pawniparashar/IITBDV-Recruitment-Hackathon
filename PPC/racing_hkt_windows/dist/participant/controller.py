'''
PPC Hackathon — Participant Boilerplate
You must implement two functions: plan() and control()
'''

# ─── TYPES (for reference) ────────────────────────────────────────────────────

# Path: list of waypoints [{"x": float, "y": float}, ...]
# State: {"x", "y", "yaw", "vx", "vy", "yaw_rate"} 
# CmdFeedback: {"throttle", "steer"}         

# ─── CONTROLLER ───────────────────────────────────────────────────────────────
import numpy as np

def steering(path: list[dict], state: dict):

    length_of_car = 2.6
    # Calculate steering angle based on path and vehicle state

    x = state["x"]
    y = state["y"]
    yaw = state["yaw"]

    # find closest waypoint
    closest_index = 0
    min_dist = 1e9

    for i in range(len(path)):
        dx = path[i]["x"] - x
        dy = path[i]["y"] - y
        dist = dx*dx + dy*dy

        if dist < min_dist:
            min_dist = dist
            closest_index = i

    # look ahead waypoint
    lookahead = 3
    target_index = min(closest_index + lookahead, len(path)-1)
    target = path[target_index]

    dx = target["x"] - x
    dy = target["y"] - y

    target_angle = np.arctan2(dy, dx)

    angle_error = target_angle - yaw

    # normalize angle
    while angle_error > np.pi:
        angle_error -= 2*np.pi
    while angle_error < -np.pi:
        angle_error += 2*np.pi

    steer = 0.5 * angle_error

    # 0.5 in the max steering angle in radians (about 28.6 degrees)
    return np.clip(steer, -0.5, 0.5)


def throttle_algorithm(target_speed, current_speed, dt):

    error = target_speed - current_speed

    if error > 0:
        throttle = 0.4
        brake = 0.0
    else:
        throttle = 0.0
        brake = 0.2

    # clip throttle and brake to [0, 1]
    return np.clip(throttle, 0.0, 1.0), np.clip(brake, 0.0, 1.0)


def control(
    path: list[dict],
    state: dict,
    cmd_feedback: dict,
    step: int,
) -> tuple[float, float, float]:

    throttle = 0.0
    steer    = 0.0
    brake = 0.0
   
    # TODO: implement your controller here
    steer = steering(path, state)

    # startup push so car moves immediately
    if step < 40:
        throttle = 0.5
        brake = 0.0
    else:
        target_speed = 5.0
        global integral
        throttle, brake = throttle_algorithm(target_speed, state["vx"], 0.05)

    return throttle, steer, brake
