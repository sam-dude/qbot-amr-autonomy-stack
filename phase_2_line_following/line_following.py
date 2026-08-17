"""
phase_2_line_following/line_following.py

Quanser-style line follower implemented in the same workflow as the lab:
- local observer on the PC
- robot-side script on the QBot
- joystick arm/disarm support
- downward camera as the primary line-following sensor
"""

import time
import numpy as np
import cv2

from qbot_helpers import QBotHardwareInterface


class LineFollowingController:
    """Simple proportional controller for a line centroid in the downward camera."""

    def __init__(self, target_speed=0.30, kp=1.0, kd=0.0):
        self.target_speed = float(target_speed)
        self.kp = float(kp)
        self.kd = float(kd)
        self.prev_error = 0.0

    def compute_line_error(self, image):
        if image is None:
            return 0.0, False

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape

        # Use the lower part of the image, consistent with the Quanser lab workflow.
        y0 = int(h * 0.5)
        roi = gray[y0:h, :]
        _, binary = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, False

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 30:
            return 0.0, False

        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0.0, False

        cx = M["m10"] / M["m00"]
        error = (cx / max(roi.shape[1], 1)) - 0.5
        return error * 2.0, True

    def step(self, image):
        error, valid = self.compute_line_error(image)
        if not valid:
            return 0.05, 0.6

        derivative = error - self.prev_error
        turn_rate = self.kp * error + self.kd * derivative
        self.prev_error = error
        forward_speed = self.target_speed * np.cos(np.clip(error, -1.0, 1.0))
        return float(forward_speed), float(np.clip(turn_rate, -1.2, 1.2))


if __name__ == "__main__":
    robot = QBotHardwareInterface(
        ip_driver="localhost",
        mode=1,
        enable_lidar=False,
        enable_realsense=False,
        enable_downward_cam=True,
    )
    controller = LineFollowingController(target_speed=0.30, kp=1.0, kd=0.0)

    print("[line_following] Waiting for the robot to be armed.")
    try:
        while True:
            sensors = robot.step(read_cameras=True, read_lidar=False)
            if robot.check_emergency_stop():
                print("[line_following] Emergency stop requested.")
                break

            if not robot.armed:
                robot.set_body_velocity(0.0, 0.0)
                time.sleep(0.02)
                continue

            image = sensors.downward_image
            v_cmd, omega_cmd = controller.step(image)
            robot.set_body_velocity(v_cmd, omega_cmd)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("[line_following] User interrupted.")
    finally:
        robot.set_body_velocity(0.0, 0.0)
        robot.close()
