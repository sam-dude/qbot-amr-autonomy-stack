"""
line_following.py

Quanser-aligned line follower for the QBot Platform.

This module models the hardware pattern used in the Quanser robotics labs:
- the robot is armed/disarmed by joystick state,
- the downward camera is used as the sensing modality,
- forward kinematics are used to command body motion,
- a proportional steering controller keeps the robot centered on a line.

This is intentionally positioned as the "driver" phase before full LiDAR SLAM.
The line follower provides dependable real-time motion while the map-building
stack is still being developed or used as a supervisory layer.
"""

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from qbot_helpers import QBotHardwareInterface, QBotKinematics


class LineFollower:
    """Downward-camera line-following controller for a differential-drive QBot."""

    def __init__(self,
                 line_color: Tuple[int, int, int] = (255, 255, 255),
                 target_speed: float = 0.18,
                 max_turn_rate: float = 0.8,
                 kp: float = 1.4,
                 roi_fraction: float = 0.45):
        self.line_color = np.array(line_color, dtype=np.uint8)
        self.target_speed = float(target_speed)
        self.max_turn_rate = float(max_turn_rate)
        self.kp = float(kp)
        self.roi_fraction = float(roi_fraction)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            return None

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Keep the lower region of the image where the line is typically visible.
        h = gray.shape[0]
        roi_h = max(1, int(h * self.roi_fraction))
        roi = gray[h - roi_h:h, :]

        # High-contrast thresholding for white or bright line markers.
        _, binary = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY)
        return binary

    def compute_line_error(self, image: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        """Returns normalized lateral error in [-1, 1] and the centroid if available."""
        binary = self._preprocess(image)
        if binary is None:
            return 0.0, None

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, None

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < 30:
            return 0.0, None

        moments = cv2.moments(contour)
        if moments["m00"] < 1e-6:
            return 0.0, None

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        centroid = np.array([cx, cy])

        # Normalize centroid position around image width so error is in [-1, 1].
        width = binary.shape[1]
        error = ((float(cx) / max(width, 1)) * 2.0) - 1.0
        return error, centroid

    def controller_step(self, image: np.ndarray) -> Tuple[float, float]:
        """Produces [v, omega] body commands for a QBot in mode 1."""
        error, centroid = self.compute_line_error(image)

        # If no line is visible, slow down and rotate in place to recover.
        if centroid is None:
            return 0.05, 0.6

        omega = -self.kp * error
        omega = float(np.clip(omega, -self.max_turn_rate, self.max_turn_rate))
        v = self.target_speed
        return v, omega


def run_line_following_loop():
    """Run the line-follower on hardware or simulation using the Quanser-style setup."""
    robot = QBotHardwareInterface(
        ip_driver="localhost",
        mode=1,
        enable_lidar=False,
        enable_realsense=False,
        enable_downward_cam=True
    )

    follower = LineFollower(target_speed=0.18, max_turn_rate=0.8, kp=1.4)
    robot.set_led(0.0, 1.0, 0.0)
    time.sleep(0.5)

    try:
        while True:
            sensors = robot.step(read_cameras=True, read_lidar=False)

            if robot.check_emergency_stop():
                break

            image = sensors.downward_image
            if image is not None:
                v_cmd, omega_cmd = follower.controller_step(image)
                robot.set_body_velocity(v_cmd, omega_cmd)
                print(f"line_follow -> v={v_cmd:.3f} m/s, omega={omega_cmd:.3f} rad/s")
            else:
                robot.set_body_velocity(0.0, 0.0)

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("[line_following] Interrupted by user.")
    finally:
        robot.set_body_velocity(0.0, 0.0)
        robot.close()


if __name__ == "__main__":
    run_line_following_loop()
