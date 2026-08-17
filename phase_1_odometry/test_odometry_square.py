"""
phase_1_odometry/test_odometry_square.py

Quanser-style odometry benchmark designed to run on the physical QBot hardware.

Flow:
1. Start observer on local PC.
2. Run this script on the QBot.
3. The robot waits until armed.
4. When arm is active, reset the odometry state and run the square motion.
5. Stop on disarm or on finish.

This is the phase-1 hardware test before line following or SLAM.
"""

import csv
import time
import os
import numpy as np

from qbot_helpers import QBotHardwareInterface, QBotKinematics
from odometry_engine import OdometryEngine


class SquareMotionDriver:
    """Arm-gated square motion benchmark for hardware validation."""

    def __init__(self, robot, odom, side_length=1.0, speed_linear=0.20, speed_turn=0.50, log_csv_path=None):
        self.robot = robot
        self.odom = odom
        self.side_length = float(side_length)
        self.speed_linear = float(speed_linear)
        self.speed_turn = float(speed_turn)
        self.log_csv_path = log_csv_path
        self._log_file = None

    def _drive_duration(self, distance, speed):
        if abs(speed) < 1e-9:
            return 0.0
        return abs(distance) / abs(speed)

    def _open_log(self):
        if self.log_csv_path is None:
            return
        directory = os.path.dirname(self.log_csv_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        self._log_file = open(self.log_csv_path, 'w', newline='')
        writer = csv.writer(self._log_file)
        writer.writerow(['time_s', 'x_m', 'y_m', 'theta_rad', 'v_mps', 'omega_radps', 'wl_radps', 'wr_radps'])
        self._log_file.flush()

    def _close_log(self):
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def _log_state(self, t, v, omega, wl, wr):
        if self._log_file is None:
            return
        pose = self.odom.history_fused[-1] if self.odom.history_fused else [self.odom.x_fused, self.odom.y_fused, self.odom.theta_fused]
        writer = csv.writer(self._log_file)
        writer.writerow([t, pose[0], pose[1], pose[2], v, omega, wl, wr])
        self._log_file.flush()

    def _wait_for_arm(self, timeout_s=20.0):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            self.robot.step(read_cameras=False, read_lidar=False)
            if self.robot.armed:
                return True
            time.sleep(0.02)
        return False

    def _reset_pose(self):
        self.odom.x_euler = 0.0
        self.odom.y_euler = 0.0
        self.odom.theta_euler = 0.0
        self.odom.x_rk2 = 0.0
        self.odom.y_rk2 = 0.0
        self.odom.theta_rk2 = 0.0
        self.odom.x_fused = 0.0
        self.odom.y_fused = 0.0
        self.odom.theta_fused = 0.0
        self.odom.prev_pos_left = None
        self.odom.prev_pos_right = None
        self.odom.history_euler = []
        self.odom.history_rk2 = []
        self.odom.history_fused = []

    def _run_side(self, v, omega, duration):
        t0 = time.time()
        while time.time() - t0 < duration:
            sensors = self.robot.step(read_cameras=False, read_lidar=False)
            if self.robot.check_emergency_stop():
                return False
            if not self.robot.armed:
                self.robot.set_body_velocity(0.0, 0.0)
                return False
            self.odom.step(
                wheel_positions=sensors.wheel_positions,
                gyro_yaw_rate=sensors.gyroscope[2],
                dt=sensors.dt if sensors.dt > 0 else 0.02,
            )
            self.robot.set_body_velocity(v, omega)
            wl, wr = QBotKinematics.inverse_kinematics(v, omega)
            self._log_state(time.time() - self._start_t, v, omega, wl, wr)
            time.sleep(0.02)
        return True

    def _heading_turn_command(self, current_theta, target_theta, max_turn_rate=0.50, kp=1.2, tolerance=0.05):
        """Compute a heading-error-driven turn command for the next timestep."""
        heading_error = self.odom.wrap_to_pi(target_theta - current_theta)
        if abs(heading_error) <= tolerance:
            return 0.0, True

        omega = np.clip(kp * heading_error, -max_turn_rate, max_turn_rate)
        return float(omega), False

    def _turn_to_heading(self, target_theta, turn_rate=0.50, tolerance=0.05, kp=1.2):
        """Turn in place until the heading reaches the target using proportional heading error control."""
        while True:
            sensors = self.robot.step(read_cameras=False, read_lidar=False)
            if self.robot.check_emergency_stop():
                return False
            if not self.robot.armed:
                self.robot.set_body_velocity(0.0, 0.0)
                return False

            self.odom.step(
                wheel_positions=sensors.wheel_positions,
                gyro_yaw_rate=sensors.gyroscope[2],
                dt=sensors.dt if sensors.dt > 0 else 0.02,
            )

            current_theta = self.odom.theta_fused
            omega, done = self._heading_turn_command(
                current_theta=current_theta,
                target_theta=target_theta,
                max_turn_rate=turn_rate,
                kp=kp,
                tolerance=tolerance,
            )

            if done:
                self.robot.set_body_velocity(0.0, 0.0)
                return True

            self.robot.set_body_velocity(0.0, omega)
            wl, wr = QBotKinematics.inverse_kinematics(0.0, omega)
            self._log_state(time.time() - self._start_t, 0.0, omega, wl, wr)
            time.sleep(0.02)

    def run_square(self):
        # arm-gated sequence: robot waits for the joystick to arm it
        print("[square] Waiting for arm command. Hold left trigger/button to arm.")
        if not self._wait_for_arm():
            print("[square] Arm timeout: robot never armed.")
            return False

        print("[square] Robot armed. Calibrating and starting square benchmark.")
        self.robot.set_led(0.0, 1.0, 0.0)
        self._start_t = time.time()
        self._open_log()

        def read_stationary_sample():
            return self.robot.step(read_cameras=False, read_lidar=False)

        self.odom.calibrate_stationary_gyro_bias(
            sensor_reader=read_stationary_sample,
            max_wait_s=3.0,
            wheel_threshold=1e-5,
            gyro_threshold=0.01,
            min_samples=25,
        )

        self._reset_pose()
        self.robot.set_body_velocity(0.0, 0.0)

        side_duration = self._drive_duration(self.side_length, self.speed_linear)

        for side_id in range(4):
            if not self.robot.armed:
                print("[square] Disarmed during square motion.")
                break

            print(f"[square] Side {side_id + 1}/4: driving straight")
            if not self._run_side(self.speed_linear, 0.0, side_duration):
                break

            if side_id < 3:
                current_theta = self.odom.theta_fused
                target_theta = self.odom.wrap_to_pi(current_theta + np.pi / 2.0)
                print(f"[square] Side {side_id + 1}/4: turning 90 degrees")
                if not self._turn_to_heading(target_theta, turn_rate=self.speed_turn, tolerance=0.05):
                    break

        self.robot.set_body_velocity(0.0, 0.0)
        print("[square] Completed square motion.")
        self._close_log()
        return True


if __name__ == "__main__":
    robot = QBotHardwareInterface(
        ip_driver="localhost",
        mode=1,
        enable_lidar=False,
        enable_realsense=False,
        enable_downward_cam=False,
    )
    odom = OdometryEngine(init_pose=(0.0, 0.0, 0.0))
    log_path = os.path.join(os.path.dirname(__file__), 'square_run.csv')
    driver = SquareMotionDriver(robot, odom, log_csv_path=log_path)
    try:
        driver.run_square()
    finally:
        robot.set_body_velocity(0.0, 0.0)
        robot.close()
