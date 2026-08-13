"""
odometry_engine.py — Core Differential Drive Odometry & Heading Fusion Engine

You will implement the core mathematical logic in the TODO sections below:
  1. wrap_to_pi()                 — Angle normalization to [-pi, +pi]
  2. update_pose_euler()          — 1st-Order Forward Euler Integration
  3. update_pose_rk2()            — 2nd-Order Runge-Kutta (Midpoint) Integration
  4. fuse_heading_complementary() — IMU Gyroscope + Encoder Heading Blending

Robot Physical Constants:
  - Wheel Radius (r) = 0.04445 m  (3.5 inches / 2)
  - Wheelbase / Track (L) = 0.3928 m
"""

import numpy as np
from typing import Tuple


class OdometryEngine:
    """
    Computes dead reckoning pose [x, y, theta] from wheel encoder positions
    and 6-DOF IMU gyroscope measurements.
    """

    def __init__(
        self,
        wheel_radius: float = 0.04445,
        wheelbase: float = 0.3928,
        init_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ):
        self.r = float(wheel_radius)
        self.L = float(wheelbase)

        # State estimates [x (m), y (m), theta (rad)]
        self.x_euler = float(init_pose[0])
        self.y_euler = float(init_pose[1])
        self.theta_euler = float(init_pose[2])

        self.x_rk2 = float(init_pose[0])
        self.y_rk2 = float(init_pose[1])
        self.theta_rk2 = float(init_pose[2])

        self.x_fused = float(init_pose[0])
        self.y_fused = float(init_pose[1])
        self.theta_fused = float(init_pose[2])

        # Previous encoder readings (cumulative angle in radians)
        self.prev_pos_left = None
        self.prev_pos_right = None

        # Trajectory history for analysis and plotting
        self.history_euler = []
        self.history_rk2 = []
        self.history_fused = []

    # =========================================================================
    # TODO 1: Angle Normalization
    # =========================================================================
    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        """
        Normalizes an angle to the range [-pi, +pi].

        Mathematical hint:
            (angle + pi) % (2 * pi) - pi
            or using np.arctan2(np.sin(angle), np.cos(angle))

        Args:
            angle: Input angle in radians.

        Returns:
            Normalized angle in radians in range [-pi, +pi].
        """
        # =====================================================================
        # >>>>> YOUR CODE HERE <<<<<
        return np.arctan2(np.sin(angle), np.cos(angle))
        # =====================================================================

    # =========================================================================
    # TODO 2: Forward Euler Integration (1st-Order)
    # =========================================================================
    def update_pose_euler(self, delta_sL: float, delta_sR: float) -> Tuple[float, float, float]:
        """
        Updates robot pose using Forward Euler integration.

        Math:
            delta_s = (delta_sR + delta_sL) / 2
            delta_theta = (delta_sR - delta_sL) / L

            x_{k+1} = x_k + delta_s * cos(theta_k)
            y_{k+1} = y_k + delta_s * sin(theta_k)
            theta_{k+1} = wrap_to_pi(theta_k + delta_theta)

        Notice: Euler uses theta at the BEGINNING of the interval (theta_k).
        On curved paths, this accumulates truncation error (drifts outward).

        Args:
            delta_sL: Distance traveled by left wheel in this step (meters).
            delta_sR: Distance traveled by right wheel in this step (meters).

        Returns:
            (x_euler, y_euler, theta_euler)
        """
        # =====================================================================
        # >>>>> YOUR CODE HERE <<<<<
        # 1. Compute delta_s and delta_theta:
        delta_s = 0.0
        delta_theta = 0.0

        delta_s = (delta_sR + delta_sL) / 2
        delta_theta = (delta_sR - delta_sL) / self.L

        # 2. Update self.x_euler, self.y_euler, self.theta_euler:
        self.x_euler += delta_s * np.cos(self.theta_euler)
        self.y_euler += delta_s * np.sin(self.theta_euler)
        self.theta_euler = self.wrap_to_pi(self.theta_euler + delta_theta)

        # =====================================================================
        return self.x_euler, self.y_euler, self.theta_euler

    # =========================================================================
    # TODO 3: Runge-Kutta 2nd-Order / Midpoint Integration
    # =========================================================================
    def update_pose_rk2(self, delta_sL: float, delta_sR: float) -> Tuple[float, float, float]:
        """
        Updates robot pose using 2nd-Order Runge-Kutta (Midpoint) integration.

        Math:
            delta_s = (delta_sR + delta_sL) / 2
            delta_theta = (delta_sR - delta_sL) / L

            theta_mid = theta_k + delta_theta / 2
            x_{k+1} = x_k + delta_s * cos(theta_mid)
            y_{k+1} = y_k + delta_s * sin(theta_mid)
            theta_{k+1} = wrap_to_pi(theta_k + delta_theta)

        Notice: By projecting displacement along the AVERAGE heading (theta_mid),
        the local circle arc is accurately approximated, significantly reducing drift!

        Args:
            delta_sL: Distance traveled by left wheel in this step (meters).
            delta_sR: Distance traveled by right wheel in this step (meters).

        Returns:
            (x_rk2, y_rk2, theta_rk2)
        """
        # =====================================================================
        # >>>>> YOUR CODE HERE <<<<<
        # 1. Compute delta_s and delta_theta:
        delta_s = (delta_sR + delta_sL) / 2
        delta_theta = (delta_sR - delta_sL) / self.L

        # 2. Compute theta_mid:
        theta_mid = self.theta_rk2 + delta_theta / 2

        # 3. Update self.x_rk2, self.y_rk2, self.theta_rk2:
        self.x_rk2 += delta_s * np.cos(theta_mid)
        self.y_rk2 += delta_s * np.sin(theta_mid)
        self.theta_rk2 = self.wrap_to_pi(self.theta_rk2 + delta_theta)

        # =====================================================================
        return self.x_rk2, self.y_rk2, self.theta_rk2

    # =========================================================================
    # TODO 4: IMU Gyroscope Complementary Heading Fusion
    # =========================================================================
    def fuse_heading_complementary(
        self,
        delta_s: float,
        delta_theta_enc: float,
        gyro_yaw_rate: float,
        dt: float,
        alpha: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Fuses wheel encoder heading rate with IMU gyroscope yaw rate.

        Why:
          - Wheel encoders provide good steady-state heading but slip during fast turns.
          - Gyroscope measures angular velocity directly without wheel slip, but drifts over time.
          - Complementary blending gives high-frequency responsiveness + low drift!

        Math:
          delta_theta_gyro = gyro_yaw_rate * dt
          delta_theta_fused = alpha * delta_theta_enc + (1 - alpha) * delta_theta_gyro

          theta_mid = theta_fused_k + delta_theta_fused / 2
          x_{k+1} = x_k + delta_s * cos(theta_mid)
          y_{k+1} = y_k + delta_s * sin(theta_mid)
          theta_{k+1} = wrap_to_pi(theta_fused_k + delta_theta_fused)

        Args:
            delta_s: Linear arc displacement in meters.
            delta_theta_enc: Heading delta from wheel encoders (radians).
            gyro_yaw_rate: Gyroscope Z-axis reading in rad/s (sensors.gyroscope[2]).
            dt: Sample time interval in seconds.
            alpha: Filter weight (0.0 to 1.0). Higher alpha favors encoders; lower favors gyro.

        Returns:
            (x_fused, y_fused, theta_fused)
        """
        # =====================================================================
        # >>>>> YOUR CODE HERE <<<<<
        # 1. Compute delta_theta_gyro:
        delta_theta_gyro = gyro_yaw_rate * dt

        # 2. Blend encoder and gyro heading deltas:
        delta_theta_fused = alpha * delta_theta_enc + (1 - alpha) * delta_theta_gyro

        # 3. Update self.x_fused, self.y_fused, self.theta_fused using midpoint:
        theta_mid = self.theta_fused + delta_theta_fused / 2
        self.x_fused += delta_s * np.cos(theta_mid)
        self.y_fused += delta_s * np.sin(theta_mid)
        self.theta_fused = self.wrap_to_pi(self.theta_fused + delta_theta_fused)

        # =====================================================================
        return self.x_fused, self.y_fused, self.theta_fused

    # =========================================================================
    # Master Step Function
    # =========================================================================
    def step(
        self,
        wheel_positions: np.ndarray,
        gyro_yaw_rate: float,
        dt: float,
        alpha: float = 0.95
    ) -> dict:
        """
        Processes new sensor measurements and updates all estimators.

        Args:
            wheel_positions: array [pos_left, pos_right] in cumulative radians.
            gyro_yaw_rate: IMU Gyro yaw rate around Z in rad/s.
            dt: Sample delta time in seconds.
            alpha: Complementary filter weight.

        Returns:
            dict containing current poses: {'euler': [x, y, theta], 'rk2': [...], 'fused': [...]}
        """
        pos_L, pos_R = float(wheel_positions[0]), float(wheel_positions[1])

        # First call initialization
        if self.prev_pos_left is None:
            self.prev_pos_left = pos_L
            self.prev_pos_right = pos_R
            return {
                'euler': np.array([self.x_euler, self.y_euler, self.theta_euler]),
                'rk2': np.array([self.x_rk2, self.y_rk2, self.theta_rk2]),
                'fused': np.array([self.x_fused, self.y_fused, self.theta_fused])
            }

        # 1. Compute incremental wheel rotations (radians)
        d_theta_L = pos_L - self.prev_pos_left
        d_theta_R = pos_R - self.prev_pos_right
        self.prev_pos_left = pos_L
        self.prev_pos_right = pos_R

        # 2. Convert angular delta to metric linear arc length
        delta_sL = self.r * d_theta_L
        delta_sR = self.r * d_theta_R
        delta_s = (delta_sR + delta_sL) / 2.0
        delta_theta_enc = (delta_sR - delta_sL) / self.L

        # 3. Update all 3 state estimators
        self.update_pose_euler(delta_sL, delta_sR)
        self.update_pose_rk2(delta_sL, delta_sR)
        self.fuse_heading_complementary(delta_s, delta_theta_enc, gyro_yaw_rate, dt, alpha)

        # 4. Save history for trajectory evaluation
        self.history_euler.append([self.x_euler, self.y_euler, self.theta_euler])
        self.history_rk2.append([self.x_rk2, self.y_rk2, self.theta_rk2])
        self.history_fused.append([self.x_fused, self.y_fused, self.theta_fused])

        return {
            'euler': np.array([self.x_euler, self.y_euler, self.theta_euler]),
            'rk2': np.array([self.x_rk2, self.y_rk2, self.theta_rk2]),
            'fused': np.array([self.x_fused, self.y_fused, self.theta_fused])
        }
