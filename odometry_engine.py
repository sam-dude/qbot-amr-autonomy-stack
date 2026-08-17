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

import time
import numpy as np
from typing import Tuple, Callable, Optional, List, Sequence


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

        # IMU bias estimation to suppress heading drift when the robot is stationary.
        self.gyro_bias = 0.0
        self.gyro_bias_ready = False
        self.motion_threshold = 1e-4
        self.stationary_gyro_threshold = 0.01

        # Covariance-based fusion weights. By default the IMU is trusted more for
        # heading changes, while encoder motion remains the main source of distance.
        self.encoder_variance = 0.015
        self.gyro_variance = 0.0025
        self.heading_variance = 0.05

        # Lightweight EKF-style covariance tracking for the fused pose estimate.
        self.P = np.diag([1e-3, 1e-3, 1e-2])
        self.Q = np.diag([1e-6, 1e-6, 1e-5])
        self.R = np.diag([self.encoder_variance, self.gyro_variance, self.heading_variance])

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

    def calibrate_gyro_bias(self, gyro_samples: Sequence[float]) -> float:
        """Estimate the yaw-rate bias from a stationary calibration sample set."""
        if len(gyro_samples) == 0:
            return self.gyro_bias

        samples = np.asarray(gyro_samples, dtype=float)
        self.gyro_bias = float(np.mean(samples))
        self.gyro_bias_ready = True
        self.heading_variance = float(np.var(samples)) + 1e-6
        return self.gyro_bias

    def calibrate_stationary_gyro_bias(
        self,
        sensor_reader: Callable[[], object],
        max_wait_s: float = 3.0,
        wheel_threshold: float = 1e-5,
        gyro_threshold: float = 0.01,
        min_samples: int = 25
    ) -> float:
        """Wait until the robot is still, then estimate the gyro bias from a quiet window."""
        samples = []
        prev_wheel = None
        start = time.monotonic()

        while time.monotonic() - start < max_wait_s:
            sample = sensor_reader()
            if sample is None:
                continue

            wheel = np.asarray(getattr(sample, 'wheel_positions', np.zeros(2)), dtype=float)
            gyro = float(np.asarray(getattr(sample, 'gyroscope', np.zeros(3)), dtype=float)[2])

            if prev_wheel is not None:
                delta_wheel = np.max(np.abs(wheel - prev_wheel))
            else:
                delta_wheel = 0.0

            if delta_wheel <= wheel_threshold and abs(gyro) <= gyro_threshold:
                samples.append(gyro)
                if len(samples) >= min_samples:
                    break
            else:
                samples = []

            prev_wheel = wheel
            time.sleep(0.01)

        if len(samples) == 0:
            self.gyro_bias = 0.0
            return self.gyro_bias

        self.calibrate_gyro_bias(samples)
        return self.gyro_bias

    def fuse_compass_heading(self, compass_heading_rad: Optional[float], compass_variance: float = 0.05) -> float:
        """Optional absolute heading correction when a magnetic compass is present."""
        if compass_heading_rad is None:
            return self.theta_fused

        heading_error = self.wrap_to_pi(compass_heading_rad - self.theta_fused)
        correction_gain = self.heading_variance / (self.heading_variance + compass_variance)
        self.theta_fused = self.wrap_to_pi(self.theta_fused + correction_gain * heading_error)
        return self.theta_fused

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
        # 1. Correct the raw gyro measurement for slow bias drift before fusion.
        corrected_gyro = gyro_yaw_rate - self.gyro_bias
        delta_theta_gyro = corrected_gyro * dt

        # 2. Use a covariance-based blend so the estimate behaves more like a
        #    practical EKF-style complementary filter: when motion is large, the
        #    encoder is trusted more; during turns or higher yaw-rate, the gyro gets
        #    larger weight.
        delta_speed_magnitude = abs(delta_s)
        speed_term = np.clip(delta_speed_magnitude / 0.02, 0.1, 1.0)
        encoder_weight = np.clip(alpha * speed_term, 0.25, 0.9)
        gyro_weight = 1.0 - encoder_weight

        encoder_var = self.encoder_variance * (1.0 + speed_term)
        gyro_var = self.gyro_variance + 0.05 * abs(corrected_gyro)
        if encoder_var + gyro_var > 0.0:
            encoder_weight = encoder_var / (encoder_var + gyro_var)
            gyro_weight = 1.0 - encoder_weight

        delta_theta_fused = encoder_weight * delta_theta_enc + gyro_weight * delta_theta_gyro

        # 3. EKF-style covariance update: increase uncertainty as motion grows,
        #    then weight the fused heading update by the measurement covariance.
        self.P += self.Q
        self.P[2, 2] = max(self.P[2, 2], abs(delta_theta_fused) * 0.5 + 1e-4)

        theta_mid = self.theta_fused + delta_theta_fused / 2
        self.x_fused += delta_s * np.cos(theta_mid)
        self.y_fused += delta_s * np.sin(theta_mid)
        self.theta_fused = self.wrap_to_pi(self.theta_fused + delta_theta_fused)
        self.P[0, 0] += max(self.encoder_variance * abs(delta_s), 1e-6)
        self.P[1, 1] += max(self.encoder_variance * abs(delta_s), 1e-6)
        self.P[2, 2] += max(self.gyro_variance * abs(delta_theta_fused), 1e-6)

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

        # 2. Reject tiny residual drift as non-motion. This prevents encoder noise
        #    from manufacturing fake displacement when the robot is stationary or
        #    effectively stalled.
        tiny_wheel_motion = (
            abs(d_theta_L) <= self.motion_threshold and
            abs(d_theta_R) <= self.motion_threshold and
            abs(gyro_yaw_rate - self.gyro_bias) <= self.stationary_gyro_threshold
        )

        if tiny_wheel_motion:
            self.gyro_bias = 0.99 * self.gyro_bias + 0.01 * gyro_yaw_rate
            return {
                'euler': np.array([self.x_euler, self.y_euler, self.theta_euler]),
                'rk2': np.array([self.x_rk2, self.y_rk2, self.theta_rk2]),
                'fused': np.array([self.x_fused, self.y_fused, self.theta_fused])
            }

        # 3. Convert angular delta to metric linear arc length
        delta_sL = self.r * d_theta_L
        delta_sR = self.r * d_theta_R
        delta_s = (delta_sR + delta_sL) / 2.0
        delta_theta_enc = (delta_sR - delta_sL) / self.L

        # 4. Update IMU bias estimate when the chassis is near standstill but the
        #    gyro still measures a small residual offset.
        if abs(delta_sL) <= self.r * self.motion_threshold and abs(delta_sR) <= self.r * self.motion_threshold:
            self.gyro_bias = 0.95 * self.gyro_bias + 0.05 * gyro_yaw_rate

        # 5. Update all 3 state estimators
        self.update_pose_euler(delta_sL, delta_sR)
        self.update_pose_rk2(delta_sL, delta_sR)
        self.fuse_heading_complementary(delta_s, delta_theta_enc, gyro_yaw_rate, dt, alpha)

        # 6. Save history for trajectory evaluation
        self.history_euler.append([self.x_euler, self.y_euler, self.theta_euler])
        self.history_rk2.append([self.x_rk2, self.y_rk2, self.theta_rk2])
        self.history_fused.append([self.x_fused, self.y_fused, self.theta_fused])

        return {
            'euler': np.array([self.x_euler, self.y_euler, self.theta_euler]),
            'rk2': np.array([self.x_rk2, self.y_rk2, self.theta_rk2]),
            'fused': np.array([self.x_fused, self.y_fused, self.theta_fused])
        }
