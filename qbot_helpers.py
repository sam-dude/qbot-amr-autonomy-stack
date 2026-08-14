"""
qbot_helpers.py - Modular Helper Framework for Quanser QBot Platform
Supports both Digital Twin (QLabs) and Physical Hardware execution.
"""

import os
import sys
import time
import atexit
import platform
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Tuple

# Quanser PAL and Device drivers
from pal.products.qbot_platform import (
    QBotPlatformDriver,
    QBotPlatformLidar,
    QBotPlatformRealSense,
    QBotPlatformCSICamera,
    IS_PHYSICAL_QBOTPLATFORM
)
from pal.utilities.gamepad import LogitechF710
from quanser.hardware import HILError

@dataclass
class QBotSensors:
    """Synchronized sensor snapshot from QBot Platform."""
    timestamp: float = 0.0
    dt: float = 0.0
    
    # Wheel Odometry Raw Data
    wheel_positions: np.ndarray = field(default_factory=lambda: np.zeros(2))  # [rad_left, rad_right]
    wheel_speeds: np.ndarray = field(default_factory=lambda: np.zeros(2))     # [rad/s_left, rad/s_right]
    
    # 6-DOF IMU
    accelerometer: np.ndarray = field(default_factory=lambda: np.zeros(3))    # [ax, ay, az] in m/s^2
    gyroscope: np.ndarray = field(default_factory=lambda: np.zeros(3))        # [wx, wy, wz] in rad/s (wz is yaw rate)
    
    # Diagnostics
    currents: np.ndarray = field(default_factory=lambda: np.zeros(2))         # Motor currents [A]
    battery_voltage: float = 0.0                                             # Volts
    
    # Perception Streams (Optional / rate-limited)
    lidar_distances: Optional[np.ndarray] = None                             # shape (1680, 1) or (384, 1) [m]
    lidar_angles: Optional[np.ndarray] = None                                # shape (1680, 1) [rad]
    realsense_rgb: Optional[np.ndarray] = None                               # (480, 640, 3) BGR uint8
    realsense_depth: Optional[np.ndarray] = None                             # (480, 640) in meters (float32)
    downward_image: Optional[np.ndarray] = None                              # (400, 640) grayscale uint8


class QBotKinematics:
    """Differential Drive Kinematic Calculations for QBot Platform."""
    
    WHEEL_RADIUS = 0.04445   # meters (3.5 inch diameter / 2)
    WHEEL_BASE   = 0.3928    # meters (distance between wheels)
    LIDAR_OFFSET_X = 0.22225 # meters (LIDAR is mounted 8.75 inches forward of axle)

    @classmethod
    def inverse_kinematics(cls, v: float, omega: float) -> Tuple[float, float]:
        """
        Converts desired body speeds [v (m/s), omega (rad/s)] to wheel speeds [w_L, w_R (rad/s)].
        """
        w_L = (v - (omega * cls.WHEEL_BASE / 2.0)) / cls.WHEEL_RADIUS
        w_R = (v + (omega * cls.WHEEL_BASE / 2.0)) / cls.WHEEL_RADIUS
        return w_L, w_R

    @classmethod
    def forward_velocity_kinematics(cls, w_L: float, w_R: float) -> Tuple[float, float]:
        """
        Converts wheel speeds [w_L, w_R (rad/s)] to linear velocity v (m/s) and angular velocity omega (rad/s).
        """
        v = cls.WHEEL_RADIUS * (w_R + w_L) / 2.0
        omega = cls.WHEEL_RADIUS * (w_R - w_L) / cls.WHEEL_BASE
        return v, omega


class QBotHardwareInterface:
    """
    High-level manager for QBot Platform hardware/simulation I/O.
    Abstracts driver initialization, sensor reading, motor control, and LED feedback.
    """

    def __init__(
        self,
        ip_driver: str = "localhost",
        mode: int = 1, # Mode 1: Body velocity [v, w], Mode 2: Wheel velocity [wL, wR]
        enable_lidar: bool = True,
        enable_realsense: bool = True,
        enable_downward_cam: bool = False
    ):
        self.ip_driver = ip_driver
        self.mode = mode
        self.enable_lidar = enable_lidar
        self.enable_realsense = enable_realsense
        self.enable_downward_cam = enable_downward_cam

        print(f"[QBotInterface] Running on {'Physical Hardware' if IS_PHYSICAL_QBOTPLATFORM else 'Digital Twin / Simulation'}")

        if IS_PHYSICAL_QBOTPLATFORM:
            self._load_platform_driver()

        # 1. Base platform driver (Motors, Encoders, IMU, Battery)
        self.driver = QBotPlatformDriver(mode=self.mode, ip=self.ip_driver)

        # 2. Sensors
        self.lidar = QBotPlatformLidar(numMeasurements=1680) if enable_lidar else None
        self.realsense = QBotPlatformRealSense(mode='RGB&DEPTH') if enable_realsense else None
        self.down_cam = QBotPlatformCSICamera(exposure=10) if enable_downward_cam else None
        self.gamepad = LogitechF710(1)

        self.start_time = time.time()
        self.prev_time = self.start_time
        self.sensor_data = QBotSensors()
        
        # Arming & Control state
        # Match the Quanser hardware flow: the robot starts disarmed until the
        # user presses and holds the left trigger/button to arm it.
        self.armed = False
        self._left_press_seen = False
        self._right_press_seen = False
        self.command = np.zeros(2, dtype=np.float64)
        self.led_color = [0.0, 1.0, 0.0]  # Green by default

    def _load_platform_driver(self):
        """Load the physical QBot platform driver before other hardware objects are created."""
        print("[QBotInterface] Loading physical QBot driver...")
        os.system('quarc_run -q -Q -t tcpip://localhost:17000 *.rt-linux_qbot_platform -d /tmp')
        time.sleep(5)
        os.system('quarc_run -r -t tcpip://localhost:17000 qbot_platform_driver_physical.rt-linux_qbot_platform -d /tmp -uri tcpip://localhost:17099')
        time.sleep(3)
        print("[QBotInterface] Driver loaded")

        # Auto-shutdown on process exit
        atexit.register(self.close)

    def set_led(self, r: float, g: float, b: float):
        """Set user LED color (0.0 to 1.0 per channel)."""
        self.led_color = [float(r), float(g), float(b)]

    def set_body_velocity(self, v_linear: float, w_angular: float):
        """Command body velocities in Mode 1: v (m/s), w (rad/s)."""
        if self.mode == 1:
            self.command = np.array([v_linear, w_angular], dtype=np.float64)
        elif self.mode == 2:
            wL, wR = QBotKinematics.inverse_kinematics(v_linear, w_angular)
            self.command = np.array([wL, wR], dtype=np.float64)

    def set_wheel_speeds(self, w_left: float, w_right: float):
        """Command individual wheel speeds in Mode 2 (rad/s)."""
        if self.mode == 2:
            self.command = np.array([w_left, w_right], dtype=np.float64)
        elif self.mode == 1:
            v, w = QBotKinematics.forward_velocity_kinematics(w_left, w_right)
            self.command = np.array([v, w], dtype=np.float64)

    def step(self, read_cameras: bool = True, read_lidar: bool = True) -> QBotSensors:
        """
        Executes one control cycle: writes motor commands and reads all sensor streams.
        """
        curr_time = time.time()
        dt = curr_time - self.prev_time
        self.prev_time = curr_time
        timestamp = curr_time - self.start_time

        self._update_gamepad_state()

        # Send command to hardware / simulation
        new_data = self.driver.read_write_std(
            timestamp=timestamp,
            arm=1 if self.armed else 0,
            commands=self.command,
            userLED=True,
            color=self.led_color
        )

        if new_data:
            self.sensor_data.timestamp = timestamp
            self.sensor_data.dt = dt
            self.sensor_data.wheel_positions = np.copy(self.driver.wheelPositions)
            self.sensor_data.wheel_speeds = np.copy(self.driver.wheelSpeeds)
            self.sensor_data.accelerometer = np.copy(self.driver.accelerometer)
            self.sensor_data.gyroscope = np.copy(self.driver.gyroscope)
            self.sensor_data.currents = np.copy(self.driver.currents)
            self.sensor_data.battery_voltage = float(np.ravel(self.driver.battVoltage)[0])

        # Optional sensor reads (can be throttled to save CPU)
        if self.lidar and read_lidar:
            if self.lidar.read():
                self.sensor_data.lidar_distances = np.copy(self.lidar.distances)
                self.sensor_data.lidar_angles = np.copy(self.lidar.angles)

        if self.realsense and read_cameras:
            if self.realsense.read_RGB() != -1:
                self.sensor_data.realsense_rgb = np.copy(self.realsense.imageBufferRGB)
            if self.realsense.read_depth(dataMode='M') != -1:
                self.sensor_data.realsense_depth = np.copy(self.realsense.imageBufferDepthM)

        if self.down_cam and read_cameras:
            if self.down_cam.read():
                self.sensor_data.downward_image = np.copy(self.down_cam.imageData)

        return self.sensor_data

    def _update_gamepad_state(self):
        """Read the LogitechF710 controller state and update arming / emergency-stop state."""
        if not hasattr(self, 'gamepad') or self.gamepad is None:
            return
        try:
            new_data = self.gamepad.read()
            if not new_data:
                return

            left_pressed = bool(getattr(self.gamepad, 'buttonLeft', False))
            right_pressed = bool(getattr(self.gamepad, 'buttonRight', False))

            if right_pressed:
                self.armed = False
                self._left_press_seen = False
                self._right_press_seen = True
                self.set_body_velocity(0.0, 0.0)
                self.set_led(1.0, 0.0, 0.0)
                print("\n[EMERGENCY STOP TRIGGERED] Operator halted motors with RB!")
                return

            # One-click arm latch: the robot becomes armed on the first LB press,
            # and stays armed until RB emergency stop is pressed. We do not disarm on release.
            if left_pressed and not self._left_press_seen:
                self.armed = True
                self._left_press_seen = True
                self.set_led(0.0, 1.0, 0.0)
                print("[QBotInterface] Armed by left trigger/button press.")
            elif not left_pressed:
                self._left_press_seen = False

        except Exception:
            pass

    def check_emergency_stop(self) -> bool:
        """Check the gamepad arm/emergency-stop state using the Quanser lab controller pattern."""
        self._update_gamepad_state()
        if not self.armed:
            self.set_body_velocity(0.0, 0.0)
            return True
        return False

    def close(self):
        """Safely stops motors and releases all sensor handles."""
        if getattr(self, '_closed', False):
            return
        self._closed = True
        print("[QBotInterface] Shutting down hardware streams...")
        try:
            self.command = np.zeros(2, dtype=np.float64)
            for _ in range(3):
                self.driver.read_write_std(timestamp=0, arm=0, commands=np.zeros(2), userLED=False)
                time.sleep(0.01)
        except Exception:
            pass
        try:
            self.driver.terminate()
        except Exception:
            pass
        if self.lidar:
            try: self.lidar.terminate()
            except Exception: pass
        if self.realsense:
            try: self.realsense.terminate()
            except Exception: pass
        if self.down_cam:
            try: self.down_cam.terminate()
            except Exception: pass
        if hasattr(self, 'gamepad') and self.gamepad is not None:
            try: self.gamepad.terminate()
            except Exception: pass
        try:
            cv2.destroyAllWindows()
        except Exception: pass


class QBotDashboard:
    """
    Lightweight OpenCV-based HUD to inspect all sensors, odometry trajectory, and diagnostics in real time.
    """
    def __init__(self, window_name: str = "QBot Platform Telemetry HUD"):
        self.window_name = window_name
        self.trajectory_map = np.zeros((400, 400, 3), dtype=np.uint8)
        self.map_scale = 50.0  # pixels per meter
        self.map_center = np.array([200, 200])

    def update_trajectory(self, x: float, y: float, theta: float):
        """Draws robot pose on a 2D local path map."""
        px = int(self.map_center[0] + x * self.map_scale)
        py = int(self.map_center[1] - y * self.map_scale)
        
        # Keep within bounds
        px = np.clip(px, 0, 399)
        py = np.clip(py, 0, 399)
        cv2.circle(self.trajectory_map, (px, py), 1, (0, 255, 255), -1)

    def render(self, sensors: QBotSensors, pose: Tuple[float, float, float]):
        """Renders a combined 2x2 or side-by-side dashboard."""
        x, y, theta = pose
        self.update_trajectory(x, y, theta)
        
        # Dashboard canvas
        hud = np.zeros((480, 800, 3), dtype=np.uint8)
        
        # Left side: Camera feed or Trajectory Map
        if sensors.realsense_rgb is not None:
            rgb_small = cv2.resize(sensors.realsense_rgb, (360, 240))
            hud[0:240, 0:360] = rgb_small
        
        traj_small = cv2.resize(self.trajectory_map, (360, 240))
        hud[240:480, 0:360] = traj_small

        # Right side: Sensor telemetry text
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        lines = [
            f"--- QBOT PLATFORM TELEMETRY ---",
            f"Time: {sensors.timestamp:.2f} s | dt: {sensors.dt*1000:.1f} ms",
            f"",
            f"Odometry Pose:",
            f"  X: {x:+.3f} m",
            f"  Y: {y:+.3f} m",
            f"  Theta: {np.degrees(theta):+.1f} deg",
            f"",
            f"Wheel Pos [rad]: L={sensors.wheel_positions[0]:.2f}, R={sensors.wheel_positions[1]:.2f}",
            f"Wheel Spd [rad/s]: L={sensors.wheel_speeds[0]:.2f}, R={sensors.wheel_speeds[1]:.2f}",
            f"",
            f"IMU Gyro (Yaw Rate): {sensors.gyroscope[2]:+.3f} rad/s",
            f"IMU Accel (Forward X): {sensors.accelerometer[0]:+.3f} m/s^2",
            f"",
            f"Battery: {sensors.battery_voltage:.2f} V",
            f"Motor Currents: L={sensors.currents[0]:.2f}A, R={sensors.currents[1]:.2f}A"
        ]

        y_offset = 30
        for line in lines:
            cv2.putText(hud, line, (380, y_offset), font, 0.45, color, 1, cv2.LINE_AA)
            y_offset += 22

        cv2.imshow(self.window_name, hud)
        return cv2.waitKey(1) & 0xFF
