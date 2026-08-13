"""
test_odometry.py — Benchmark & Verification Test for QBot Odometry Engine

Features:
  - Supports QLabs Digital Twin simulation AND Physical Hardware.
  - Commands the QBot to drive a calibrated 1.0m x 1.0m square trajectory.
  - Concurrently evaluates 3 estimators:
      1. Forward Euler Odometry
      2. 2nd-Order Runge-Kutta (Midpoint) Odometry
      3. Complementary Gyro-Encoder Fused Odometry
  - Renders live telemetry HUD.
  - Automatically calculates Loop Closure Error (LCE) at the end and saves
    a high-resolution comparison plot to 'docs/figures/odometry_benchmark.png'.

Usage:
    python test_odometry.py
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from qbot_helpers import (
    QBotHardwareInterface,
    QBotDashboard,
    QBotKinematics,
    IS_PHYSICAL_QBOTPLATFORM
)
from odometry_engine import OdometryEngine

# If running in simulation, set up QLabs world
if not IS_PHYSICAL_QBOTPLATFORM:
    try:
        from qlabs_setup import setup
        print("[test_odometry] Initializing QLabs Digital Twin arena...")
        setup(locationQBotP=[0.0, 0.0, 0.05], rotationQBotP=[0, 0, 0], verbose=True)
        time.sleep(1.0)
    except Exception as e:
        print(f"[test_odometry] Note: Simulation setup failed or QLabs not running: {e}")


def drive_timed(robot: QBotHardwareInterface, odom: OdometryEngine, dashboard: QBotDashboard,
                v: float, omega: float, duration: float, dt_target: float = 0.016) -> bool:
    """
    Drives the robot at [v, omega] for a specified duration while updating odometry.
    Returns False if stopped early, True if completed normally.
    """
    t_start = time.time()
    robot.set_body_velocity(v, omega)

    while (time.time() - t_start) < duration:
        t_loop_start = time.time()

        # Step 1: Read all sensors & send motor commands
        sensors = robot.step(read_cameras=False, read_lidar=False)

        # Step 2: Emergency stop check
        if robot.check_emergency_stop():
            return False

        # Step 3: Update odometry estimators
        poses = odom.step(
            wheel_positions=sensors.wheel_positions,
            gyro_yaw_rate=sensors.gyroscope[2],
            dt=sensors.dt if sensors.dt > 0 else dt_target
        )

        # Step 4: Render HUD with RK2 pose
        key = dashboard.render(sensors=sensors, pose=poses['rk2'])
        if key == 27 or key == ord('q'):  # ESC or 'q' in HUD window
            print("\n[EMERGENCY STOP TRIGGERED] Operator requested stop from GUI!")
            robot.armed = False
            robot.set_body_velocity(0.0, 0.0)
            return False

        # Maintain control loop rate
        elapsed = time.time() - t_loop_start
        if elapsed < dt_target:
            time.sleep(dt_target - elapsed)

    return True


def run_square_benchmark(side_length: float = 1.0, speed_linear: float = 0.2, speed_turn: float = 0.5):
    """
    Executes a 4-side square trajectory and evaluates odometry performance.
    """
    robot = QBotHardwareInterface(
        ip_driver="localhost",
        mode=1,  # Body velocity mode [v, omega]
        enable_lidar=False,
        enable_realsense=False,
        enable_downward_cam=False
    )
    odom = OdometryEngine(init_pose=(0.0, 0.0, 0.0))
    dashboard = QBotDashboard(window_name="QBot Odometry Benchmark HUD")

    # Time calculations for forward driving and 90-degree turning
    drive_duration = side_length / speed_linear
    turn_duration = (np.pi / 2.0) / speed_turn

    print("\n" + "=" * 60)
    print("  QBOT ODOMETRY BENCHMARK: 1.0m x 1.0m Square Test")
    print(f"  Side Length: {side_length:.1f} m  | Forward Speed: {speed_linear:.2f} m/s")
    print(f"  Turn Speed:  {speed_turn:.2f} rad/s (90 deg per corner)")
    print("=" * 60 + "\n")

    robot.set_led(0.0, 1.0, 0.0)  # Green LED
    time.sleep(1.0)

    try:
        for side in range(4):
            print(f"[Side {side + 1}/4] Driving forward {side_length:.1f} m...")
            if not drive_timed(robot, odom, dashboard, v=speed_linear, omega=0.0, duration=drive_duration):
                print("[test_odometry] Benchmark stopped early.")
                break

            # Settle briefly before turning
            drive_timed(robot, odom, dashboard, v=0.0, omega=0.0, duration=0.5)

            if side < 3:  # Don't need to turn after the 4th side completes the loop
                print(f"[Corner {side + 1}/3] Turning 90 degrees CCW...")
                if not drive_timed(robot, odom, dashboard, v=0.0, omega=speed_turn, duration=turn_duration):
                    print("[test_odometry] Benchmark stopped early.")
                    break
                drive_timed(robot, odom, dashboard, v=0.0, omega=0.0, duration=0.5)

        # Stop robot completely
        robot.set_body_velocity(0.0, 0.0)
        drive_timed(robot, odom, dashboard, v=0.0, omega=0.0, duration=0.5)

    except KeyboardInterrupt:
        print("\n[test_odometry] User interrupted test.")

    finally:
        robot.close()

    # =========================================================================
    # Quantitative Trajectory Evaluation & Comparison Plot
    # =========================================================================
    hist_euler = np.array(odom.history_euler)
    hist_rk2 = np.array(odom.history_rk2)
    hist_fused = np.array(odom.history_fused)

    if len(hist_rk2) == 0:
        print("[test_odometry] No odometry data recorded.")
        return

    # Loop closure errors (distance from final position back to origin [0, 0])
    lce_euler = np.linalg.norm(hist_euler[-1, 0:2])
    lce_rk2 = np.linalg.norm(hist_rk2[-1, 0:2])
    lce_fused = np.linalg.norm(hist_fused[-1, 0:2])

    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS & LOOP CLOSURE ERROR (LCE)")
    print("=" * 60)
    print(f"  1. Forward Euler Final Pose:  X={hist_euler[-1,0]:+.3f}m, Y={hist_euler[-1,1]:+.3f}m, LCE = {lce_euler*100:.1f} cm")
    print(f"  2. RK2 (Midpoint) Final Pose: X={hist_rk2[-1,0]:+.3f}m, Y={hist_rk2[-1,1]:+.3f}m, LCE = {lce_rk2*100:.1f} cm")
    print(f"  3. IMU-Fused Final Pose:      X={hist_fused[-1,0]:+.3f}m, Y={hist_fused[-1,1]:+.3f}m, LCE = {lce_fused*100:.1f} cm")
    print("=" * 60 + "\n")

    # Generate publication-quality comparison figure
    os.makedirs("docs/figures", exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # Subplot 1: 2D Spatial Trajectory Comparison
    # Ground truth reference square
    square_x = [0.0, side_length, side_length, 0.0, 0.0]
    square_y = [0.0, 0.0, side_length, side_length, 0.0]
    ax[0].plot(square_x, square_y, 'k--', linewidth=1.5, label='Nominal Reference Square')

    ax[0].plot(hist_euler[:, 0], hist_euler[:, 1], 'r-', linewidth=1.8, label=f'Forward Euler (LCE: {lce_euler*100:.1f} cm)')
    ax[0].plot(hist_rk2[:, 0], hist_rk2[:, 1], 'b-', linewidth=2.0, label=f'RK2 Midpoint (LCE: {lce_rk2*100:.1f} cm)')
    ax[0].plot(hist_fused[:, 0], hist_fused[:, 1], 'g-', linewidth=2.0, label=f'IMU-Fused (LCE: {lce_fused*100:.1f} cm)')

    ax[0].scatter([0], [0], color='green', s=80, zorder=5, label='Start Point (0,0)')
    ax[0].set_title("2D Trajectory: 1.0m x 1.0m Square Test", fontsize=12, fontweight='bold')
    ax[0].set_xlabel("X Position [meters]", fontsize=10)
    ax[0].set_ylabel("Y Position [meters]", fontsize=10)
    ax[0].legend(loc='best', fontsize=9)
    ax[0].grid(True, linestyle=':', alpha=0.6)
    ax[0].set_aspect('equal', adjustable='box')

    # Subplot 2: Heading (Theta) over time
    time_steps = np.arange(len(hist_rk2)) * 0.016
    ax[1].plot(time_steps, np.degrees(hist_euler[:, 2]), 'r-', label='Euler Heading')
    ax[1].plot(time_steps, np.degrees(hist_rk2[:, 2]), 'b-', label='RK2 Heading')
    ax[1].plot(time_steps, np.degrees(hist_fused[:, 2]), 'g-', label='Fused Gyro Heading')

    ax[1].set_title("Estimated Heading (Yaw) vs Time", fontsize=12, fontweight='bold')
    ax[1].set_xlabel("Time [seconds]", fontsize=10)
    ax[1].set_ylabel("Heading Angle [degrees]", fontsize=10)
    ax[1].legend(loc='best', fontsize=9)
    ax[1].grid(True, linestyle=':', alpha=0.6)

    fig.tight_layout()
    output_path = "docs/figures/odometry_benchmark.png"
    plt.savefig(output_path)
    print(f"[test_odometry] Benchmark plot saved to: {output_path}")
    plt.show()


if __name__ == '__main__':
    run_square_benchmark(side_length=1.0, speed_linear=0.2, speed_turn=0.5)
