import numpy as np

from qlabs_setup import build_arena_layout


def test_arena_layout_uses_consistent_grid():
    wall_positions, tile_positions, obstacles = build_arena_layout()

    expected_wall_positions = np.array([-2.4, -1.2, 0.0, 1.2, 2.4], dtype=float)
    expected_tile_positions = np.array([-1.8, -0.6, 0.6, 1.8], dtype=float)

    assert np.allclose(wall_positions, expected_wall_positions)
    assert np.allclose(tile_positions, expected_tile_positions)
    assert np.allclose(np.diff(wall_positions), np.full(4, 1.2))
    assert np.allclose(np.diff(tile_positions), np.full(3, 1.2))
    assert len(obstacles) == 4
    assert all(len(obs) == 3 for obs in obstacles)


def test_odometry_freezes_when_robot_is_stalled():
    from odometry_engine import OdometryEngine

    odom = OdometryEngine(init_pose=(0.0, 0.0, 0.0))
    odom.step(np.array([0.0, 0.0]), gyro_yaw_rate=0.0, dt=0.016)
    pose_before = np.array([odom.x_rk2, odom.y_rk2, odom.theta_rk2])

    odom.step(np.array([1e-4, 1e-4]), gyro_yaw_rate=0.0, dt=0.016)
    pose_after = np.array([odom.x_rk2, odom.y_rk2, odom.theta_rk2])

    assert np.allclose(pose_after, pose_before)
