import numpy as np

from odometry_engine import OdometryEngine
from phase_1_odometry.test_odometry_square import SquareMotionDriver


def test_turn_controller_tracks_heading_error():
    driver = SquareMotionDriver(robot=None, odom=OdometryEngine(init_pose=(0.0, 0.0, 0.0)))

    omega, done = driver._heading_turn_command(
        current_theta=0.0,
        target_theta=np.pi / 2.0,
        max_turn_rate=0.5,
        kp=1.2,
        tolerance=0.05,
    )

    assert not done
    assert omega > 0.0
    assert omega <= 0.5

    omega_small, done_small = driver._heading_turn_command(
        current_theta=np.pi / 2.0 - 0.02,
        target_theta=np.pi / 2.0,
        max_turn_rate=0.5,
        kp=1.2,
        tolerance=0.05,
    )

    assert done_small
    assert abs(omega_small) == 0.0
