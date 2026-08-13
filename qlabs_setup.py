"""
qlabs_setup.py — QLabs Digital Twin Environment Setup for AMR Project

Spawns:
  - QBot Platform at a configurable start position
  - A walled warehouse arena (~3m x 3m)
  - Obstacles (walls at angles) to make the environment interesting for mapping
  - Floor tiles for the downward camera

This file is ONLY used in simulation (QLabs Digital Twin).
On physical hardware, this file is skipped entirely.

Usage:
    from qlabs_setup import setup
    setup(locationQBotP=[0, 0, 0.05], rotationQBotP=[0, 0, 0], verbose=True)
"""

import sys
import time
import numpy as np
import os
import subprocess

from qvl.walls import QLabsWalls
from qvl.qlabs import QuanserInteractiveLabs
from qvl.free_camera import QLabsFreeCamera
from qvl.qbot_platform import QLabsQBotPlatform
from qvl.qbot_platform_flooring import QLabsQBotPlatformFlooring
from qvl.real_time import QLabsRealTime
import pal.resources.rtmodels as rtmodels


def setup(
    locationQBotP=[0, 0, 0.05],
    rotationQBotP=[0, 0, 0],
    verbose=True,
    rtModel_workspace=rtmodels.QBOT_PLATFORM,
    rtModel_driver=rtmodels.QBOT_PLATFORM_DRIVER
):
    """
    Sets up the QLabs simulation environment for the AMR project.

    Args:
        locationQBotP: [x, y, z] spawn position of the QBot in meters.
        rotationQBotP: [roll, pitch, yaw] spawn orientation in degrees.
        verbose: Print status messages during setup.
        rtModel_workspace: Real-time model for the QBot Platform workspace.
        rtModel_driver: Real-time model for the QBot Platform driver.

    Returns:
        hQBot: Handle to the spawned QBot Platform actor.
    """

    # Start the host peripheral client (handles keyboard, probes, etc.)
    subprocess.Popen(['quanser_host_peripheral_client.exe', '-q'])
    time.sleep(2.0)
    subprocess.Popen(
        ['quanser_host_peripheral_client.exe', '-uri',
         'tcpip://localhost:18444']
    )

    # Terminate any pre-existing RT models
    qrt = QLabsRealTime()
    if verbose:
        print("Stopping any pre-existing RT models...")
    qrt.terminate_real_time_model(rtModel_workspace)
    time.sleep(1.0)
    qrt.terminate_real_time_model(rtModel_driver)
    time.sleep(1.0)
    qrt.terminate_all_real_time_models()

    # Connect to QLabs
    qlabs = QuanserInteractiveLabs()
    if verbose:
        print("Connecting to QLabs...")
    if not qlabs.open("localhost"):
        print("Unable to connect to QLabs")
        sys.exit()
        return
    if verbose:
        print("Connected to QLabs!")

    # Clear any previously spawned actors
    qlabs.destroy_all_spawned_actors()

    # ======================== Spawn QBot Platform ========================
    if verbose:
        print("Spawning QBot Platform...")
    hQBot = QLabsQBotPlatform(qlabs)
    hQBot.spawn_id_degrees(
        actorNumber=0,
        location=locationQBotP,
        rotation=rotationQBotP,
        scale=[1, 1, 1],
        configuration=1,
        waitForConfirmation=False
    )
    hQBot.possess(hQBot.VIEWPOINT_TRAILING)

    # ======================== Spawn Arena Walls ========================
    # Creates a crisp, perfectly aligned 4.8m x 4.8m walled arena.
    # enDynamics=False keeps all walls rock-solid, perfectly upright and immovable.
    if verbose:
        print("Spawning arena walls...")

    hWall = QLabsWalls(qlabs)
    enDynamics = True  # Keep walls static and perfectly vertical (prevents tipping/buckling)

    # --- Outer boundary walls (4.8m x 4.8m perimeter, walls flush at z=0.0) ---
    wall_positions = [-2.0, -1.0, 0.0, 1.0, 2.0]

    # North wall (y = +2.4)
    for x in wall_positions:
        hWall.spawn_degrees(location=[x, 2.4, 0.0], rotation=[0, 0, 90])
        hWall.set_enable_dynamics(enDynamics)

    # South wall (y = -2.4)
    for x in wall_positions:
        hWall.spawn_degrees(location=[x, -2.4, 0.0], rotation=[0, 0, 90])
        hWall.set_enable_dynamics(enDynamics)

    # East wall (x = +2.4)
    for y in wall_positions:
        hWall.spawn_degrees(location=[2.4, y, 0.0], rotation=[0, 0, 0])
        hWall.set_enable_dynamics(enDynamics)

    # West wall (x = -2.4)
    for y in wall_positions:
        hWall.spawn_degrees(location=[-2.4, y, 0.0], rotation=[0, 0, 0])
        hWall.set_enable_dynamics(enDynamics)

    # --- Interior obstacle walls (clean upright obstacles outside the test quadrant) ---
    # NW obstacle (angled barrier)
    hWall.spawn_degrees(location=[-1.2, 1.0, 0.0], rotation=[0, 0, 45])
    hWall.set_enable_dynamics(enDynamics)

    # SW obstacle
    hWall.spawn_degrees(location=[-1.0, -1.2, 0.0], rotation=[0, 0, -45])
    hWall.set_enable_dynamics(enDynamics)

    # ======================== Spawn Floor Tiles ========================
    if verbose:
        print("Spawning flooring...")

    hFloor = QLabsQBotPlatformFlooring(qlabs)
    # Seamless 4x4 grid of 1.2m x 1.2m tiles covering 4.8m x 4.8m
    # Configured with a cohesive outer track loop and clean interior
    actor_id = 0
    tile_coords = [-1.8, -0.6, 0.6, 1.8]
    for x in tile_coords:
        for y in tile_coords:
            # Determine tile orientation & config for a neat track layout
            if x == -1.8 and y == 1.8:      # NW Corner
                rot, cfg = [0, 0, 0], 0
            elif x == 1.8 and y == 1.8:     # NE Corner
                rot, cfg = [0, 0, -np.pi/2], 0
            elif x == 1.8 and y == -1.8:    # SE Corner
                rot, cfg = [0, 0, np.pi], 0
            elif x == -1.8 and y == -1.8:   # SW Corner
                rot, cfg = [0, 0, np.pi/2], 0
            elif y in [1.8, -1.8]:          # North / South Straights
                rot, cfg = [0, 0, 0], 1
            elif x in [1.8, -1.8]:          # East / West Straights
                rot, cfg = [0, 0, np.pi/2], 1
            else:                           # Center 2x2 Arena
                rot, cfg = [0, 0, 0], 1

            hFloor.spawn_id(
                actorNumber=actor_id,
                location=[x, y, 0.0],
                rotation=rot,
                scale=[1, 1, 1],
                configuration=cfg,
                waitForConfirmation=False
            )
            actor_id += 1

    # ======================== Top-Down Camera ========================
    top_camera = QLabsFreeCamera(qlabs)
    top_camera.spawn_degrees(location=[0.0, 0.0, 5.5], rotation=[0, 90.0, 90.0])
    top_camera.possess()
    time.sleep(3)
    top_camera.set_camera_properties(90, False, 3.5, 12.0)

    # Switch back to trailing view on the QBot
    hQBot.possess(hQBot.VIEWPOINT_TRAILING)

    # ======================== Start RT Models ========================
    if verbose:
        print("Starting RT models...")
    time.sleep(2)
    qrt.start_real_time_model(rtModel_workspace, userArguments=False)
    time.sleep(1)
    qrt.start_real_time_model(
        rtModel_driver,
        userArguments=True,
        additionalArguments="-uri tcpip://localhost:17098"
    )
    if verbose:
        print("QLabs setup completed successfully!")

    return hQBot


if __name__ == '__main__':
    setup(locationQBotP=[0, 0, 0.05], rotationQBotP=[0, 0, 0], verbose=True)
