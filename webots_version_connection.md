Looking at your e-puck Webots code, you have already built several core concepts that map directly to what we are building for the QBot Platform.

Here is a side-by-side breakdown of the exact similar parts, showing how the logic you already know transfers directly to the QBot.

1. Differential Drive Kinematics & Arc Lengths
Your E-Puck Code:
python
delta_x = (WHEEL_RADIUS * phildot + WHEEL_RADIUS * phirdot) / 2 * DELTA_T
delta_omega_z = (WHEEL_RADIUS * phirdot - WHEEL_RADIUS * phildot) / WHEELBASE * DELTA_T
theta += delta_omega_z
How it maps to QBot (odometry_engine.py):
In QBot, the driver gives cumulative wheel angles $\theta_L, \theta_R$.
The incremental distance per wheel is: $$\Delta s_L = r \cdot \Delta \theta_L, \quad \Delta s_R = r \cdot \Delta \theta_R$$
The center displacement and heading change are identical: $$\Delta s = \frac{\Delta s_R + \Delta s_L}{2}, \quad \Delta \theta = \frac{\Delta s_R - \Delta s_L}{L}$$
The only difference: On e-puck you used Forward Euler (theta += delta_omega_z). In QBot update_pose_rk2(), you will use the midpoint angle ($\theta_{\text{mid}} = \theta + \Delta \theta / 2$) to eliminate curve drift.
2. Angle Wrapping & Normalization
Your E-Puck Code:
python
if (alpha > np.pi):
    alpha = alpha - 2 * np.pi
How it maps to QBot (wrap_to_pi):
This is the exact same concept as wrap_to_pi(angle) in Task 1.1 — keeping angles strictly inside $[-\pi, +\pi]$ so heading calculations never jump wildly when crossing $\pm 180^\circ$.
3. Coordinate Frame Transformation Matrix ($^wT_r$)
Your E-Puck Code:
python
w_T_r = np.array([[np.cos(theta), -np.sin(theta), xw],
                  [np.sin(theta),  np.cos(theta), yw],
                  [0,              0,             1]])
X_r = np.array([ranges * np.cos(angles), 
                ranges * np.sin(angles),
                np.ones(len(angles))])
D = w_T_r @ X_r
How it maps to QBot (Phase 2 Lidar Mapping & Phase 3 3D Vision):
This homogeneous transform ${}^wT_r \in SE(2)$ is 100% identical.
We will use this exact matrix math to transform the QBot's Leishen 360° Lidar points from robot frame to world frame, and to project 3D camera detections into world coordinates.
4. World-to-Map Discretization (world2map)
Your E-Puck Code:
python
def world2map(xw, yw):
    resolution = 0.003333
    origin_x = -0.195
    origin_y = -0.25
    px = (xw - origin_x) / resolution
    py = (yw - origin_y) / resolution
    px = int(np.clip(px, 0, 299))
    py = 299 - int(np.clip(py, 0, 299))  # Inverted Y for image coordinates
    return [px, py]
How it maps to QBot (Phase 2 Grid Mapping):
The exact same formula: $$\text{col} = \frac{x_w - x_{\text{origin}}}{\text{resolution}}, \quad \text{row} = \frac{y_w - y_{\text{origin}}}{\text{resolution}}$$
Converting continuous world meters into discrete grid/pixel cells.
5. Obstacle Inflation / Costmap
Your E-Puck Code:
python
cmap = signal.convolve2d(occupancy_map, kernel, mode='same')
cspace = cmap > 0.9
How it maps to QBot (Phase 4 Dynamic Costmaps):
You already used 2D convolution to inflate obstacles by a safety kernel radius! We will use this exact approach to inflate the Lidar/camera obstacles on the QBot.
6. Goal Tracking / Waypoint Steering
Your E-Puck Code:
python
rho = np.sqrt((xw - WP[index][0])**2 + (yw - WP[index][1])**2)
alpha = np.arctan2(WP[index][1] - yw, WP[index][0] - xw) - theta
leftSpeed = -alpha * p1 + rho * p2
rightSpeed =  alpha * p1 + rho * p2
How it maps to QBot:
Distance error $\rho$ and heading error $\alpha$ are the foundational control laws for differential drive waypoint navigation.
Summary of What is Already Familiar vs. What is New on QBot
Feature	Your E-Puck Project	QBot Platform
Geometry	$r = 0.0201\text{m}, L = 0.052\text{m}$	$r = 0.04445\text{m}, L = 0.3928\text{m}$
Integration	1st-Order Euler	Runge-Kutta 2nd-Order Midpoint (Higher accuracy)
Heading Source	Compass / GPS ground truth	IMU Gyroscope ($\omega_z$) Complementary Fusion
Lidar Transform	${}^wT_r @ X_r$ (Polar $\rightarrow$ Cartesian $\rightarrow$ World)	Same ${}^wT_r$ Matrix + 1680-point Leishen Lidar
Map Structure	2D array + world2map	Same Grid + Log-Odds Probability Update
Perception	2D Lidar only	2D Lidar + Intel RealSense RGB-D 3D Projection
You already understand the underlying geometry and transforms — now you are transferring it to real physical hardware, adding IMU sensor fusion and 3D vision!