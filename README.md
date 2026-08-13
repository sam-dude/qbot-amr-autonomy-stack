# 🤖 QBot Metric-Semantic AMR Autonomy Stack

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-Quanser%20QBot%203-orange.svg)](https://www.quanser.com)
[![Digital Twin](https://img.shields.io/badge/Sim-QLabs%20%7C%20Webots-purple.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An industry-grade **Autonomous Mobile Robot (AMR)** pipeline built for the **Quanser QBot Platform**, fully integrated with both the **QLabs Digital Twin** and **Physical Lab Hardware**.

---

## 🚀 Key Capabilities

1. **High-Precision Odometry & State Estimation**
   - 2nd-Order Runge-Kutta (RK2 midpoint) numerical integration for differential drive kinematics.
   - Complementary sensor fusion combining wheel encoders with 6-DOF IMU gyroscope yaw rate ($\omega_z$).
2. **2D Occupancy Grid Mapping (Lidar SLAM)**
   - Log-odds inverse sensor model mapping with fast Bresenham raycasting from 360° 1680-point 2D Lidar scans.
3. **RGB-D 3D Semantic Perception**
   - Real-time object detection (YOLOv8) back-projected using camera pinhole intrinsics into metric world coordinates $(X, Y, Z)$.
4. **Dynamic Obstacle Avoidance & Safety Supervisor**
   - Speed-dependent dynamic frontal safety cone with automatic stop/slowdown zones and visual LED state alerts.
5. **Real-Time OpenCV HUD**
   - Integrated telemetry dashboard rendering camera feeds with 3D bounding boxes, live SLAM occupancy grid, and state estimations.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Sensors["1. QBot Platform Sensors (Hardware & Simulation)"]
        ENC["Wheel Encoders (Left / Right)"]
        IMU["6-DOF IMU (Gyro yaw rate wz)"]
        LIDAR["2D Lidar (360°)"]
        RGBD["Intel RealSense (RGB + Depth)"]
    end

    subgraph State_Estimation["2. State Estimation & Odometry Engine"]
        EULER["Euler Odometry"]
        RK2["2nd-Order Runge-Kutta"]
        FUSE["IMU Complementary Fusion"]
    end

    subgraph Mapping_Perception["3. Mapping & Perception"]
        GRID["2D Occupancy Grid Mapping"]
        YOLO["3D Semantic Projection (YOLOv8)"]
    end

    subgraph Avoidance_Control["4. Navigation & Safety"]
        CONE["Dynamic Lidar Safety Cone"]
        REACT["Reactive Obstacle Avoidance"]
        SAFE["Driver & Watchdog Supervisor"]
    end

    subgraph Dashboard["5. Unified HUD"]
        HUD["Real-Time OpenCV Telemetry Dashboard"]
    end

    ENC --> EULER
    ENC --> RK2
    IMU --> FUSE
    RK2 --> FUSE

    FUSE --> GRID
    LIDAR --> GRID

    RGBD --> YOLO
    FUSE --> YOLO

    LIDAR --> CONE
    CONE --> REACT
    YOLO --> REACT
    REACT --> SAFE
    SAFE --> Sensors

    GRID --> HUD
    YOLO --> HUD
    FUSE --> HUD
```

---

## 📐 Mathematical Formulation

### Differential Drive Kinematics
Given wheel radius $r = 0.04445\text{ m}$ and track width (wheelbase) $L = 0.3928\text{ m}$:

$$\Delta s = \frac{\Delta s_R + \Delta s_L}{2}, \quad \Delta \theta_{\text{enc}} = \frac{\Delta s_R - \Delta s_L}{L}$$

### Runge-Kutta 2nd-Order (RK2 Midpoint Integration)
$$\theta_{\text{mid}} = \theta_k + \frac{\Delta \theta}{2}$$
$$x_{k+1} = x_k + \Delta s \cos(\theta_{\text{mid}}), \quad y_{k+1} = y_k + \Delta s \sin(\theta_{\text{mid}}), \quad \theta_{k+1} = \text{wrap}(\theta_k + \Delta \theta)$$

### Complementary Sensor Fusion
$$\Delta \theta_{\text{fused}} = \alpha \cdot \Delta \theta_{\text{enc}} + (1 - \alpha) \cdot (\omega_z \cdot \Delta t)$$

### 3D Pinhole Back-Projection
Given pixel $(u, v)$ and depth $Z = d(u, v)$ with camera intrinsics $(f_x, f_y, c_x, c_y)$:

$$X_c = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y_c = \frac{(v - c_y) \cdot Z}{f_y}, \quad Z_c = Z$$

---

## 📁 Repository Structure

```text
├── PROJECT_SPEC.md              # Project specification, checklist & technical formulas
├── README.md                    # Project documentation & overview
├── .gitignore                   # Ignored files & build artifacts
├── odometry_engine.py           # Core odometry & sensor fusion algorithms
├── qbot_helpers.py              # Hardware wrappers, sensor dataclass & OpenCV HUD
├── qlabs_setup.py               # QLabs digital twin environment setup & spawning
├── test_odometry.py             # 1.0m x 1.0m benchmark test harness & plot generator
├── webots_version_connection.md # Webots simulation connectivity documentation
└── docs/
    └── figures/                 # Benchmark charts, plots & telemetry figures
```

---

## ⚡ Quick Start

### 1. Prerequisites & Installation
Ensure Python 3.8+ is installed:
```bash
pip install numpy matplotlib opencv-python ultralytics
```
*(Install Quanser QLabs / HIL SDK if interfacing with physical hardware or QLabs simulator).*

### 2. Launch QLabs Simulation Environment
```bash
python qlabs_setup.py
```

### 3. Run Odometry Benchmark Test
```bash
python test_odometry.py
```

---

## 📌 Project Roadmap
- [x] **Phase 1**: Odometry Engine (Euler, RK2 & Complementary IMU fusion)
- [ ] **Phase 2**: 2D Occupancy Grid SLAM (Log-odds raycasting)
- [ ] **Phase 3**: RGB-D 3D Semantic Perception (YOLOv8 + metric back-projection)
- [ ] **Phase 4**: Reactive Obstacle Avoidance & Safety Dynamic Cone
- [ ] **Phase 5**: Full Integration & Physical Lab Deployment
