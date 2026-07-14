# Shared-Vision-for-Dual-Arm

**Dual-arm robotic collaboration using shared vision and intelligent arm selection for coordinated pick-and-place operations with two Kinova Gen3 Lite manipulators equipped with individual Eye-in-Hand (EIH) cameras.**

## Overview

This project enables two Kinova Gen3 Lite (6-DOF) robotic arms to work collaboratively in a shared workspace. The system uses ArUco marker detection from dual eye-in-hand cameras to localize targets and intelligently selects which arm should perform the pick-and-place operation based on workspace reachability and proximity.

### Key Features

- **Dual-Arm Collaboration**: Two Kinova Gen3 Lite arms working in a shared workspace
- **Shared Vision System**: Eye-in-hand cameras on each arm with synchronized detection
- **Intelligent Arm Selection**: Automatic selection of the optimal arm based on:
  - Workspace reachability (0.5m symmetric x/y box per arm)
  - Arm proximity to the target (closer arm is preferred in overlap regions)
- **ArUco Marker Detection**: Robust marker detection and localization using OpenCV
- **Resolved-Rate Control**: DLS (Damped Least Squares) Jacobian-based motion control
- **Safety-First Design**: Motion and gripper safeguards with dry-run modes

## Hardware Requirements

- **2x Kinova Gen3 Lite Manipulators** (6-DOF, 6.5kg payload each)
- **2x Eye-in-Hand Cameras** (individually calibrated)
- **Individual Electrical-Gripper Hardware** (EIH) per arm
- **Network Connection**: Right arm at `192.168.2.10`, Left arm at `192.168.3.11`
- **Printed ArUco Markers** (5x5_50 dictionary, 0.05m size, shared across workspace)

## File Descriptions

### `dual_arm_pick_place.py`
The main dual-arm pick-and-place controller with streaming-based marker detection.

**Workflow**:
1. Connect to both arms via TCP (192.168.2.10 for RIGHT, 192.168.3.11 for LEFT)
2. Stream both eye-in-hand cameras continuously with live OpenCV windows
3. Run ArUco detection on every frame
4. When marker is found in either camera:
   - Localize in both arms' base_link frames using the fixed rig transform
   - Check workspace reachability for each arm
   - Select the closest reachable arm (or the only reachable one)
5. Execute full pick-and-place sequence with the selected arm:
   - Open gripper → Approach → Descend → Grasp → Lift → Place → Release

**Key Configurations**:
- `ENABLE_MOTION`: Disable for dry-run (no actual motion)
- `ENABLE_GRIPPER`: Disable for dry-run (no gripper movement)
- `WORKSPACE_XY_LIMIT`: 0.5m (reachable workspace in each arm's x/y)
- `CAMERA_INDEX_LEFT`, `CAMERA_INDEX_RIGHT`: Video device indices (typically /dev/video0, /dev/video2)
- `MARKER_SIZE`: 0.05m (physical ArUco marker size)

### `dual_arm_search_pick_place.py`
An alternative controller that actively searches for the marker by sweeping each arm's joint_1.

**Workflow**:
1. Home both arms to predefined `JOINT_START_LEFT` / `JOINT_START_RIGHT` angles
2. Simultaneously sweep joint_1 on both arms (±120° at 8°/s by default)
3. On every sweep iteration, detect the marker in both cameras
4. Apply same reachability and proximity logic as `dual_arm_pick_place.py`
5. Execute pick-and-place with selected arm

**Use Case**: Better for finding markers in occluded or hard-to-reach areas by actively exploring the workspace.

## System Architecture

### Coordinate Frames

- **REFERENCE Frame**: LEFT arm's base_link (shared reference for both arms)
- **RIGHT arm's base_link**: Expressed relative to LEFT with fixed transform:
  ```
  T_REF_RIGHT = [[-1, 0, 0, 1.08],
                 [ 0,-1, 0, 0.00],
                 [ 0, 0, 1, 0.00],
                 [ 0, 0, 0, 1.00]]
  ```
  This represents a 1.08m separation along the X-axis with 180° yaw rotation.

### Kinematics

- **Forward Kinematics**: Implements Denavit-Hartenberg (DH) chain for 6-DOF arms
- **Jacobian**: Computed numerically (ε = 1e-6) for resolved-rate control
- **DLS Solver**: Damped Least Squares with adaptive damping (λ ≈ 0.12–0.45)
- **PD Control**: Position gain Kp=2.0, derivative gain Kd=0.05; Orientation Kp=0.9, Kd=0.04

### Camera Calibration

Each arm's camera is individually calibrated with intrinsic matrices and distortion coefficients:

- **LEFT Camera**: `CAMERA_MATRIX_LEFT`, `DIST_COEFFS_LEFT`
- **RIGHT Camera**: `CAMERA_MATRIX_RIGHT`, `DIST_COEFFS_RIGHT`
- **Fixed Tool Offset**: +130mm along Z from end-effector to camera
- **Tool Frame**: 90° YAW rotation + 0.130m Z offset from joint_6

## Usage

### Prerequisites

```bash
pip install numpy scipy opencv-python kortex-api
```

Ensure the Kinova SDK (`kortex_api`) is installed and the robot controller PC can reach both arm IPs.

### Basic Streaming Pick-and-Place

```bash
# Start with dry-run (no motion):
ENABLE_MOTION=False python3 dual_arm_pick_place.py

# Once detection/selection logic looks correct in logs:
ENABLE_MOTION=True ENABLE_GRIPPER=True python3 dual_arm_pick_place.py

# Press 'q' in the OpenCV window to abort the stream
```

### Active Search Pick-and-Place

```bash
# Tune JOINT_START_LEFT / JOINT_START_RIGHT to a pose with clear workspace visibility
ENABLE_MOTION=False python3 dual_arm_search_pick_place.py

# Enable actual motion once confident:
ENABLE_MOTION=True ENABLE_GRIPPER=True python3 dual_arm_search_pick_place.py
```

## Tuning Parameters

### Workspace & Safety

- `WORKSPACE_XY_LIMIT` (0.5m): Maximum reachable x/y distance from each arm's base_link
- `Z_FLOOR` (0.0m): Minimum tool-frame Z (prevents fingers hitting the table)
- `DESCEND_OFFSET` (0.08m): How far to descend before grasping
- `LIFT_HEIGHT` (0.10m): How high to lift after grasping
- `LIMIT_MARGIN` (0.08): Joint limit safety margin (degrees)

### Detection & Markers

- `MARKER_SIZE` (0.05m): Physical size of the printed ArUco marker
- `CAMERA_INDEX_LEFT`, `CAMERA_INDEX_RIGHT`: /dev/video indices (verify with `ls -la /dev/video*`)
- `CLOSE_VALUE` (0.6): Gripper closure fraction [0=open, 1=closed]
- `GRASP_WAIT_TIME` (2.0s): Wait after grasp/release for gripper settling

### Motion Control

- `KP_POS`, `KD_POS`: Position error gains (2.0, 0.05)
- `KP_ORI`, `KD_ORI`: Orientation error gains (0.9, 0.04)
- `VEL_MAX` (0.2): Maximum joint velocity (rad/s)
- `POS_TOL_DEFAULT`, `ORI_TOL_DEFAULT`: Standard convergence tolerances (8mm, 5°)
- `POS_TOL_GRASP`, `ORI_TOL_GRASP`: Tighter tolerances for grasping (5mm, 3°)
- `TIMEOUT` (60s): Max time to reach any waypoint

### Search-Specific (dual_arm_search_pick_place.py)

- `SEARCH_SPEED_DEG` (8.0°/s): Joint_1 sweep speed
- `SEARCH_LIMIT_DEG` (120.0°): Sweep range (±120° from start)
- `SEARCH_LOOP_HZ` (10Hz): Detection loop frequency
- `HOME_ACTION_TIMEOUT` (5s): Time to reach home pose before fallback
- `HOME_POS_TOL_DEG` (5.0°): Fallback tolerance for homing verification

## Example Workflow

1. **Setup**: Mount both arms on the table ~1.08m apart, attach cameras and grippers
2. **Calibration**: Calibrate each camera intrinsics; verify IPs are reachable
3. **Place Marker**: Put an ArUco marker in the shared workspace
4. **Dry-Run**: Run `dual_arm_pick_place.py` with `ENABLE_MOTION=False` to check:
   - Both cameras capture the workspace clearly
   - ArUco detection works in both views
   - Arm selection logic (reachability, proximity) is correct
   - No IK singularities in the approach
5. **Enable Motion**: Set `ENABLE_MOTION=True`, `ENABLE_GRIPPER=True` and run again
6. **Observe**: Watch the selected arm pick up and place the object
7. **Iterate**: Adjust `APPROACH_OFFSET_*` or `PLACE_POS` if grasps are imprecise

## Safety Notes

- **Always start with `ENABLE_MOTION=False`** to dry-run the logic
- **Check joint limits**: `JOINT_MIN/MAX` and `LIMIT_MARGIN` prevent hard stops
- **Verify Z_FLOOR**: Ensures the gripper doesn't hit the table
- **Emergency stops**: Use the Kinova teach pendant or power off the breaker
- **Workspace boundaries**: The 0.5m workspace box is enforced per arm; markers outside are not picked

## Coordinate Transform Math

When a marker is detected in LEFT camera:

```
1. Get T_cam_marker from ArUco pose estimation (in camera frame)
2. Get T0c from FK(q_LEFT) (base_link → camera, LEFT frame)
3. T_arm_marker = T0c @ T_cam_marker (base_link → marker, LEFT frame)

4. Convert to shared REFERENCE (LEFT is already the reference):
   pos_ref, quat_ref = pos_arm_marker, quat_arm_marker

5. Localize in RIGHT frame using T_REF_RIGHT (fixed rig transform):
   T_ref_marker = compose(pos_ref, quat_ref)
   T_right_marker = inv(T_REF_RIGHT) @ T_ref_marker
   pos_right, quat_right = decompose(T_right_marker)

6. Check which arm(s) can reach: in_workspace(pos_left) and/or in_workspace(pos_right)
```

## Contributing

- **Bug Reports**: Open an issue with logs from a dry-run
- **Tuning Tips**: Share calibration data or parameter updates for different setups
- **Extensions**: Pull requests welcome for additional features (e.g., force feedback, multi-object handling)

## License


## References

- Kinova Gen3 Lite: [https://www.kinovarobotics.com/product/gen3-lite](https://www.kinovarobotics.com/product/gen3-lite)
- OpenCV ArUco: [https://docs.opencv.org/4.5.2/d5/dae/tutorial_aruco_detection.html](https://docs.opencv.org/4.5.2/d5/dae/tutorial_aruco_detection.html)
- Damped Least Squares (DLS) Control: Wampler, C. W., & Leifer, L. J. (1985). "Application of damped least-squares method to resolved-rate teleoperated manipulator control." IEEE Transactions on Automatic Control, 30(3), 199-205.

## Support

For questions or issues, please refer to the docstrings in the source code or contact the project maintainers.
