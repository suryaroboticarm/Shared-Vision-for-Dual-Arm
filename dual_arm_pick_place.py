#!/usr/bin/env python3
"""
dual_arm_pick_place.py -- Two Kinova Gen3 Lite (6-DOF) arms, LEFT + RIGHT,
detect -> pick -> place, whichever arm can actually reach the marker.
================================================================================
Builds on dual_arm_live_localize.py's dual-camera streaming + cross-frame
localization, and aruco_pick_left.py / aruco_pick _right.py's DLS resolved-
rate pick-and-place sequence (open gripper -> detect -> approach -> descend
-> grasp -> lift -> place -> release).
 
Flow
----
1. Connect RIGHT (192.168.2.10) + LEFT (192.168.3.11) via utilities_dual.py.
2. Stream BOTH eye-in-hand cameras continuously (RIGHT=camera_index 0,
   LEFT=camera_index 2). On every frame, run ArUco detection.
3. Whenever the marker is found in EITHER camera, localize it in BOTH arms'
   own base_link frames (via the fixed LEFT<->RIGHT rig transform) and check
   each arm's own WORKSPACE_XY_LIMIT (0.5m box in that arm's own x/y).
     - Neither arm can reach it  -> keep streaming, try the next frame.
     - Exactly one arm can reach it -> that arm is selected.
     - BOTH arms can reach it (overlap region) -> whichever arm's own
       base_link is physically CLOSER to the marker is selected.
4. Stop streaming, run the full pick-and-place sequence (DLS/PD resolved-
   rate control, gripper 180 deg yaw-symmetry disambiguation using the
   SELECTED arm's own current tool orientation) with the selected arm.
 
Safety: ENABLE_MOTION=False, ENABLE_GRIPPER=False (dry-run) -- flip both to
True only once the detection/selection/target logic looks right in the logs.
"""
 
import os, sys, time, math
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
 
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
 
# ── SAFETY ────────────────────────────────────────────────────────────────
ENABLE_MOTION  = True  # TUNE: False = dry-run, no arm motion
ENABLE_GRIPPER = True  # TUNE: False = dry-run, no gripper motion
 
# ── RIG GEOMETRY: RIGHT arm's base_link, expressed in LEFT arm's base_link
# frame (confirmed). LEFT's own base_link IS the shared reference frame. ───
T_REF_LEFT  = np.eye(4)
T_REF_RIGHT = np.array([[-1,  0, 0, 1.08],
                         [ 0, -1, 0, 0.00],
                         [ 0,  0, 1, 0.00],
                         [ 0,  0, 0, 1.00]])
 
# ── PER-ARM REACHABLE WORKSPACE (confirmed): symmetric box in that arm's
# OWN base_link x/y -- |x| <= LIMIT and |y| <= LIMIT. ──────────────────────
WORKSPACE_XY_LIMIT = 0.5  # metres
 
 
# ── REFERENCE <-> ARM-LOCAL POSE CONVERSION HELPERS ────────────────────────
 
def _pose_to_T(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    T[:3, 3]  = pos
    return T
 
 
def _T_to_pose(T: np.ndarray):
    pos  = T[:3, 3]
    quat = R.from_matrix(T[:3, :3]).as_quat()  # xyzw
    return pos, quat
 
 
def arm_to_ref(pos_arm: np.ndarray, quat_arm_xyzw: np.ndarray, T_ref_arm: np.ndarray):
    """Convert a pose measured in an arm's own base_link frame into the shared REFERENCE (LEFT) frame."""
    T_arm = _pose_to_T(pos_arm, quat_arm_xyzw)
    T_ref = T_ref_arm @ T_arm
    return _T_to_pose(T_ref)
 
 
def ref_to_arm(pos_ref: np.ndarray, quat_ref_xyzw: np.ndarray, T_ref_arm: np.ndarray):
    """Convert a pose expressed in the shared REFERENCE (LEFT) frame into an arm's own base_link frame."""
    T_ref = _pose_to_T(pos_ref, quat_ref_xyzw)
    T_arm = np.linalg.inv(T_ref_arm) @ T_ref
    return _T_to_pose(T_arm)
 
 
def in_workspace(pos_local: np.ndarray) -> bool:
    """True if pos_local (this arm's own base_link frame) is within that arm's 0.5m x/y box."""
    return abs(pos_local[0]) <= WORKSPACE_XY_LIMIT and abs(pos_local[1]) <= WORKSPACE_XY_LIMIT
 
 
# ── SHARED DH-CHAIN KINEMATICS (identical in aruco_pick_left.py / aruco_pick
# _right.py -- both arms are the same Gen3 Lite model) ─────────────────────
 
def _dh_transforms(q: np.ndarray):
    q1, q2, q3, q4, q5, q6 = q
 
    T01 = np.array([[math.cos(q1), -math.sin(q1), 0, 0],
                     [math.sin(q1),  math.cos(q1), 0, 0],
                     [0,             0,             1, 0.1283],
                     [0,             0,             0, 1]])
 
    T12 = np.array([[math.cos(q2), -math.sin(q2), 0,  0],
                     [0,             0,           -1, -0.03],
                     [math.sin(q2),  math.cos(q2), 0,  0.1150],
                     [0,             0,             0, 1]])
 
    T23 = np.array([[math.cos(q3), -math.sin(q3),  0, 0],
                     [-math.sin(q3), -math.cos(q3), 0, 0.280],
                     [0,              0,           -1, 0],
                     [0,              0,             0, 1]])
 
    T34 = np.array([[math.cos(q4), -math.sin(q4), 0,  0],
                     [0,             0,           -1, -0.140],
                     [math.sin(q4),  math.cos(q4), 0,  0.020],
                     [0,             0,             0, 1]])
 
    T45 = np.array([[0,             0,            1,  0.0285],
                     [math.sin(q5),  math.cos(q5), 0,  0],
                     [-math.cos(q5), math.sin(q5), 0,  0.105],
                     [0,             0,             0, 1]])
 
    T56 = np.array([[0,             0,            -1, -0.105],
                     [math.sin(q6),  math.cos(q6), 0,   0],
                     [math.cos(q6), -math.sin(q6), 0,   0.0285],
                     [0,             0,             0,  1]])
 
    # Fixed tool_frame offset after joint_6 (Rot_z(+90 deg), +0.130 m along z)
    T6e = np.array([[0, -1, 0, 0],
                     [1,  0, 0, 0],
                     [0,  0, 1, 0.130],
                     [0,  0, 0, 1]])
 
    # Fixed offset from tool_frame to the eye-in-hand camera frame (same
    # physical mount on both arms, per aruco_pick_left.py/aruco_pick _right.py)
    Tec = np.array([
        [1, 0, 0,  0.010],
        [0, 1, 0, -0.045],
        [0, 0, 1, -0.060],
        [0, 0, 0,  1.000]
    ])
 
    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T34
    T05 = T04 @ T45
    T06 = T05 @ T56
    T0e = T06 @ T6e     # base_link -> tool_frame
    T0c = T0e @ Tec     # base_link -> camera
    return T0e, T0c
 
 
def fk(q: np.ndarray) -> np.ndarray:
    """Forward kinematics to the TOOL FRAME, in THIS ARM's OWN base_link frame."""
    T0e, _ = _dh_transforms(q)
    return T0e
 
 
def fk_camera(q: np.ndarray) -> np.ndarray:
    """Forward kinematics to the eye-in-hand CAMERA FRAME, in THIS ARM's OWN base_link frame."""
    _, T0c = _dh_transforms(q)
    return T0c
 
 
def jacobian(q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    J = np.zeros((6, 6))
    for i in range(6):
        dq = np.zeros(6); dq[i] = eps
        Tp, Tm = fk(q + dq), fk(q - dq)
        J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * eps)
        dR = Tp[:3, :3] @ Tm[:3, :3].T
        J[3:, i] = R.from_matrix(dR).as_rotvec() / (2 * eps)
    return J
 
 
def dls(J: np.ndarray, v: np.ndarray, lam: float) -> np.ndarray:
    return J.T @ np.linalg.inv(J @ J.T + lam ** 2 * np.eye(6)) @ v
 
 
def quat_error(q_curr: np.ndarray, q_tgt: np.ndarray):
    q_curr = q_curr / np.linalg.norm(q_curr)
    q_tgt  = q_tgt  / np.linalg.norm(q_tgt)
    q_e = (R.from_quat(q_tgt) * R.from_quat(q_curr).inv()).as_quat()
    if q_e[3] < 0.0:
        q_e = -q_e
    eo    = 2.0 * q_e[:3]
    theta = 2.0 * math.atan2(np.linalg.norm(q_e[:3]), q_e[3])
    return eo, theta
 
 
# ── ARUCO / CAMERA SETTINGS -- SEPARATE PER ARM (own intrinsic calibration,
# copied from aruco_pick_left.py / aruco_pick _right.py) ───────────────────
 
CAMERA_MATRIX_LEFT = np.array(
    [[976.1292728, 0.0, 335.86371525],
     [0.0, 967.34540179, 224.60492455],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
DIST_COEFFS_LEFT = np.array(
    [[2.40737359e-01, -2.34812580e+00, -9.14598030e-03, 1.09061781e-02, 1.00142578e+01]],
    dtype=np.float64,
)
 
CAMERA_MATRIX_RIGHT = np.array(
    [[959.93906701, 0.0, 286.77927284],
     [0.0, 954.28325113, 251.94571559],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
DIST_COEFFS_RIGHT = np.array(
    [[8.1540904e-02, 4.94546396e-01, 8.33127947e-04, -1.23350346e-02, -6.32870701e+00]],
    dtype=np.float64,
)
 
CAMERA_INDEX_LEFT  = 2  # /dev/video index, TUNE per host
CAMERA_INDEX_RIGHT = 0  # /dev/video index, TUNE per host
 
MARKER_SIZE = 0.05  # metres, TUNE to your printed marker size (shared, same physical markers)
 
_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
_DETECTOR_PARAMS = cv2.aruco.DetectorParameters()
_DETECTOR = cv2.aruco.ArucoDetector(_ARUCO_DICT, _DETECTOR_PARAMS)
 
 
# ── PICK-AND-PLACE WAYPOINTS / GRASP SETTINGS (from aruco_pick_left.py /
# aruco_pick _right.py -- APPROACH_OFFSET differs per arm, rest is shared) ─
DESCEND_OFFSET = 0.08  # metres, descend for grasp
LIFT_HEIGHT    = 0.10   # metres, lift after grasp
 
Z_FLOOR = 0.0  # metres, TUNE: minimum allowed tool-frame Z in the arm's OWN base_link
                # frame -- both arms are mounted 5cm above the table, so the tool must
                # never be commanded below this or the fingers hit the table/ground.
 
TARGET_RPY_DEG = [0.0, -180.0, 0.0]  # fixed roll/pitch for the pick orientation; yaw comes from the marker
 
PLACE_POS  = np.array([0.439, 0.193, 0.448])   # in the picking arm's OWN base_link frame
PLACE_QUAT = R.from_euler('xyz', [90.65, -0.98, 150.028], degrees=True).as_quat()
 
# Per-arm hover-above-marker offset (own base_link frame) -- kept separate
# per arm in case each arm's own tool/camera calibration needs its own value.
APPROACH_OFFSET_LEFT  = np.array([0.05, 0.02, 0.07])
APPROACH_OFFSET_RIGHT = np.array([0.03, 0.07, 0.07])
 
CLOSE_VALUE     = 0.6  # gripper close fraction [0=open .. 1=closed]
OPEN_VALUE      = 0.0
GRASP_WAIT_TIME = 2.0   # seconds to wait after grasp/release
 
# ── TOLERANCES / CONTROLLER GAINS (identical to aruco_pick_left.py/right.py) ─
POS_TOL_DEFAULT = 0.008
ORI_TOL_DEFAULT = math.radians(5.0)
POS_TOL_GRASP   = 0.005
ORI_TOL_GRASP   = math.radians(3.0)
TIMEOUT         = 60.0
KP_POS, KD_POS  = 2.0, 0.05
KP_ORI, KD_ORI  = 0.9, 0.04
DERIV_TAU       = 0.06
VEL_MAX         = 0.2
JOINT_MIN = np.array([-2.69, -2.69, -2.69, -2.59, -2.57, -2.59])
JOINT_MAX = np.array([ 2.69,  2.36,  2.69,  2.59,  2.57,  2.59])
LIMIT_MARGIN = 0.08
 
 
def brake(q: np.ndarray, dq: np.ndarray) -> np.ndarray:
    out = dq.copy()
    for i in range(6):
        if JOINT_MAX[i] - q[i] < LIMIT_MARGIN and dq[i] > 0:
            out[i] *= max(0.0, (JOINT_MAX[i] - q[i]) / LIMIT_MARGIN)
        if q[i] - JOINT_MIN[i] < LIMIT_MARGIN and dq[i] < 0:
            out[i] *= max(0.0, (q[i] - JOINT_MIN[i]) / LIMIT_MARGIN)
    return out
 
 
# ── PER-ARM HARDWARE HANDLE ─────────────────────────────────────────────────
 
class ArmHandle:
    def __init__(self, base: BaseClient, base_cyclic: BaseCyclicClient,
                 T_ref_arm: np.ndarray, name: str, camera_index: int,
                 camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                 approach_offset: np.ndarray, place_pos: np.ndarray, place_quat: np.ndarray):
        self.base            = base
        self.base_cyclic     = base_cyclic
        self.T_ref_arm       = T_ref_arm     # this arm's base_link, expressed in the shared LEFT reference frame
        self.name            = name
        self.camera_index    = camera_index
        self.camera_matrix   = camera_matrix
        self.dist_coeffs     = dist_coeffs
        self.approach_offset = approach_offset  # this arm's own hover-above-marker offset
        self.place_pos       = place_pos        # this arm's own drop-off position (own base_link frame)
        self.place_quat      = place_quat        # this arm's own drop-off orientation (own base_link frame)
 
    def get_q(self) -> np.ndarray:
        fb    = self.base_cyclic.RefreshFeedback()
        q_deg = np.array([fb.actuators[i].position for i in range(6)], dtype=float)
        return np.mod(np.radians(q_deg) + np.pi, 2 * np.pi) - np.pi
 
    def send(self, msg, dq_deg: np.ndarray) -> None:
        del msg.joint_speeds[:]
        for j in range(6):
            js = msg.joint_speeds.add()
            js.joint_identifier = j
            js.value    = float(dq_deg[j])
            js.duration = 0
        if ENABLE_MOTION:
            self.base.SendJointSpeedsCommand(msg)
        else:
            print(f"  [{self.name} DRY-RUN] ENABLE_MOTION=False -- would send joint speeds "
                  f"(deg/s): {np.round(dq_deg, 3)}")
 
    def send_gripper_position(self, value: float) -> None:
        if not ENABLE_GRIPPER:
            print(f"[{self.name} GRIPPER DRY-RUN] ENABLE_GRIPPER=False -- would set position={value:.2f}")
            return
        cmd = Base_pb2.GripperCommand()
        cmd.mode = Base_pb2.GRIPPER_POSITION
        finger = cmd.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = float(value)
        self.base.SendGripperCommand(cmd)
 
    def reach_ref(self, target_pos_ref: np.ndarray, target_quat_ref_xyzw: np.ndarray,
                  pos_tol: float = POS_TOL_DEFAULT, ori_tol: float = ORI_TOL_DEFAULT,
                  label: str = "") -> bool:
        """
        Drive THIS arm's tool_frame to (target_pos_ref, target_quat_ref) given
        in the shared REFERENCE (LEFT) frame. Converts once to this arm's own
        base_link frame (ref_to_arm) then runs the same DLS/PD resolved-rate
        loop as aruco_pick_left.py/aruco_pick _right.py's reach().
        """
        target_pos, q_tgt = ref_to_arm(target_pos_ref, target_quat_ref_xyzw, self.T_ref_arm)
        q_tgt = q_tgt / np.linalg.norm(q_tgt)
 
        msg = Base_pb2.JointSpeeds()
        dt_nominal = 1.0 / 10.0
        t0 = time.time()
        ok_counter = 0
        _first_iter = True
        prev_ep = np.zeros(3); prev_eo = np.zeros(3)
        dfp = np.zeros(3); dfo = np.zeros(3)
 
        print(f"\n[{self.name}/{label}] Reach target -- REFERENCE(LEFT) frame: "
              f"pos={np.round(target_pos_ref, 4)} m")
        print(f"[{self.name}/{label}] Reach target -- converted to {self.name}'s own "
              f"base_link frame: pos={np.round(target_pos, 4)} m")
 
        t_prev = time.time()
        while True:
            t_loop = time.time()
            dt = max(t_loop - t_prev, 1e-3)
            t_prev = t_loop
 
            q = self.get_q()
            T_tool = fk(q)
            pos_curr = T_tool[:3, 3]
            q_curr = R.from_matrix(T_tool[:3, :3]).as_quat()
 
            if _first_iter:
                if np.dot(q_tgt, q_curr) < 0:
                    q_tgt = -q_tgt
                _first_iter = False
 
            ep = target_pos - pos_curr
            eo, eo_theta = quat_error(q_curr, q_tgt)
            ep_n = np.linalg.norm(ep)
 
            print(f"  [{self.name}/{label}] closing in on target: position error={ep_n*1000:6.1f} mm, "
                  f"orientation error={math.degrees(eo_theta):5.2f} deg", end='\r')
 
            if ep_n < pos_tol and eo_theta < ori_tol:
                ok_counter += 1
                if ok_counter >= 3:
                    self.send(msg, np.zeros(6))
                    print(f"\n[{self.name}/{label}] Converged -- within tolerance "
                          f"(pos<{pos_tol*1000:.1f}mm, ori<{math.degrees(ori_tol):.1f}deg) "
                          f"for 3 consecutive samples. Stopping.")
                    return True
            else:
                ok_counter = 0
 
            if time.time() - t0 > TIMEOUT:
                print(f"\n[{self.name}/{label}] FAILED -- {TIMEOUT:.0f}s timeout exceeded before "
                      f"converging (last error: pos={ep_n*1000:.1f}mm, ori={math.degrees(eo_theta):.2f}deg).")
                self.send(msg, np.zeros(6))
                return False
 
            alpha = dt / (DERIV_TAU + dt)
            dfp = alpha * (ep - prev_ep) / dt + (1 - alpha) * dfp
            dfo = alpha * (eo - prev_eo) / dt + (1 - alpha) * dfo
            prev_ep, prev_eo = ep.copy(), eo.copy()
 
            vp = KP_POS * ep + KD_POS * dfp
            vo = KP_ORI * eo + KD_ORI * dfo
            v = np.concatenate([vp, vo])
 
            J = jacobian(q)
            lam = 0.45 if math.sqrt(max(0.0, np.linalg.det(J @ J.T))) < 0.01 else 0.12
            dq = dls(J, v, lam)
            dq = np.clip(brake(q, dq), -VEL_MAX, VEL_MAX)
 
            self.send(msg, np.degrees(dq))
 
            sleep = dt_nominal - (time.time() - t_loop)
            if sleep > 0.0:
                time.sleep(sleep)
 
 
# ── CONTINUOUS DUAL-CAMERA STREAM -> FIRST REACHABLE DETECTION ─────────────
 
def _detect_marker(arm: ArmHandle, frame: np.ndarray):
    """
    Run ArUco detection on a single frame from `arm`'s camera. Draws the
    detection overlay on `frame` in-place. Returns (found, pos_local,
    rot_local) -- pos/rot in `arm`'s OWN base_link frame -- or
    (False, None, None) if no marker was found in this frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _DETECTOR.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return False, None, None
 
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, MARKER_SIZE, arm.camera_matrix, arm.dist_coeffs
    )
    dists = [float(np.linalg.norm(tvecs[i])) for i in range(len(ids))]
    best  = int(np.argmin(dists))
    tvec, rvec = tvecs[best][0], rvecs[best][0]
    print(f"Z is:::::::::::: {tvec[2]}")
 
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.drawFrameAxes(frame, arm.camera_matrix, arm.dist_coeffs, rvec, tvec, MARKER_SIZE * 0.5)
 
    q   = arm.get_q()          # live joint feedback, THIS arm only, at detection time
    T0c = fk_camera(q)         # base_link -> camera, THIS arm's own frame
 
    R_cam_marker, _ = cv2.Rodrigues(rvec)
    T_cam_marker = np.eye(4)
    T_cam_marker[:3, :3] = R_cam_marker
    T_cam_marker[:3, 3]  = tvec
 
    T_arm_marker = T0c @ T_cam_marker   # base_link -> marker, THIS arm's own frame
    pos_local = T_arm_marker[:3, 3]
    rot_local = T_arm_marker[:3, :3]
    return True, pos_local, rot_local
 
 
def stream_until_reachable(arm_left: ArmHandle, arm_right: ArmHandle):
    """
    Streams both arms' cameras continuously (live cv2.imshow windows).
    Whenever the marker is detected in either camera, localizes it in BOTH
    arms' own base_link frames and checks each against WORKSPACE_XY_LIMIT.
 
      - Neither arm can reach it -> keep streaming.
      - One arm can reach it     -> that arm is selected.
      - Both can reach it        -> whichever arm's own base_link is
                                     physically closer to the marker wins.
 
    Returns (chosen_arm, marker_pos_local, marker_rot_local) -- pos/rot
    already in chosen_arm's OWN base_link frame, ready for the pick
    sequence -- or None if 'q' was pressed before a reachable detection.
    """
    caps = {}
    for arm in (arm_left, arm_right):
        cap = cv2.VideoCapture(arm.camera_index)
        if not cap.isOpened():
            print(f"[{arm.name}] ERROR: could not open camera (/dev/video{arm.camera_index}) "
                  f"-- this arm's stream will be skipped.")
        caps[arm.name] = cap
 
    if not any(c.isOpened() for c in caps.values()):
        print("No cameras available on either arm -- aborting.")
        return None
 
    print(f"Streaming both arms' cameras, searching for a marker within either arm's "
          f"{WORKSPACE_XY_LIMIT}m x/y workspace. Press 'q' to abort.")
 
    try:
        while True:
            for arm, other in ((arm_left, arm_right), (arm_right, arm_left)):
                cap = caps[arm.name]
                if not cap.isOpened():
                    continue
                ret, frame = cap.read()
                if not ret:
                    continue
 
                found, pos_own, rot_own = _detect_marker(arm, frame)
                cv2.imshow(arm.name, frame)
 
                if found:
                    quat_own = R.from_matrix(rot_own).as_quat()
                    pos_ref, quat_ref = arm_to_ref(pos_own, quat_own, arm.T_ref_arm)
                    pos_other, quat_other = ref_to_arm(pos_ref, quat_ref, other.T_ref_arm)
                    rot_other = R.from_quat(quat_other).as_matrix()
 
                    local_poses = {arm.name: (pos_own, rot_own), other.name: (pos_other, rot_other)}
                    reachable = {n: in_workspace(p) for n, (p, _) in local_poses.items()}
 
                    print(f"[{arm.name}] Marker detected -- "
                          f"{arm.name} local pos={np.round(pos_own, 4)} m (reachable={reachable[arm.name]}), "
                          f"{other.name} local pos={np.round(pos_other, 4)} m (reachable={reachable[other.name]})")
 
                    candidates = [n for n, ok in reachable.items() if ok]
                    if not candidates:
                        print(f"  -> outside BOTH arms' {WORKSPACE_XY_LIMIT}m workspace -- continuing to search.")
                    else:
                        if len(candidates) == 2:
                            chosen_name = min(candidates, key=lambda n: np.linalg.norm(local_poses[n][0]))
                            print(f"  -> reachable by BOTH arms; {chosen_name} is closer -- selected.")
                        else:
                            chosen_name = candidates[0]
                            print(f"  -> reachable only by {chosen_name} -- selected.")
 
                        chosen_arm = arm if chosen_name == arm.name else other
                        chosen_pos, chosen_rot = local_poses[chosen_name]
                        return chosen_arm, chosen_pos, chosen_rot
 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n'q' pressed -- aborting before any reachable detection.")
                return None
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()
 
 
# ── PICK-AND-PLACE SEQUENCE (adapted from aruco_pick_left.py / aruco_pick
# _right.py's main(), generalized to whichever arm was selected) ──────────
 
def run_pick_sequence(arm: ArmHandle, marker_pos_local: np.ndarray, marker_rot_local: np.ndarray) -> bool:
    print(f"\n=== [{arm.name}] selected for pick-and-place "
          f"(marker at {np.round(marker_pos_local, 4)} m in its own base_link frame) ===")
 
    print(f"[{arm.name}] Opening gripper...")
    arm.send_gripper_position(OPEN_VALUE)
    time.sleep(GRASP_WAIT_TIME)
 
    _, _, y_marker = R.from_matrix(marker_rot_local).as_euler('xyz', degrees=True)
    r_target_deg, p_target_deg, _ = TARGET_RPY_DEG
 
    # Gripper has 180-deg yaw symmetry: y_marker and y_marker+180 produce
    # identical grasps -- pick whichever is closer to the SELECTED arm's
    # CURRENT tool orientation to avoid an unnecessary 180-deg wrist detour.
    # (See aruco_pick _right.py's identical disambiguation; quaternion dot
    # product is used instead of Euler comparison since as_euler('xyz') is
    # not well-behaved for pitch=-180 targets.)
    q_curr_now = R.from_matrix(fk(arm.get_q())[:3, :3]).as_quat()
 
    y_cand1 = y_marker
    y_cand2 = (y_marker + 180.0 + 180.0) % 360.0 - 180.0
    q_cand1 = R.from_euler('xyz', [r_target_deg, p_target_deg, y_cand1], degrees=True).as_quat()
    q_cand2 = R.from_euler('xyz', [r_target_deg, p_target_deg, y_cand2], degrees=True).as_quat()
    dot1 = abs(np.dot(q_cand1, q_curr_now))
    dot2 = abs(np.dot(q_cand2, q_curr_now))
 
    if dot2 > dot1:
        pick_quat = q_cand2
        print(f"[{arm.name}] Gripper yaw symmetry: chose +180 candidate "
              f"yaw={y_cand2:.1f} deg (dot={dot2:.4f} vs {dot1:.4f})")
    else:
        pick_quat = q_cand1
        print(f"[{arm.name}] Gripper yaw symmetry: kept original candidate "
              f"yaw={y_cand1:.1f} deg (dot={dot1:.4f} vs {dot2:.4f})")
 
    approach_pos = marker_pos_local + arm.approach_offset
    descend_pos  = approach_pos.copy(); descend_pos[2] -= DESCEND_OFFSET
    lift_pos     = descend_pos.copy();  lift_pos[2]    += LIFT_HEIGHT
 
    # Hard Z-floor clamp -- both arms are mounted 5cm above the table, so no
    # tool-frame target may go below Z_FLOOR or the fingers hit the table.
    for name, pos in (("APPROACH", approach_pos), ("DESCEND", descend_pos), ("LIFT", lift_pos)):
        if pos[2] < Z_FLOOR:
            print(f"[{arm.name}] Z-floor clamp: {name} target z={pos[2]:.4f}m < "
                  f"Z_FLOOR={Z_FLOOR:.4f}m -- clamping to {Z_FLOOR:.4f}m.")
            pos[2] = Z_FLOOR
 
    def to_ref(pos, quat):
        return arm_to_ref(pos, quat, arm.T_ref_arm)
 
    pos_ref, quat_ref = to_ref(approach_pos, pick_quat)
    if not arm.reach_ref(pos_ref, quat_ref, POS_TOL_DEFAULT, ORI_TOL_DEFAULT, "APPROACH"):
        return False
 
    pos_ref, quat_ref = to_ref(descend_pos, pick_quat)
    if not arm.reach_ref(pos_ref, quat_ref, POS_TOL_GRASP, ORI_TOL_GRASP, "DESCEND"):
        return False
 
    print(f"[{arm.name}] Closing gripper...")
    arm.send_gripper_position(CLOSE_VALUE)
    time.sleep(GRASP_WAIT_TIME)
 
    pos_ref, quat_ref = to_ref(lift_pos, pick_quat)
    if not arm.reach_ref(pos_ref, quat_ref, POS_TOL_DEFAULT, ORI_TOL_DEFAULT, "LIFT"):
        return False
 
    pos_ref, quat_ref = to_ref(arm.place_pos, arm.place_quat)
    if not arm.reach_ref(pos_ref, quat_ref, POS_TOL_DEFAULT, ORI_TOL_DEFAULT, "PLACE"):
        return False
 
    print(f"[{arm.name}] Opening gripper to release...")
    arm.send_gripper_position(OPEN_VALUE)
    time.sleep(GRASP_WAIT_TIME)
 
    print(f"[{arm.name}] Pick-and-place sequence complete.")
    return True
 
 
# ── MAIN ────────────────────────────────────────────────────────────────
 
def main() -> int:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities_dual
 
    # RIGHT = 192.168.2.10, LEFT = 192.168.3.11 (confirmed).
    args_right = utilities_dual.parseConnectionArguments()
    args_left  = utilities_dual.parseConnectionArguments2()
 
    with utilities_dual.DeviceConnection.createTcpConnection(args_right) as router_right, \
         utilities_dual.DeviceConnection.createTcpConnection(args_left) as router_left:
 
        base_right = BaseClient(router_right); base_cyclic_right = BaseCyclicClient(router_right)
        base_left  = BaseClient(router_left);  base_cyclic_left  = BaseCyclicClient(router_left)
 
        for b in (base_right, base_left):
            mode = Base_pb2.ServoingModeInformation()
            mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
            b.SetServoingMode(mode)
 
        arm_left = ArmHandle(base_left, base_cyclic_left, T_REF_LEFT, "LEFT",
                              camera_index=CAMERA_INDEX_LEFT, camera_matrix=CAMERA_MATRIX_LEFT,
                              dist_coeffs=DIST_COEFFS_LEFT,
                              approach_offset=APPROACH_OFFSET_LEFT,
                              place_pos=PLACE_POS, place_quat=PLACE_QUAT)
        arm_right = ArmHandle(base_right, base_cyclic_right, T_REF_RIGHT, "RIGHT",
                               camera_index=CAMERA_INDEX_RIGHT, camera_matrix=CAMERA_MATRIX_RIGHT,
                               dist_coeffs=DIST_COEFFS_RIGHT,
                               approach_offset=APPROACH_OFFSET_RIGHT,
                               place_pos=PLACE_POS, place_quat=PLACE_QUAT)
 
        # Sanity check: each arm's own base_link origin, in the shared
        # REFERENCE(LEFT) frame. LEFT should read (0,0,0); RIGHT should read
        # (1.08, 0, 0) (per the given rig transform).
        origin_left_ref, _  = arm_to_ref(np.zeros(3), np.array([0, 0, 0, 1]), T_REF_LEFT)
        origin_right_ref, _ = arm_to_ref(np.zeros(3), np.array([0, 0, 0, 1]), T_REF_RIGHT)
        print(f"[Sanity check] LEFT  base_link origin in REFERENCE frame: "
              f"{np.round(origin_left_ref, 4)} m (expected [0, 0, 0])")
        print(f"[Sanity check] RIGHT base_link origin in REFERENCE frame: "
              f"{np.round(origin_right_ref, 4)} m (expected [1.08, 0, 0])")
 
        result = stream_until_reachable(arm_left, arm_right)
        if result is None:
            print("No reachable marker detected before stream was stopped -- exiting.")
            return 1
 
        chosen_arm, marker_pos_local, marker_rot_local = result
        ok = run_pick_sequence(chosen_arm, marker_pos_local, marker_rot_local)
        return 0 if ok else 1
 
 
if __name__ == "__main__":
    raise SystemExit(main())
