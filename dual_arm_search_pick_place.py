#!/usr/bin/env python3
"""
dual_arm_search_pick_place.py -- Two Kinova Gen3 Lite (6-DOF) arms, LEFT +
RIGHT: home to a predefined joint pose, then ACTIVELY SEARCH for the ArUco
marker by sweeping each arm's OWN joint_1 back and forth, instead of
streaming from a single stationary pose.
 
Reuses dual_arm_pick_place.py unchanged for everything except "how do we
find the marker": kinematics, ArmHandle, ArUco detection, the
reachable/closer-arm selection logic, and the full pick-and-place sequence
all come straight from that module (imported as `dap`).
 
Flow
----
1. Connect RIGHT (192.168.2.10) + LEFT (192.168.3.11) via utilities_dual.py.
2. Home BOTH arms to JOINT_START_LEFT / JOINT_START_RIGHT (angular Action
   move, same mechanism as 1.py's example_angular_action_movement).
3. Sweep joint_1 on BOTH arms simultaneously, +/-SEARCH_LIMIT_DEG around
   each arm's own start joint_1 angle, reversing direction at that limit.
   Every loop iteration, run ArUco detection on both cameras (dap._detect_marker).
4. On first detection (either camera): stop BOTH arms' sweep, localize the
   marker in BOTH arms' own base_link frames (dap.arm_to_ref / dap.ref_to_arm)
   and pick the reachable/closer arm -- identical selection rule to
   dual_arm_pick_place.py's stream_until_reachable.
     - Neither arm can reach it -> resume sweeping, keep searching.
5. Run dap.run_pick_sequence(...) with the selected arm.
 
Safety: ENABLE_MOTION / ENABLE_GRIPPER live in dual_arm_pick_place.py
(imported as dap) -- flip those there, not here.
"""
 
import os, sys, time, math, threading
import numpy as np
import cv2
 
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
 
sys.path.insert(0, os.path.dirname(__file__))
import dual_arm_pick_place as dap
 
# ── PER-ARM PREDEFINED SEARCH-START JOINT ANGLES (degrees) -- TUNE: a pose
# where each arm's own camera has a clear view of the shared workspace
# before sweeping joint_1. Defaults to 1.py's example angles for both arms;
# retune independently per arm once you see each camera's view from there. ─
JOINT_START_LEFT  = [288.53, 342.95, 70.14, 256.05, 305.93, 6.00]
JOINT_START_RIGHT = [288.53, 342.95, 70.14, 256.05, 305.93, 6.00]
 
HOME_ACTION_TIMEOUT = 5  # seconds, TUNE (matches 1.py's TIMEOUT_DURATION)
 
# ── JOINT_1 SEARCH SWEEP SETTINGS ──────────────────────────────────────────
SEARCH_SPEED_DEG = 8.0    # deg/s, TUNE: joint_1 sweep angular speed
SEARCH_LIMIT_DEG = 120.0   # deg, TUNE: sweep +/- this many degrees from each arm's own start joint_1
SEARCH_LOOP_HZ   = 10.0
 
# TUNE: per-joint tolerance used as a fallback "close enough" check when the
# reach_joint_angles Action times out without ever firing ACTION_END -- this
# happens when the arm settles a fraction of a degree short of the exact
# target and the servo never reports itself as converged. If every joint is
# within this many degrees of the target, homing is treated as complete.
HOME_POS_TOL_DEG = 5.0
 
 
def _wrap_to_pi(rad: float) -> float:
    return (rad + math.pi) % (2 * math.pi) - math.pi
 
 
def _angle_diff_deg(a_deg: float, b_deg: float) -> float:
    return abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)
 
 
def _check_for_end_or_abort(e: threading.Event):
    def check(notification, e=e):
        print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
            e.set()
    return check
 
 
def home_to_joint_angles(arm: "dap.ArmHandle", joint_angles_deg) -> bool:
    """
    Blocking angular Action move to `joint_angles_deg` -- same mechanism as
    1.py's example_angular_action_movement. If the Action times out without
    ever firing ACTION_END, falls back to checking the arm's actual joint
    feedback against the target within HOME_POS_TOL_DEG before giving up --
    exact convergence isn't required to start the joint_1 search sweep.
    """
    name = arm.name
    print(f"[{name}] Homing to search-start joint angles: {np.round(joint_angles_deg, 2)} deg ...")
 
    if not dap.ENABLE_MOTION:
        print(f"  [{name} DRY-RUN] ENABLE_MOTION=False -- would home to {joint_angles_deg}")
        return True
 
    action = Base_pb2.Action()
    action.name = f"{name} search-start home"
    action.application_data = ""
 
    for joint_id, angle in enumerate(joint_angles_deg):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = joint_id
        joint_angle.value = float(angle)
 
    e = threading.Event()
    notification_handle = None
    finished = False
    try:
        notification_handle = arm.base.OnNotificationActionTopic(
            _check_for_end_or_abort(e), Base_pb2.NotificationOptions()
        )
        arm.base.ExecuteAction(action)
        finished = e.wait(HOME_ACTION_TIMEOUT)
    except Exception as exc:
        print(f"[{name}] ERROR while executing home action ({exc!r}) -- "
              f"falling back to a joint-feedback check.")
    finally:
        if notification_handle is not None:
            try:
                arm.base.Unsubscribe(notification_handle)
            except Exception as exc:
                print(f"[{name}] WARNING: Unsubscribe failed, likely a transient "
                      f"connection hiccup ({exc!r}) -- continuing anyway.")
 
    if finished:
        print(f"[{name}] Home complete.")
        return True
 
    try:
        fb = arm.base_cyclic.RefreshFeedback()
        cur_deg = [fb.actuators[i].position for i in range(6)]
    except Exception as exc:
        print(f"[{name}] ERROR: could not read joint feedback to verify home position "
              f"({exc!r}) -- treating as failed.")
        return False
 
    diffs = [_angle_diff_deg(cur_deg[i], joint_angles_deg[i]) for i in range(6)]
    max_diff = max(diffs)
 
    if max_diff <= HOME_POS_TOL_DEG:
        print(f"[{name}] Home TIMEOUT after {HOME_ACTION_TIMEOUT}s, but within tolerance "
              f"(max joint error {max_diff:.2f} deg <= {HOME_POS_TOL_DEG} deg) -- treating as complete.")
        return True
 
    print(f"[{name}] Home TIMEOUT after {HOME_ACTION_TIMEOUT}s and NOT within tolerance "
          f"(max joint error {max_diff:.2f} deg > {HOME_POS_TOL_DEG} deg) -- treating as failed.")
    return False
 
 
def home_both_arms(arm_left: "dap.ArmHandle", arm_right: "dap.ArmHandle") -> bool:
    """Home both arms to their JOINT_START_* angles CONCURRENTLY (one thread
    per arm) so neither waits idle for the other's move to finish."""
    results = {}
 
    def _run(arm, angles):
        try:
            results[arm.name] = home_to_joint_angles(arm, angles)
        except Exception as exc:
            print(f"[{arm.name}] Unexpected error during homing ({exc!r}) -- treating as failed.")
            results[arm.name] = False
 
    t_left  = threading.Thread(target=_run, args=(arm_left, JOINT_START_LEFT))
    t_right = threading.Thread(target=_run, args=(arm_right, JOINT_START_RIGHT))
    t_left.start(); t_right.start()
    t_left.join();  t_right.join()
 
    return results.get("LEFT", False) and results.get("RIGHT", False)
 
 
class _Joint1Sweeper:
    """
    Drives ONE arm's joint_1 back and forth at SEARCH_SPEED_DEG, within
    +/-SEARCH_LIMIT_DEG of that arm's own start joint_1 angle (also clipped
    to dap.JOINT_MIN/MAX[0] with dap.LIMIT_MARGIN, so the sweep never faults
    the arm against its own hard joint_1 limit).
    """
 
    def __init__(self, arm: "dap.ArmHandle", start_joint1_deg: float):
        self.arm = arm
        self.direction = 1.0
        self.msg = Base_pb2.JointSpeeds()
        center = _wrap_to_pi(math.radians(start_joint1_deg))
        self.lo = max(center - math.radians(SEARCH_LIMIT_DEG), dap.JOINT_MIN[0] + dap.LIMIT_MARGIN)
        self.hi = min(center + math.radians(SEARCH_LIMIT_DEG), dap.JOINT_MAX[0] - dap.LIMIT_MARGIN)
 
    def step(self) -> None:
        q = self.arm.get_q()
        j1 = q[0]
        if j1 >= self.hi and self.direction > 0:
            self.direction = -1.0
            print(f"[{self.arm.name}] joint_1 sweep: reached +limit, reversing.")
        elif j1 <= self.lo and self.direction < 0:
            self.direction = 1.0
            print(f"[{self.arm.name}] joint_1 sweep: reached -limit, reversing.")
 
        dq_deg = np.zeros(6)
        dq_deg[0] = self.direction * SEARCH_SPEED_DEG
        self.arm.send(self.msg, dq_deg)
 
    def stop(self) -> None:
        self.arm.send(self.msg, np.zeros(6))
 
 
def search_and_select(arm_left: "dap.ArmHandle", arm_right: "dap.ArmHandle"):
    """
    Sweeps joint_1 on BOTH arms while streaming both cameras. Whenever the
    marker is detected in either camera, stops BOTH arms' sweep and applies
    the SAME reachable/closer-arm selection rule as
    dual_arm_pick_place.stream_until_reachable.
 
      - Neither arm can reach it -> resume sweeping, keep searching.
      - One arm can reach it     -> that arm is selected.
      - Both can reach it        -> whichever arm's own base_link is
                                     physically closer to the marker wins.
 
    Returns (chosen_arm, marker_pos_local, marker_rot_local) -- pos/rot
    already in chosen_arm's OWN base_link frame -- or None if 'q' was
    pressed before a reachable detection.
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
 
    sweepers = {
        arm_left.name:  _Joint1Sweeper(arm_left,  JOINT_START_LEFT[0]),
        arm_right.name: _Joint1Sweeper(arm_right, JOINT_START_RIGHT[0]),
    }
 
    print(f"Sweeping joint_1 on both arms (+/-{SEARCH_LIMIT_DEG:.0f} deg @ {SEARCH_SPEED_DEG:.1f} deg/s), "
          f"searching for a marker within either arm's {dap.WORKSPACE_XY_LIMIT}m workspace. Press 'q' to abort.")
 
    dt_nominal = 1.0 / SEARCH_LOOP_HZ
    try:
        while True:
            t_loop = time.time()
            for arm, other in ((arm_left, arm_right), (arm_right, arm_left)):
                cap = caps[arm.name]
                if not cap.isOpened():
                    continue
 
                sweepers[arm.name].step()
 
                ret, frame = cap.read()
                if not ret:
                    continue
 
                found, pos_own, rot_own = dap._detect_marker(arm, frame)
                cv2.imshow(arm.name, frame)
 
                if not found:
                    continue
 
                for s in sweepers.values():
                    s.stop()
 
                quat_own = dap.R.from_matrix(rot_own).as_quat()
                pos_ref, quat_ref = dap.arm_to_ref(pos_own, quat_own, arm.T_ref_arm)
                pos_other, quat_other = dap.ref_to_arm(pos_ref, quat_ref, other.T_ref_arm)
                rot_other = dap.R.from_quat(quat_other).as_matrix()
 
                local_poses = {arm.name: (pos_own, rot_own), other.name: (pos_other, rot_other)}
                reachable = {n: dap.in_workspace(p) for n, (p, _) in local_poses.items()}
 
                print(f"[{arm.name}] Marker detected while sweeping -- "
                      f"{arm.name} local pos={np.round(pos_own, 4)} m (reachable={reachable[arm.name]}), "
                      f"{other.name} local pos={np.round(pos_other, 4)} m (reachable={reachable[other.name]})")
 
                candidates = [n for n, ok in reachable.items() if ok]
                if not candidates:
                    print(f"  -> outside BOTH arms' {dap.WORKSPACE_XY_LIMIT}m workspace -- resuming search.")
                    continue
 
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
                print("\n'q' pressed -- aborting search.")
                return None
 
            sleep = dt_nominal - (time.time() - t_loop)
            if sleep > 0.0:
                time.sleep(sleep)
    finally:
        for s in sweepers.values():
            s.stop()
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()
 
 
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
 
        arm_left = dap.ArmHandle(base_left, base_cyclic_left, dap.T_REF_LEFT, "LEFT",
                                  camera_index=dap.CAMERA_INDEX_LEFT, camera_matrix=dap.CAMERA_MATRIX_LEFT,
                                  dist_coeffs=dap.DIST_COEFFS_LEFT,
                                  approach_offset=dap.APPROACH_OFFSET_LEFT,
                                  place_pos=dap.PLACE_POS, place_quat=dap.PLACE_QUAT)
        arm_right = dap.ArmHandle(base_right, base_cyclic_right, dap.T_REF_RIGHT, "RIGHT",
                                   camera_index=dap.CAMERA_INDEX_RIGHT, camera_matrix=dap.CAMERA_MATRIX_RIGHT,
                                   dist_coeffs=dap.DIST_COEFFS_RIGHT,
                                   approach_offset=dap.APPROACH_OFFSET_RIGHT,
                                   place_pos=dap.PLACE_POS, place_quat=dap.PLACE_QUAT)
 
        if not home_both_arms(arm_left, arm_right):
            print("One or both arms failed to reach their search-start joint angles -- aborting.")
            return 1
 
        result = search_and_select(arm_left, arm_right)
        if result is None:
            print("No reachable marker detected before search was stopped -- exiting.")
            return 1
 
        chosen_arm, marker_pos_local, marker_rot_local = result
        ok = dap.run_pick_sequence(chosen_arm, marker_pos_local, marker_rot_local)
        return 0 if ok else 1
 
 
if __name__ == "__main__":
    raise SystemExit(main())
