"""
Experimental UR3a 3-camera image-based extrinsic + DH calibration.

Purpose
-------
This is a FUN/EXPERIMENTAL script to test what happens when the optimizer is given
all camera images directly, not only pre-triangulated checkerboard poses.

It uses:
- cam1/cam2/cam3 images
- each camera intrinsics
- current tf_base_to_camera.json
- robot joint data from dataset_ur3a_joint_images_40poses.json
- fixed T_EE_CB
- nominal DH

It optimizes:
1) small camera extrinsic corrections for cam1/cam2/cam3
2) robot joint zero offsets
3) DH a_i and d_i deltas

Core residual:
For each pose and camera, predict board pose from robot FK:
    T_base_CB_pred = FK(q, DH_cal) @ T_EE_CB
Then project the board corners into each camera using refined T_base_cam:
    T_cam_CB_pred = inv(T_base_cam_refined) @ T_base_CB_pred
and minimize the difference to detected checkerboard corners.

This directly feeds the corresponding pictures to the optimizer.

Recommended run:
    python run_3cam_image_based_extrinsic_dh_experiment.py

Outputs:
    3cam_experiment/results/05_image_based_extrinsic_dh_experiment/
        image_based_experiment_report_3cam.json
        dh_calibrated_3cam_image_based.json
        tf_base_to_camera_image_based_refined.json
        accepted_poses_image_based_3cam.csv
        rejected_poses_image_based_3cam.csv

Important:
- It does NOT overwrite shared_inputs/tf_base_to_camera.json.
- Keep your robust triangulation result as the stable thesis result.
- Use this as an experimental comparison.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# =============================================================================
# USER-EDITABLE CONFIG
# =============================================================================

ROOT = Path(r"C:\Users\kaysa\OneDrive\Desktop\robot-hand-eye-calibration-experiments\03_extrinsic_calibration")
REPO_ROOT = ROOT.parents[0]

PATHS = {
    "dataset": ROOT / "shared_inputs" / "dataset_ur3a_joint_images_40poses.json",
    "dh_nominal": ROOT / "shared_inputs" / "robot_model_dh_nominal_ur3a.json",
    "ee_cb": ROOT / "shared_inputs" / "tf_ee_to_cb.json",
    "cam_base": ROOT / "shared_inputs" / "tf_base_to_camera.json",
    "results_dir": ROOT / "3cam_experiment" / "results" / "05_image_based_extrinsic_dh_experiment",
}

CAMERAS = {
    "cam1": {
        "idx": 1,
        "images_dir": ROOT / "images" / "cam1_images",
        "intrinsics": REPO_ROOT / "02_intrinsic_calibration" / "cam1" / "04_results" / "intrinsics_cam1.json",
        "corner_rotation_mode": "top_left",  # cam1 convention from old validated pipeline
    },
    "cam2": {
        "idx": 2,
        "images_dir": ROOT / "images" / "cam2_images",
        "intrinsics": REPO_ROOT / "02_intrinsic_calibration" / "cam2" / "04_results" / "intrinsics_cam2.json",
        "corner_rotation_mode": 2,           # from previous successful 3-cam run
    },
    "cam3": {
        "idx": 3,
        "images_dir": ROOT / "images" / "cam3_images",
        "intrinsics": REPO_ROOT / "02_intrinsic_calibration" / "cam3" / "04_results" / "intrinsics_cam3.json",
        "corner_rotation_mode": 2,           # from previous successful 3-cam run
    },
}

# Dataset split
TRAIN_RATIO = 0.70
SEED = 7

# Known outliers from robust analysis
EXCLUDE_POSE_IDS = {25, 26}

# Image detection filtering
MAX_SINGLE_CAM_PNP_RMSE_PX = 3.0   # only for sanity diagnostics, not the main optimized residual
REQUIRE_ALL_CAMERAS = True

# Optimization bounds: keep these flexible for experiments.
# Start conservative, then try larger values if needed.
CAM_ROT_BOUND_DEG = 2.0
CAM_TRANS_BOUND_MM = 10.0
JOINT_OFF_BOUND_DEG = 1.0
DH_BOUND_MM = 2.0

# Loss settings
ROBUST_LOSS = "soft_l1"       # scipy options: linear, soft_l1, huber, cauchy, arctan
ROBUST_F_SCALE_PX = 3.0       # pixel scale for image residuals
MAX_NFEV = 500

# Residual weights
PIXEL_RESIDUAL_WEIGHT = 1.0
PARAMETER_REGULARIZATION_WEIGHT = 0.05  # weakly discourages unnecessary extrinsic/DH drift

# Validation: project all board corners on held-out poses and report pixel RMSE.


# =============================================================================
# Helpers
# =============================================================================

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def hat(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = w
    return np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]], dtype=float)


def rodrigues(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3) + hat(w)
    k = w / theta
    K = hat(k)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def se3_from_xi(xi: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = rodrigues(xi[:3])
    T[:3, 3] = xi[3:]
    return T


def rot_angle(R: np.ndarray) -> float:
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    return math.acos(c)


def dh_A(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,     sa,      ca,      d],
        [0.0,    0.0,     0.0,    1.0],
    ], dtype=float)


def fk_from_dh(joints_rad: np.ndarray, dh_params: List[dict], joint_zero_offsets: Optional[np.ndarray] = None) -> np.ndarray:
    if joint_zero_offsets is None:
        joint_zero_offsets = np.zeros(6, dtype=float)
    T = np.eye(4, dtype=float)
    for q, off, p in zip(joints_rad, joint_zero_offsets, dh_params):
        theta = float(q + off + p.get("theta_offset", 0.0))
        T = T @ dh_A(float(p["a"]), float(p["alpha"]), float(p["d"]), theta)
    return T


def apply_dh_deltas(dh_nominal: List[dict], delta_a: np.ndarray, delta_d: np.ndarray) -> List[dict]:
    out = []
    for i, p in enumerate(dh_nominal):
        q = dict(p)
        q["a"] = float(p["a"] + delta_a[i])
        q["d"] = float(p["d"] + delta_d[i])
        out.append(q)
    return out


def make_object_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    obj = np.zeros((cols * rows, 3), dtype=np.float64)
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    obj[:, 0] = xs.reshape(-1) * square_size_m
    obj[:, 1] = ys.reshape(-1) * square_size_m
    return obj


def metric_stats(x: List[float] | np.ndarray) -> Dict[str, Optional[float]]:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"N": 0, "mean": None, "median": None, "rmse": None, "max": None, "min": None}
    return {
        "N": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "rmse": float(np.sqrt(np.mean(arr * arr))),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


# =============================================================================
# Camera / detection
# =============================================================================

@dataclass
class Intrinsics:
    camera_id: str
    K_calib: np.ndarray
    dist: np.ndarray
    calib_wh: Tuple[int, int]
    cols: int
    rows: int
    square_size_m: float


def load_intrinsics(path: Path) -> Intrinsics:
    d = load_json(path)
    return Intrinsics(
        camera_id=str(d.get("camera_id", path.stem)),
        K_calib=np.array(d["intrinsics_K"]["matrix_3x3"], dtype=np.float64),
        dist=np.array(d["distortion"]["coefficients"], dtype=np.float64).reshape(-1, 1),
        calib_wh=(int(d["image_size_wh"]["width"]), int(d["image_size_wh"]["height"])),
        cols=int(d["pattern_size_inner_corners"]["cols"]),
        rows=int(d["pattern_size_inner_corners"]["rows"]),
        square_size_m=float(d.get("square_size_m", 0.02)),
    )


def scale_K(K: np.ndarray, calib_wh: Tuple[int, int], actual_wh: Tuple[int, int]) -> np.ndarray:
    cw, ch = calib_wh
    aw, ah = actual_wh
    sx = aw / float(cw)
    sy = ah / float(ch)
    K2 = K.copy().astype(np.float64)
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def reorder_top_left(corners: np.ndarray, cols: int, rows: int) -> np.ndarray:
    grid = corners.reshape(rows, cols, 1, 2)[:, :, 0, :]
    best = None
    best_score = float("inf")
    for k in range(4):
        g = np.rot90(grid, k=k)
        g_rc = g if (g.shape[0] == rows and g.shape[1] == cols) else np.transpose(g, (1, 0, 2))
        score = float(g_rc[0, 0, 0] + g_rc[0, 0, 1])
        if score < best_score:
            best_score = score
            best = g_rc.reshape(-1, 1, 2).astype(np.float32)
    return best


def rotate_corners(corners: np.ndarray, cols: int, rows: int, rotation_index: int) -> np.ndarray:
    grid = corners.reshape(rows, cols, 1, 2)[:, :, 0, :]
    g = np.rot90(grid, k=rotation_index)
    g_rc = g if (g.shape[0] == rows and g.shape[1] == cols) else np.transpose(g, (1, 0, 2))
    return g_rc.reshape(-1, 1, 2).astype(np.float32)


def image_path_for_pose(cam_id: str, pose_id: int) -> Path:
    cfg = CAMERAS[cam_id]
    return cfg["images_dir"] / f"img{pose_id}_cam{cfg['idx']}.png"


def detect_corners_for_image(img_path: Path, intr: Intrinsics, mode) -> Tuple[np.ndarray, Tuple[int, int], float, np.ndarray, np.ndarray]:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray,
        (intr.cols, intr.rows),
        flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
    )
    if (not found) or corners is None or len(corners) != intr.cols * intr.rows:
        raise RuntimeError(f"checkerboard not detected in {img_path.name}")

    if mode == "top_left":
        corners = reorder_top_left(corners, intr.cols, intr.rows)
    elif isinstance(mode, int):
        corners = rotate_corners(corners, intr.cols, intr.rows, mode)
    else:
        raise ValueError(f"Unknown corner rotation mode: {mode}")

    K_used = scale_K(intr.K_calib, intr.calib_wh, (w, h))
    obj = make_object_points(intr.cols, intr.rows, intr.square_size_m)
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K_used, intr.dist, flags=cv2.SOLVEPNP_ITERATIVE)
    rmse = float("nan")
    if ok:
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K_used, intr.dist)
        d = proj.reshape(-1, 2) - corners.reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.sum(d * d, axis=1))))
    return corners.reshape(-1, 2).astype(np.float64), (w, h), rmse, K_used, intr.dist


def project_points(obj_pts: np.ndarray, T_base_CB: np.ndarray, T_base_cam: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    T_cam_CB = inv_T(T_base_cam) @ T_base_CB
    R = T_cam_CB[:3, :3]
    t = T_cam_CB[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(obj_pts.astype(np.float64), rvec, t, K, dist)
    return proj.reshape(-1, 2)


@dataclass
class CameraObservation:
    cam_id: str
    image: str
    corners_px: np.ndarray
    K: np.ndarray
    dist: np.ndarray
    pnp_rmse_px: float
    image_size_wh: Tuple[int, int]


@dataclass
class PoseSample:
    pose_id: int
    joints_rad: np.ndarray
    observations: Dict[str, CameraObservation]


# =============================================================================
# Build dataset
# =============================================================================

def build_samples() -> Tuple[List[PoseSample], List[dict], Dict[str, Intrinsics], np.ndarray]:
    dataset = load_json(PATHS["dataset"])
    intrinsics = {cid: load_intrinsics(cfg["intrinsics"]) for cid, cfg in CAMERAS.items()}

    # enforce same board settings across cameras
    ref = next(iter(intrinsics.values()))
    obj_pts = make_object_points(ref.cols, ref.rows, ref.square_size_m)

    accepted: List[PoseSample] = []
    rejected: List[dict] = []

    for pose in dataset["poses"]:
        pid = int(pose["pose_id"])
        if pid in EXCLUDE_POSE_IDS:
            rejected.append({"pose_id": pid, "reason": "explicitly_excluded"})
            continue

        joints_deg = [float(pose["joint_angles_deg"][f"joint_{i}"]) for i in range(1, 7)]
        joints_rad = np.deg2rad(np.array(joints_deg, dtype=float))

        obs: Dict[str, CameraObservation] = {}
        fail_reasons = []
        for cam_id, cfg in CAMERAS.items():
            try:
                img_path = image_path_for_pose(cam_id, pid)
                intr = intrinsics[cam_id]
                corners, wh, pnp_rmse, K_used, dist = detect_corners_for_image(img_path, intr, cfg["corner_rotation_mode"])
                obs[cam_id] = CameraObservation(
                    cam_id=cam_id,
                    image=img_path.name,
                    corners_px=corners,
                    K=K_used,
                    dist=dist,
                    pnp_rmse_px=float(pnp_rmse),
                    image_size_wh=wh,
                )
            except Exception as e:
                fail_reasons.append(f"{cam_id}: {e}")

        if REQUIRE_ALL_CAMERAS and len(obs) != len(CAMERAS):
            rejected.append({"pose_id": pid, "reason": "; ".join(fail_reasons)})
            continue
        if any((o.pnp_rmse_px > MAX_SINGLE_CAM_PNP_RMSE_PX) for o in obs.values()):
            # This should normally not happen; PnP RMSE is a detection sanity check.
            bad = [f"{cid}:{o.pnp_rmse_px:.3f}px" for cid, o in obs.items() if o.pnp_rmse_px > MAX_SINGLE_CAM_PNP_RMSE_PX]
            rejected.append({"pose_id": pid, "reason": "high_single_cam_pnp_rmse " + ",".join(bad)})
            continue

        accepted.append(PoseSample(pose_id=pid, joints_rad=joints_rad, observations=obs))
        print(f"pose {pid:02d}: accepted | PnP rmse: " + ", ".join(f"{cid}={obs[cid].pnp_rmse_px:.3f}px" for cid in sorted(obs)))

    return accepted, rejected, intrinsics, obj_pts


# =============================================================================
# Optimization and evaluation
# =============================================================================

def make_train_test(samples: List[PoseSample]) -> Tuple[List[PoseSample], List[PoseSample]]:
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_train = int(round(TRAIN_RATIO * len(samples)))
    train = [samples[i] for i in idx[:n_train]]
    test = [samples[i] for i in idx[n_train:]]
    return train, test


def load_base_transforms() -> Dict[str, np.ndarray]:
    cam_json = load_json(PATHS["cam_base"])
    out = {}
    for cam_id in CAMERAS:
        out[cam_id] = np.array(cam_json["T_base_to_camera_extrinsics"][cam_id]["T_4x4"], dtype=float)
    return out


def pack_params(j_off: np.ndarray, da: np.ndarray, dd: np.ndarray, cam_xis: Dict[str, np.ndarray]) -> np.ndarray:
    xs = [j_off, da, dd]
    for cid in CAMERAS.keys():
        xs.append(cam_xis[cid])
    return np.concatenate(xs)


def unpack_params(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    j = x[0:6]
    da = x[6:12]
    dd = x[12:18]
    cam_xis = {}
    k = 18
    for cid in CAMERAS.keys():
        cam_xis[cid] = x[k:k+6]
        k += 6
    return j, da, dd, cam_xis


def refined_camera_transforms(base: Dict[str, np.ndarray], cam_xis: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {cid: se3_from_xi(cam_xis[cid]) @ base[cid] for cid in base.keys()}


def residuals_image_based(
    x: np.ndarray,
    samples: List[PoseSample],
    dh_nominal: List[dict],
    T_EE_CB: np.ndarray,
    T_base_cam_initial: Dict[str, np.ndarray],
    obj_pts: np.ndarray,
) -> np.ndarray:
    j_off, da, dd, cam_xis = unpack_params(x)
    dh_cur = apply_dh_deltas(dh_nominal, da, dd)
    cams = refined_camera_transforms(T_base_cam_initial, cam_xis)

    res = []
    for s in samples:
        T_base_CB = fk_from_dh(s.joints_rad, dh_cur, j_off) @ T_EE_CB
        for cid, obs in s.observations.items():
            proj = project_points(obj_pts, T_base_CB, cams[cid], obs.K, obs.dist)
            r = (proj - obs.corners_px).reshape(-1) * PIXEL_RESIDUAL_WEIGHT
            res.append(r)

    # weak regularization to keep the optimizer from inventing unnecessary motion
    if PARAMETER_REGULARIZATION_WEIGHT > 0:
        res.append(PARAMETER_REGULARIZATION_WEIGHT * x)

    return np.concatenate(res)


def evaluate_image_projection(
    samples: List[PoseSample],
    dh_params: List[dict],
    T_EE_CB: np.ndarray,
    cams: Dict[str, np.ndarray],
    obj_pts: np.ndarray,
    joint_zero_offsets: Optional[np.ndarray] = None,
) -> dict:
    per_pose = []
    all_rmse = []
    for s in samples:
        T_base_CB = fk_from_dh(s.joints_rad, dh_params, joint_zero_offsets) @ T_EE_CB
        per_cam = {}
        for cid, obs in s.observations.items():
            proj = project_points(obj_pts, T_base_CB, cams[cid], obs.K, obs.dist)
            d = proj - obs.corners_px
            rmse = float(np.sqrt(np.mean(np.sum(d * d, axis=1))))
            per_cam[cid] = rmse
            all_rmse.append(rmse)
        per_pose.append({
            "pose_id": s.pose_id,
            "per_cam_reproj_rmse_px": per_cam,
            "mean_reproj_rmse_px": float(np.mean(list(per_cam.values()))),
            "max_reproj_rmse_px": float(np.max(list(per_cam.values()))),
        })
    return {
        "N_poses": int(len(samples)),
        "N_camera_observations": int(len(all_rmse)),
        "camera_observation_reproj_rmse_px": metric_stats(all_rmse),
        "pose_mean_reproj_rmse_px": metric_stats([p["mean_reproj_rmse_px"] for p in per_pose]),
        "pose_max_reproj_rmse_px": metric_stats([p["max_reproj_rmse_px"] for p in per_pose]),
        "per_pose": per_pose,
    }


def run_optimization(samples: List[PoseSample], obj_pts: np.ndarray) -> dict:
    try:
        from scipy.optimize import least_squares
    except Exception as e:
        raise RuntimeError(f"SciPy is required. Install with: python -m pip install scipy. Error: {e}")

    dh_json = load_json(PATHS["dh_nominal"])
    ee_json = load_json(PATHS["ee_cb"])
    dh_nominal = dh_json["joints"]
    T_EE_CB = np.array(ee_json["transformation_matrix_T_EE_CB_4x4"], dtype=float)
    T_base_cam_initial = load_base_transforms()

    train, test = make_train_test(samples)

    zero_xis = {cid: np.zeros(6, dtype=float) for cid in CAMERAS}
    x0 = pack_params(np.zeros(6), np.zeros(6), np.zeros(6), zero_xis)

    j_b = math.radians(JOINT_OFF_BOUND_DEG)
    dh_b = DH_BOUND_MM / 1000.0
    cam_r_b = math.radians(CAM_ROT_BOUND_DEG)
    cam_t_b = CAM_TRANS_BOUND_MM / 1000.0

    lb_parts = [-np.ones(6) * j_b, -np.ones(6) * dh_b, -np.ones(6) * dh_b]
    ub_parts = [ np.ones(6) * j_b,  np.ones(6) * dh_b,  np.ones(6) * dh_b]
    for _ in CAMERAS:
        lb_parts.append(np.array([-cam_r_b, -cam_r_b, -cam_r_b, -cam_t_b, -cam_t_b, -cam_t_b], dtype=float))
        ub_parts.append(np.array([ cam_r_b,  cam_r_b,  cam_r_b,  cam_t_b,  cam_t_b,  cam_t_b], dtype=float))
    lb = np.concatenate(lb_parts)
    ub = np.concatenate(ub_parts)

    baseline_train = evaluate_image_projection(train, dh_nominal, T_EE_CB, T_base_cam_initial, obj_pts, None)
    baseline_test = evaluate_image_projection(test, dh_nominal, T_EE_CB, T_base_cam_initial, obj_pts, None)

    print("\n================ BASELINE IMAGE REPROJECTION ================")
    print("train pose mean RMSE:", baseline_train["pose_mean_reproj_rmse_px"]["rmse"])
    print("test  pose mean RMSE:", baseline_test["pose_mean_reproj_rmse_px"]["rmse"])

    print("\n================ OPTIMIZING IMAGE-BASED EXTRINSICS + DH ================")
    sol = least_squares(
        lambda x: residuals_image_based(x, train, dh_nominal, T_EE_CB, T_base_cam_initial, obj_pts),
        x0,
        bounds=(lb, ub),
        loss=ROBUST_LOSS,
        f_scale=ROBUST_F_SCALE_PX,
        max_nfev=MAX_NFEV,
        verbose=1,
    )

    j_off, da, dd, cam_xis = unpack_params(sol.x)
    dh_cal = apply_dh_deltas(dh_nominal, da, dd)
    cams_refined = refined_camera_transforms(T_base_cam_initial, cam_xis)

    calibrated_train = evaluate_image_projection(train, dh_cal, T_EE_CB, cams_refined, obj_pts, j_off)
    calibrated_test = evaluate_image_projection(test, dh_cal, T_EE_CB, cams_refined, obj_pts, j_off)

    # Extra transform diagnostics
    cam_refine_report = {}
    for cid in CAMERAS:
        xi = cam_xis[cid]
        cam_refine_report[cid] = {
            "xi_rot_deg": (xi[:3] * 180.0 / math.pi).tolist(),
            "xi_trans_mm": (xi[3:] * 1000.0).tolist(),
            "rotation_update_norm_deg": float(np.linalg.norm(xi[:3]) * 180.0 / math.pi),
            "translation_update_norm_mm": float(np.linalg.norm(xi[3:]) * 1000.0),
            "T_base_cam_refined_4x4": cams_refined[cid].tolist(),
            "T_base_cam_initial_4x4": T_base_cam_initial[cid].tolist(),
        }

    return {
        "settings": {
            "TRAIN_RATIO": TRAIN_RATIO,
            "SEED": SEED,
            "EXCLUDE_POSE_IDS": sorted(EXCLUDE_POSE_IDS),
            "CAM_ROT_BOUND_DEG": CAM_ROT_BOUND_DEG,
            "CAM_TRANS_BOUND_MM": CAM_TRANS_BOUND_MM,
            "JOINT_OFF_BOUND_DEG": JOINT_OFF_BOUND_DEG,
            "DH_BOUND_MM": DH_BOUND_MM,
            "ROBUST_LOSS": ROBUST_LOSS,
            "ROBUST_F_SCALE_PX": ROBUST_F_SCALE_PX,
            "PARAMETER_REGULARIZATION_WEIGHT": PARAMETER_REGULARIZATION_WEIGHT,
            "corner_rotation_modes": {cid: CAMERAS[cid]["corner_rotation_mode"] for cid in CAMERAS},
        },
        "dataset_summary": {
            "num_accepted_samples": int(len(samples)),
            "train_pose_ids": [s.pose_id for s in train],
            "test_pose_ids": [s.pose_id for s in test],
        },
        "optimization": {
            "success": bool(sol.success),
            "status": int(sol.status),
            "message": str(sol.message),
            "cost": float(sol.cost),
            "nfev": int(sol.nfev),
        },
        "baseline_train": baseline_train,
        "baseline_test": baseline_test,
        "calibrated_train": calibrated_train,
        "calibrated_test": calibrated_test,
        "joint_zero_offsets_rad": j_off.tolist(),
        "joint_zero_offsets_deg": (j_off * 180.0 / math.pi).tolist(),
        "delta_a_m": da.tolist(),
        "delta_d_m": dd.tolist(),
        "camera_refinement": cam_refine_report,
        "dh_calibrated": dh_cal,
        "refined_camera_transforms": {cid: {"T_4x4": cams_refined[cid].tolist()} for cid in CAMERAS},
    }


# =============================================================================
# Output helpers
# =============================================================================

def write_pose_csv(path: Path, samples: List[PoseSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pose_id", "cam1_pnp_rmse_px", "cam2_pnp_rmse_px", "cam3_pnp_rmse_px"])
        w.writeheader()
        for s in samples:
            row = {"pose_id": s.pose_id}
            for cid in CAMERAS:
                row[f"{cid}_pnp_rmse_px"] = s.observations[cid].pnp_rmse_px if cid in s.observations else None
            w.writerow(row)


def write_rejected_csv(path: Path, rejected: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pose_id", "reason"])
        w.writeheader()
        for r in rejected:
            w.writerow(r)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    out_dir = PATHS["results_dir"]
    ensure_dir(out_dir)

    print("\n================ BUILDING IMAGE OBSERVATIONS ================")
    samples, rejected, intrinsics, obj_pts = build_samples()
    print(f"Accepted samples: {len(samples)}")
    print(f"Rejected samples: {len(rejected)}")

    if len(samples) < 12:
        raise RuntimeError("Not enough accepted samples. Relax filters or check image paths/detection.")

    write_pose_csv(out_dir / "accepted_poses_image_based_3cam.csv", samples)
    write_rejected_csv(out_dir / "rejected_poses_image_based_3cam.csv", rejected)

    report = run_optimization(samples, obj_pts)

    # Save full report
    save_json(out_dir / "image_based_experiment_report_3cam.json", report)

    # Save calibrated DH
    dh_out = {
        "meta": {
            "robot": "UR3a",
            "dh_convention": "standard",
            "units": {"length": "m", "angle": "rad"},
            "source": "image_based_extrinsic_plus_dh_experiment_3cam",
            "warning": "Experimental result. Compare against robust triangulation result before using in thesis.",
        },
        "joints": report["dh_calibrated"],
        "joint_zero_offsets_rad": report["joint_zero_offsets_rad"],
        "joint_zero_offsets_deg": report["joint_zero_offsets_deg"],
    }
    save_json(out_dir / "dh_calibrated_3cam_image_based.json", dh_out)

    # Save refined camera transforms in same useful block style
    cam_json_original = load_json(PATHS["cam_base"])
    cam_json_exp = dict(cam_json_original)
    cam_json_exp["T_base_to_camera_extrinsics"] = dict(cam_json_original["T_base_to_camera_extrinsics"])
    for cid, block in report["refined_camera_transforms"].items():
        old_block = dict(cam_json_exp["T_base_to_camera_extrinsics"].get(cid, {}))
        old_block["T_4x4"] = block["T_4x4"]
        old_block["source"] = "image_based_extrinsic_plus_dh_experiment_3cam"
        cam_json_exp["T_base_to_camera_extrinsics"][cid] = old_block
    save_json(out_dir / "tf_base_to_camera_image_based_refined.json", cam_json_exp)

    print("\n================ EXPERIMENT SUMMARY ================")
    print("Baseline TEST pose mean reproj RMSE px:", report["baseline_test"]["pose_mean_reproj_rmse_px"]["rmse"])
    print("Calibrated TEST pose mean reproj RMSE px:", report["calibrated_test"]["pose_mean_reproj_rmse_px"]["rmse"])
    print("Camera updates:")
    for cid, cr in report["camera_refinement"].items():
        print(f"  {cid}: rot_norm={cr['rotation_update_norm_deg']:.3f} deg | trans_norm={cr['translation_update_norm_mm']:.3f} mm")
    print("\nSaved outputs in:", out_dir)


if __name__ == "__main__":
    main()
