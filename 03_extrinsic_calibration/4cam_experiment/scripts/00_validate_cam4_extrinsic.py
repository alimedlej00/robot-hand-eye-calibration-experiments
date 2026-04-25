"""
Validate / diagnose T_base_cam4 before 4-camera experiment for the 4-camera thesis experiment.

Purpose
-------
This script estimates the camera-4 extrinsic transform using ONLY cam4 images,
robot joint values, nominal/calibrated DH, and the fixed EE->checkerboard transform.

For each successful cam4 image:
    1) Detect checkerboard corners in cam4 image
    2) solvePnP gives T_cam4_CB  (checkerboard/board -> cam4)
    3) FK(q, DH) @ T_EE_CB gives T_base_CB predicted from robot
    4) Therefore each pose estimates:
          T_base_cam4_i = T_base_CB_from_robot @ inv(T_cam4_CB)
    5) Robustly average/refine T_base_cam4 over all poses

Then it compares:
    - existing T_base_cam4 from shared_inputs/tf_base_to_camera.json
    - estimated/refined T_base_cam4 from this script

Outputs
-------
03_extrinsic_calibration/4cam_experiment/results/00_validate_cam4_extrinsic/
    cam4_extrinsic_validation_report.json
    tf_base_to_camera_cam4_estimated_block.json
    cam4_per_pose_diagnostics.csv

Run
---
python 00_validate_cam4_extrinsic.py

Dependencies
------------
pip install numpy opencv-python scipy
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from scipy.optimize import least_squares
except Exception:
    least_squares = None


# =========================
# USER CONFIG
# =========================
ROOT = Path(r"C:\Users\kaysa\OneDrive\Desktop\robot-hand-eye-calibration-experiments\03_extrinsic_calibration")
REPO_ROOT = ROOT.parents[0]

CAM_ID = "cam4"
CAM_INDEX = 4
PATTERN_SIZE = (9, 7)  # cols, rows

PATHS = {
    "cam4_images": ROOT / "images" / "cam4_images",
    "intrinsics_cam4": REPO_ROOT / "02_intrinsic_calibration" / "cam4" / "04_results" / "intrinsics_cam4.json",
    "dataset": ROOT / "shared_inputs" / "dataset_ur3a_joint_images_40poses.json",
    "tf_base_to_camera": ROOT / "shared_inputs" / "tf_base_to_camera.json",
    "tf_ee_to_cb": ROOT / "shared_inputs" / "tf_ee_to_cb.json",
    "dh_nominal": ROOT / "shared_inputs" / "robot_model_dh_nominal_ur3a.json",
    # Optional: use calibrated DH if available. If missing, nominal is used.
    "dh_calibrated_1cam": ROOT / "1cam_experiment" / "results" / "02_calibration_1cam" / "dh_calibrated_1cam.json",
    "dh_calibrated_2cam": ROOT / "2cam_experiment" / "results" / "02_calibration_2cam" / "dh_calibrated_2cam.json",
    "results_dir": ROOT / "4cam_experiment" / "results" / "00_validate_cam4_extrinsic",
}

# Choose DH source:
#   "nominal"       = safest diagnostic baseline
#   "calibrated_1cam" = often useful if 1cam calibration was reliable
#   "calibrated_2cam" = use only if you trust 2cam result
DH_SOURCE = "nominal"

# PnP filter. Cam4 single-view PnP should normally be low, often <1-3 px.
MAX_PNP_RMSE_PX = 3.0

# Robust refinement settings
ROBUST_LOSS = "soft_l1"
LEVER_ARM_M = 0.15
MAX_NFEV = 500


# =========================
# BASIC HELPERS
# =========================
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
    return np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]], dtype=float)


def rodrigues(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3) + hat(w)
    k = w / theta
    K = hat(k)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def rot_log(R: np.ndarray) -> np.ndarray:
    tr = float(np.trace(R))
    c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    theta = math.acos(c)
    if theta < 1e-12:
        return np.zeros(3, dtype=float)
    W = (R - R.T) * (0.5 / math.sin(theta))
    return theta * np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=float)


def rot_angle_deg(R: np.ndarray) -> float:
    tr = float(np.trace(R))
    c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    return math.degrees(math.acos(c))


def T_from_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def se3_left_update(xi: np.ndarray, T: np.ndarray) -> np.ndarray:
    dT = np.eye(4, dtype=float)
    dT[:3, :3] = rodrigues(xi[:3])
    dT[:3, 3] = xi[3:]
    return dT @ T


def dh_A(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)


def fk_from_dh(joints_rad: np.ndarray, dh_params: List[dict], joint_zero_offsets: Optional[np.ndarray] = None) -> np.ndarray:
    if joint_zero_offsets is None:
        joint_zero_offsets = np.zeros(6, dtype=float)
    T = np.eye(4, dtype=float)
    for q, off, p in zip(joints_rad, joint_zero_offsets, dh_params):
        theta = float(q + off + p.get("theta_offset", 0.0))
        T = T @ dh_A(float(p["a"]), float(p["alpha"]), float(p["d"]), theta)
    return T


def make_object_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    obj = np.zeros((cols * rows, 3), dtype=np.float64)
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    obj[:, 0] = xs.reshape(-1) * square_size_m
    obj[:, 1] = ys.reshape(-1) * square_size_m
    return obj


def reorder_corners_assume_image_topleft(corners: np.ndarray, cols: int, rows: int) -> np.ndarray:
    # Same convention used in your earlier scripts: board origin = detected corner closest to image top-left.
    grid = corners.reshape(rows, cols, 1, 2)[:, :, 0, :]
    best = None
    best_score = float("inf")
    for k in range(4):
        g = np.rot90(grid, k=k)
        g_rc = g if (g.shape[0] == rows and g.shape[1] == cols) else np.transpose(g, (1, 0, 2))
        origin = g_rc[0, 0]
        score = float(origin[0] + origin[1])
        if score < best_score:
            best_score = score
            best = g_rc.reshape(-1, 1, 2).astype(np.float32)
    return best


def reprojection_rmse(objp: np.ndarray, corners: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray, dist: np.ndarray) -> float:
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    d = proj.reshape(-1, 2) - corners.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def scale_camera_matrix(K: np.ndarray, calib_wh: Tuple[int, int], actual_wh: Tuple[int, int]) -> Tuple[np.ndarray, Dict[str, float]]:
    cw, ch = calib_wh
    aw, ah = actual_wh
    sx = aw / float(cw)
    sy = ah / float(ch)
    K2 = K.copy().astype(np.float64)
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2, {"sx": float(sx), "sy": float(sy)}


def summarize(values: List[float]) -> Dict[str, float]:
    x = np.array(values, dtype=float)
    if len(x) == 0:
        return {"N": 0, "mean": None, "median": None, "rmse": None, "max": None, "min": None}
    return {
        "N": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "rmse": float(np.sqrt(np.mean(x * x))),
        "max": float(np.max(x)),
        "min": float(np.min(x)),
    }


# =========================
# DATA STRUCTURES
# =========================
@dataclass
class Intrinsics:
    K: np.ndarray
    dist: np.ndarray
    calib_wh: Tuple[int, int]
    cols: int
    rows: int
    square_size_m: float


@dataclass
class Cam4Sample:
    pose_id: int
    image: str
    joints_rad: np.ndarray
    T_cam_CB: np.ndarray
    T_base_CB_robot: np.ndarray
    T_base_cam_est_i: np.ndarray
    pnp_rmse_px: float


# =========================
# LOADERS
# =========================
def load_intrinsics(path: Path) -> Intrinsics:
    d = load_json(path)
    return Intrinsics(
        K=np.array(d["intrinsics_K"]["matrix_3x3"], dtype=np.float64),
        dist=np.array(d["distortion"]["coefficients"], dtype=np.float64).reshape(-1, 1),
        calib_wh=(int(d["image_size_wh"]["width"]), int(d["image_size_wh"]["height"])),
        cols=int(d["pattern_size_inner_corners"]["cols"]),
        rows=int(d["pattern_size_inner_corners"]["rows"]),
        square_size_m=float(d.get("square_size_m", 0.02)),
    )


def load_dh() -> Tuple[List[dict], np.ndarray, str]:
    if DH_SOURCE == "calibrated_1cam" and PATHS["dh_calibrated_1cam"].exists():
        d = load_json(PATHS["dh_calibrated_1cam"])
        return d["joints"], np.array(d.get("joint_zero_offsets_rad", [0.0] * 6), dtype=float), "calibrated_1cam"
    if DH_SOURCE == "calibrated_2cam" and PATHS["dh_calibrated_2cam"].exists():
        d = load_json(PATHS["dh_calibrated_2cam"])
        return d["joints"], np.array(d.get("joint_zero_offsets_rad", [0.0] * 6), dtype=float), "calibrated_2cam"
    d = load_json(PATHS["dh_nominal"])
    return d["joints"], np.zeros(6, dtype=float), "nominal"


def image_path_for_pose(pose_id: int) -> Path:
    return PATHS["cam4_images"] / f"img{pose_id}_cam4.png"


def joints_from_pose(pose: dict) -> Tuple[List[float], np.ndarray]:
    deg = [float(pose["joint_angles_deg"][f"joint_{i}"]) for i in range(1, 7)]
    rad = np.deg2rad(np.array(deg, dtype=float))
    return deg, rad


# =========================
# CORE STEPS
# =========================
def detect_and_pnp(img_path: Path, intr: Intrinsics, objp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, float], Tuple[int, int]]:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    h, w = img.shape[:2]
    K_used, scale = scale_camera_matrix(intr.K, intr.calib_wh, (w, h))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray,
        (intr.cols, intr.rows),
        flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
    )
    if (not found) or corners is None or len(corners) != intr.cols * intr.rows:
        raise RuntimeError(f"Checkerboard not detected: {img_path.name}")

    corners = reorder_corners_assume_image_topleft(corners, intr.cols, intr.rows)
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K_used, intr.dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError(f"solvePnP failed: {img_path.name}")
    rmse = reprojection_rmse(objp, corners, rvec, tvec, K_used, intr.dist)
    R, _ = cv2.Rodrigues(rvec)
    T_cam_CB = T_from_rt(R, tvec.reshape(3))
    return T_cam_CB, K_used, rmse, scale, (w, h)


def build_samples() -> Tuple[List[Cam4Sample], Dict]:
    ensure_dir(PATHS["results_dir"])
    dataset = load_json(PATHS["dataset"])
    intr = load_intrinsics(PATHS["intrinsics_cam4"])
    dh_params, joint_offsets, dh_source_used = load_dh()
    ee_cb_json = load_json(PATHS["tf_ee_to_cb"])
    T_EE_CB = np.array(ee_cb_json["transformation_matrix_T_EE_CB_4x4"], dtype=float)
    objp = make_object_points(intr.cols, intr.rows, intr.square_size_m)

    samples: List[Cam4Sample] = []
    failures = []
    pnp_rows = []

    for pose in dataset["poses"]:
        pid = int(pose["pose_id"])
        img_path = image_path_for_pose(pid)
        try:
            _, joints_rad = joints_from_pose(pose)
            T_cam_CB, K_used, rmse, scale, actual_wh = detect_and_pnp(img_path, intr, objp)
            T_base_EE = fk_from_dh(joints_rad, dh_params, joint_offsets)
            T_base_CB_robot = T_base_EE @ T_EE_CB
            T_base_cam_est_i = T_base_CB_robot @ inv_T(T_cam_CB)
            samples.append(Cam4Sample(pid, img_path.name, joints_rad, T_cam_CB, T_base_CB_robot, T_base_cam_est_i, rmse))
            print(f"pose {pid:02d}: PnP OK | rmse={rmse:.3f}px")
            pnp_rows.append({"pose_id": pid, "image": img_path.name, "pnp_rmse_px": rmse, "used": rmse <= MAX_PNP_RMSE_PX})
        except Exception as e:
            print(f"pose {pid:02d}: FAIL | {e}")
            failures.append({"pose_id": pid, "image": img_path.name, "error": str(e)})

    meta = {
        "dh_source_used": dh_source_used,
        "num_success": len(samples),
        "num_failed": len(failures),
        "failures": failures,
        "pnp_rmse_summary_px": summarize([s.pnp_rmse_px for s in samples]),
    }
    return samples, meta


def average_initial_transform(samples: List[Cam4Sample]) -> np.ndarray:
    # Robust-ish simple initial guess: median translation + iterative rotation average.
    ts = np.array([s.T_base_cam_est_i[:3, 3] for s in samples], dtype=float)
    t_med = np.median(ts, axis=0)

    R_avg = samples[0].T_base_cam_est_i[:3, :3].copy()
    for _ in range(50):
        logs = []
        for s in samples:
            R_i = s.T_base_cam_est_i[:3, :3]
            logs.append(rot_log(R_i @ R_avg.T))
        mean_log = np.mean(np.array(logs), axis=0)
        if np.linalg.norm(mean_log) < 1e-12:
            break
        R_avg = rodrigues(mean_log) @ R_avg

    T = np.eye(4, dtype=float)
    T[:3, :3] = R_avg
    T[:3, 3] = t_med
    return T


def residuals_for_T_base_cam(xi: np.ndarray, T_init: np.ndarray, samples: List[Cam4Sample]) -> np.ndarray:
    T_base_cam = se3_left_update(xi, T_init)
    out = []
    for s in samples:
        # We want: T_base_cam @ T_cam_CB == T_base_CB_robot
        T_pred = T_base_cam @ s.T_cam_CB
        T_err = inv_T(s.T_base_CB_robot) @ T_pred
        out.extend(T_err[:3, 3].tolist())
        out.extend((LEVER_ARM_M * rot_log(T_err[:3, :3])).tolist())
    return np.array(out, dtype=float)


def refine_transform(samples: List[Cam4Sample], T_init: np.ndarray) -> Tuple[np.ndarray, dict]:
    if least_squares is None:
        return T_init, {"used_scipy": False, "message": "scipy not available; used average transform only"}
    sol = least_squares(
        lambda x: residuals_for_T_base_cam(x, T_init, samples),
        np.zeros(6, dtype=float),
        loss=ROBUST_LOSS,
        max_nfev=MAX_NFEV,
        verbose=0,
    )
    return se3_left_update(sol.x, T_init), {
        "used_scipy": True,
        "success": bool(sol.success),
        "status": int(sol.status),
        "message": str(sol.message),
        "cost": float(sol.cost),
        "nfev": int(sol.nfev),
        "xi_update_rad_m": sol.x.astype(float).tolist(),
        "xi_update_rot_deg": (sol.x[:3] * 180.0 / math.pi).astype(float).tolist(),
        "xi_update_trans_mm": (sol.x[3:] * 1000.0).astype(float).tolist(),
    }


def transform_errors(T_base_cam: np.ndarray, samples: List[Cam4Sample]) -> List[dict]:
    rows = []
    for s in samples:
        T_meas = T_base_cam @ s.T_cam_CB
        T_err = inv_T(s.T_base_CB_robot) @ T_meas
        rows.append({
            "pose_id": s.pose_id,
            "image": s.image,
            "pnp_rmse_px": float(s.pnp_rmse_px),
            "e_trans_mm": float(np.linalg.norm(T_err[:3, 3]) * 1000.0),
            "e_rot_deg": float(rot_angle_deg(T_err[:3, :3])),
        })
    return rows


def compare_transforms(T_existing: np.ndarray, T_est: np.ndarray) -> dict:
    T_delta = inv_T(T_existing) @ T_est
    return {
        "translation_existing_m": T_existing[:3, 3].astype(float).tolist(),
        "translation_estimated_m": T_est[:3, 3].astype(float).tolist(),
        "translation_delta_existing_to_estimated_m": (T_est[:3, 3] - T_existing[:3, 3]).astype(float).tolist(),
        "translation_delta_existing_to_estimated_mm": ((T_est[:3, 3] - T_existing[:3, 3]) * 1000.0).astype(float).tolist(),
        "delta_transform_existing_inverse_times_estimated_4x4": T_delta.astype(float).tolist(),
        "delta_rotation_deg": float(rot_angle_deg(T_delta[:3, :3])),
        "delta_translation_norm_mm": float(np.linalg.norm(T_delta[:3, 3]) * 1000.0),
    }


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print("\n================ CAM4 BASE TRANSFORM ESTIMATION ================")
    samples_all, meta = build_samples()
    usable = [s for s in samples_all if s.pnp_rmse_px <= MAX_PNP_RMSE_PX]
    if len(usable) < 10:
        print(f"\nWARNING: only {len(usable)} poses pass MAX_PNP_RMSE_PX={MAX_PNP_RMSE_PX}. Using all successful poses instead.")
        usable = samples_all
    if len(usable) < 6:
        raise RuntimeError("Not enough cam4 PnP samples to estimate a stable transform.")

    cam_base_json = load_json(PATHS["tf_base_to_camera"])
    T_existing = np.array(cam_base_json["T_base_to_camera_extrinsics"][CAM_ID]["T_4x4"], dtype=float)

    T_avg = average_initial_transform(usable)
    T_refined, opt_info = refine_transform(usable, T_avg)

    rows_existing = transform_errors(T_existing, usable)
    rows_est = transform_errors(T_refined, usable)

    existing_summary = {
        "e_trans_mm": summarize([r["e_trans_mm"] for r in rows_existing]),
        "e_rot_deg": summarize([r["e_rot_deg"] for r in rows_existing]),
    }
    estimated_summary = {
        "e_trans_mm": summarize([r["e_trans_mm"] for r in rows_est]),
        "e_rot_deg": summarize([r["e_rot_deg"] for r in rows_est]),
    }

    # Combined per-pose CSV
    combined_rows = []
    by_est = {r["pose_id"]: r for r in rows_est}
    for r0 in rows_existing:
        r1 = by_est[r0["pose_id"]]
        combined_rows.append({
            "pose_id": r0["pose_id"],
            "image": r0["image"],
            "pnp_rmse_px": r0["pnp_rmse_px"],
            "existing_e_trans_mm": r0["e_trans_mm"],
            "estimated_e_trans_mm": r1["e_trans_mm"],
            "existing_e_rot_deg": r0["e_rot_deg"],
            "estimated_e_rot_deg": r1["e_rot_deg"],
            "trans_improvement_mm": r0["e_trans_mm"] - r1["e_trans_mm"],
            "rot_improvement_deg": r0["e_rot_deg"] - r1["e_rot_deg"],
        })

    comparison = compare_transforms(T_existing, T_refined)

    report = {
        "purpose": "Estimate corrected T_base_cam4 from cam4 PnP + robot FK + fixed T_EE_CB, then compare to existing shared tf_base_to_camera cam4.",
        "method": "For each pose: T_base_cam4_i = FK(q,DH) @ T_EE_CB @ inv(T_cam4_CB). Then average/refine T_base_cam4 robustly.",
        "settings": {
            "CAM_ID": CAM_ID,
            "DH_SOURCE_requested": DH_SOURCE,
            "MAX_PNP_RMSE_PX": MAX_PNP_RMSE_PX,
            "ROBUST_LOSS": ROBUST_LOSS,
            "LEVER_ARM_M": LEVER_ARM_M,
            "MAX_NFEV": MAX_NFEV,
        },
        "paths_used": {k: str(v) for k, v in PATHS.items()},
        "input_summary": meta,
        "num_usable_samples": int(len(usable)),
        "existing_transform_summary_against_robot_fk": existing_summary,
        "estimated_transform_summary_against_robot_fk": estimated_summary,
        "comparison_existing_vs_estimated": comparison,
        "optimization_info": opt_info,
        "T_base_cam4_existing_4x4": T_existing.astype(float).tolist(),
        "T_base_cam4_estimated_4x4": T_refined.astype(float).tolist(),
        "per_pose_diagnostics": combined_rows,
    }

    out_dir = PATHS["results_dir"]
    ensure_dir(out_dir)
    save_json(out_dir / "cam4_extrinsic_validation_report.json", report)
    save_json(out_dir / "tf_base_to_camera_cam4_estimated_block.json", {
        "cam4": {
            "T_4x4": T_refined.astype(float).tolist(),
            "source": "estimated_from_cam4_PnP_robot_FK_fixed_T_EE_CB",
            "note": "Compare first before replacing shared_inputs/tf_base_to_camera.json."
        }
    })
    write_csv(out_dir / "cam4_per_pose_diagnostics.csv", combined_rows)

    print("\n================ SUMMARY ================")
    print(f"Usable samples: {len(usable)} / {len(samples_all)}")
    print("\nExisting cam4 vs robot FK:")
    print(json.dumps(existing_summary, indent=2))
    print("\nEstimated cam4 vs robot FK:")
    print(json.dumps(estimated_summary, indent=2))
    print("\nExisting -> estimated difference:")
    print(json.dumps(comparison, indent=2))
    print(f"\nSaved report to: {out_dir / 'cam4_extrinsic_validation_report.json'}")
    print(f"Saved replacement block to: {out_dir / 'tf_base_to_camera_cam4_estimated_block.json'}")


if __name__ == "__main__":
    main()
