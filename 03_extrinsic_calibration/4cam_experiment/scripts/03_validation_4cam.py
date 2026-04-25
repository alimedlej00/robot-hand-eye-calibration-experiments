"""
Robust UR3a 4-camera validation.

Purpose:
- Use the already-improved cam3 transform from shared_inputs/tf_base_to_camera.json.
- Triangulate checkerboard pose from cam1+cam2+cam3+cam4.
- Automatically reject bad/high-error poses before DH calibration.
- Keep thresholds and extrinsic refinement options easy to edit.

Run:
    python 03_validation_4cam.py

Main outputs:
    4cam_experiment/results/04_robust_calibration_4cam/
        robust_calibration_report_4cam.json
        robust_validation_report_4cam.json
        dh_calibrated_4cam_robust.json
        rejected_poses_4cam.csv
        accepted_poses_4cam.csv

Notes:
- This script assumes the 4cam triangulation results already exist from:
    4cam_experiment/results/01_triangulation_4cam/triangulated_checkerboard_poses_4cam.json
- It does NOT overwrite shared_inputs/tf_base_to_camera.json.
- It can optionally optimize small camera extrinsic corrections during calibration.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# USER-EDITABLE CONFIG
# =============================================================================

ROOT = Path(r"C:\Users\kaysa\OneDrive\Desktop\robot-hand-eye-calibration-experiments\03_extrinsic_calibration")

PATHS = {
    "triangulated": ROOT / "4cam_experiment" / "results" / "01_triangulation_4cam" / "triangulated_checkerboard_poses_4cam.json",
    "joints": ROOT / "4cam_experiment" / "results" / "01_triangulation_4cam" / "ur3a_joints_data_4cam.json",
    "dh_nominal": ROOT / "shared_inputs" / "robot_model_dh_nominal_ur3a.json",
    "ee_cb": ROOT / "shared_inputs" / "tf_ee_to_cb.json",
    "cam_base": ROOT / "shared_inputs" / "tf_base_to_camera.json",
    "results_dir": ROOT / "4cam_experiment" / "results" / "03_validation_4cam",
    "dh_calibrated": ROOT / "4cam_experiment" / "results" / "02_calibration_4cam" / "dh_calibrated_4cam.json",
    "calibration_report": ROOT / "4cam_experiment" / "results" / "02_calibration_4cam" / "calibration_report_4cam.json",
}

# Data split
TRAIN_RATIO = 0.70
SEED = 7

# Initial measurement-quality filtering from triangulation output.
# After fixing cam3, reprojection should usually be around 3-6 px and rigid fit around 1-2 mm.
MAX_REPROJ_MEAN_PX = 10.0
MAX_REPROJ_MAX_PX = 15.0
MAX_RIGID_FIT_MM = 3.5

# Optional explicit pose rejection. Pose 26 is a known rotation outlier from the previous run.
EXCLUDE_POSE_IDS = {26}

# Robust post-filtering after nominal comparison against FK(q,DH) @ T_EE_CB.
# This removes samples that probably have wrong joint values, wrist ambiguity, or inconsistent robot/FK data.
MAX_NOMINAL_TRANS_ERR_MM = 45.0
MAX_NOMINAL_ROT_ERR_DEG = 8.0

# Calibration parameter bounds.
JOINT_OFF_BOUND_DEG = 1.0
DH_BOUND_MM = 2.0
LEVER_ARM_M = 0.15
MAX_NFEV = 800

# Robust optimization loss.
ROBUST_LOSS = "soft_l1"     # options in scipy: linear, soft_l1, huber, cauchy, arctan
ROBUST_F_SCALE = 0.01

# Optional camera extrinsic correction during calibration.
# This is useful if cam1/cam2/cam3/cam4 still have small residual errors.
# Corrections are applied as: T_base_cam_corrected = exp(xi) @ T_base_cam_initial.
ENABLE_EXTRINSIC_REFINEMENT = True
CAMERA_IDS_TO_REFINE = ["cam1", "cam2", "cam3", "cam4"]
CAM_ROT_BOUND_DEG = 2.0
CAM_TRANS_BOUND_MM = 10.0

# If True, the calibrated output includes corrected camera transforms.
SAVE_REFINED_CAMERA_TRANSFORMS = True


# =============================================================================
# Math helpers
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


def rot_log(R: np.ndarray) -> np.ndarray:
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    theta = math.acos(c)
    if theta < 1e-12:
        return np.zeros(3, dtype=float)
    w_hat = (R - R.T) * (0.5 / math.sin(theta))
    return theta * np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]], dtype=float)


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


def metric_stats(x: np.ndarray) -> Dict[str, float]:
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


@dataclass
class PoseSample:
    pose_id: int
    image_triplet: List[str]
    joints_rad: np.ndarray
    T_base_CB_meas: np.ndarray
    reproj_mean_rmse_px: float
    reproj_max_rmse_px: float
    rigid_fit_rms_mm: float
    per_cam_rmse_px: Dict[str, float]


def build_samples(triang_json: dict, joints_json: dict) -> List[PoseSample]:
    joints_by_id = {int(p["pose_id"]): p for p in joints_json["poses"]}
    out: List[PoseSample] = []
    for p in triang_json["poses"]:
        pid = int(p["pose_id"])
        if not p.get("success", False) or pid not in joints_by_id:
            continue
        jj = joints_by_id[pid]
        per_cam = {
            "cam1": float(p.get("reproj_cam1_rmse_px", float("nan"))),
            "cam2": float(p.get("reproj_cam2_rmse_px", float("nan"))),
            "cam3": float(p.get("reproj_cam3_rmse_px", float("nan"))),
            "cam4": float(p.get("reproj_cam4_rmse_px", float("nan"))),
        }
        out.append(PoseSample(
            pose_id=pid,
            image_triplet=list(jj.get("image_triplet", [p["images"]["cam1"], p["images"]["cam2"], p["images"]["cam3"], p["images"]["cam4"]])),
            joints_rad=np.array(jj["joints_rad"], dtype=float),
            T_base_CB_meas=np.array(p["T_base_CB_4x4"], dtype=float),
            reproj_mean_rmse_px=float(p.get("reproj_mean_rmse_px", float("nan"))),
            reproj_max_rmse_px=float(p.get("reproj_max_rmse_px", float("nan"))),
            rigid_fit_rms_mm=float(p.get("rigid_fit_rms_mm", float("nan"))),
            per_cam_rmse_px=per_cam,
        ))
    return sorted(out, key=lambda s: s.pose_id)


def pose_error(sample: PoseSample, dh_params: List[dict], T_EE_CB: np.ndarray,
               joint_zero_offsets: Optional[np.ndarray] = None) -> Dict[str, float]:
    T_pred = fk_from_dh(sample.joints_rad, dh_params, joint_zero_offsets) @ T_EE_CB
    T_err = inv_T(T_pred) @ sample.T_base_CB_meas
    e_t_m = float(np.linalg.norm(T_err[:3, 3]))
    e_r_rad = float(rot_angle(T_err[:3, :3]))
    return {
        "e_trans_m": e_t_m,
        "e_trans_mm": e_t_m * 1000.0,
        "e_rot_rad": e_r_rad,
        "e_rot_deg": math.degrees(e_r_rad),
    }


def evaluate_set(samples: List[PoseSample], dh_params: List[dict], T_EE_CB: np.ndarray,
                 joint_zero_offsets: Optional[np.ndarray] = None) -> Dict:
    rows = []
    for s in samples:
        err = pose_error(s, dh_params, T_EE_CB, joint_zero_offsets)
        rows.append({
            "pose_id": s.pose_id,
            "image_triplet": s.image_triplet,
            "reproj_mean_rmse_px": s.reproj_mean_rmse_px,
            "reproj_max_rmse_px": s.reproj_max_rmse_px,
            "rigid_fit_rms_mm": s.rigid_fit_rms_mm,
            "per_cam_rmse_px": s.per_cam_rmse_px,
            **err,
        })
    return {
        "N": len(rows),
        "e_trans_mm": metric_stats(np.array([r["e_trans_mm"] for r in rows], dtype=float)),
        "e_rot_deg": metric_stats(np.array([r["e_rot_deg"] for r in rows], dtype=float)),
        "reproj_mean_rmse_px": metric_stats(np.array([r["reproj_mean_rmse_px"] for r in rows], dtype=float)),
        "reproj_max_rmse_px": metric_stats(np.array([r["reproj_max_rmse_px"] for r in rows], dtype=float)),
        "rigid_fit_rms_mm": metric_stats(np.array([r["rigid_fit_rms_mm"] for r in rows], dtype=float)),
        "per_pose": rows,
    }


def split_samples(samples: List[PoseSample]) -> Tuple[List[PoseSample], List[PoseSample]]:
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_train = int(round(TRAIN_RATIO * len(samples)))
    train = [samples[i] for i in idx[:n_train]]
    test = [samples[i] for i in idx[n_train:]]
    return train, test


def filter_samples(samples: List[PoseSample], dh_nominal: List[dict], T_EE_CB: np.ndarray) -> Tuple[List[PoseSample], List[Dict]]:
    accepted: List[PoseSample] = []
    rejected: List[Dict] = []

    for s in samples:
        reasons = []
        if s.pose_id in EXCLUDE_POSE_IDS:
            reasons.append("explicit_pose_exclusion")
        if not math.isfinite(s.reproj_mean_rmse_px) or s.reproj_mean_rmse_px > MAX_REPROJ_MEAN_PX:
            reasons.append("high_reproj_mean")
        if not math.isfinite(s.reproj_max_rmse_px) or s.reproj_max_rmse_px > MAX_REPROJ_MAX_PX:
            reasons.append("high_reproj_max")
        if not math.isfinite(s.rigid_fit_rms_mm) or s.rigid_fit_rms_mm > MAX_RIGID_FIT_MM:
            reasons.append("high_rigid_fit")

        nom_err = pose_error(s, dh_nominal, T_EE_CB, None)
        if nom_err["e_trans_mm"] > MAX_NOMINAL_TRANS_ERR_MM:
            reasons.append("high_nominal_translation_error")
        if nom_err["e_rot_deg"] > MAX_NOMINAL_ROT_ERR_DEG:
            reasons.append("high_nominal_rotation_error")

        row = {
            "pose_id": s.pose_id,
            "reproj_mean_rmse_px": s.reproj_mean_rmse_px,
            "reproj_max_rmse_px": s.reproj_max_rmse_px,
            "rigid_fit_rms_mm": s.rigid_fit_rms_mm,
            "nominal_e_trans_mm": nom_err["e_trans_mm"],
            "nominal_e_rot_deg": nom_err["e_rot_deg"],
            "reasons": ";".join(reasons),
        }
        if reasons:
            rejected.append(row)
        else:
            accepted.append(s)

    return accepted, rejected


def save_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def calibrate(train: List[PoseSample], test: List[PoseSample], dh_nominal: List[dict], T_EE_CB: np.ndarray) -> Dict:
    try:
        from scipy.optimize import least_squares
    except Exception as e:
        raise RuntimeError("SciPy is required. Install with: python -m pip install scipy") from e

    if len(train) < 8:
        raise RuntimeError(f"Not enough accepted training samples: {len(train)}. Relax thresholds or collect more valid poses.")

    def pack(j_off: np.ndarray, da: np.ndarray, dd: np.ndarray, cam_xis: np.ndarray) -> np.ndarray:
        return np.concatenate([j_off, da, dd, cam_xis], axis=0)

    def unpack(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        j_off = x[0:6]
        da = x[6:12]
        dd = x[12:18]
        cam_xis = x[18:]
        return j_off, da, dd, cam_xis

    n_cam = len(CAMERA_IDS_TO_REFINE) if ENABLE_EXTRINSIC_REFINEMENT else 0
    x0 = pack(np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6 * n_cam))

    j_b = math.radians(JOINT_OFF_BOUND_DEG)
    dh_b = DH_BOUND_MM / 1000.0
    cam_r_b = math.radians(CAM_ROT_BOUND_DEG)
    cam_t_b = CAM_TRANS_BOUND_MM / 1000.0

    lb = [*(-np.ones(6) * j_b), *(-np.ones(6) * dh_b), *(-np.ones(6) * dh_b)]
    ub = [*( np.ones(6) * j_b), *( np.ones(6) * dh_b), *( np.ones(6) * dh_b)]
    for _ in range(n_cam):
        lb += [-cam_r_b, -cam_r_b, -cam_r_b, -cam_t_b, -cam_t_b, -cam_t_b]
        ub += [ cam_r_b,  cam_r_b,  cam_r_b,  cam_t_b,  cam_t_b,  cam_t_b]
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)

    # For this robust calibration step, measurement T_base_CB is already triangulated.
    # Camera xi variables are kept as a flexible placeholder and regularized, but do not directly
    # modify T_base_CB unless triangulation is rerun. We include a penalty so the report can show if
    # optimizer wants extrinsic movement; true extrinsic refinement should happen in triangulation stage.
    def residuals(x: np.ndarray, data: List[PoseSample]) -> np.ndarray:
        j_off, da, dd, cam_xis = unpack(x)
        dh_cur = apply_dh_deltas(dh_nominal, da, dd)
        out = []
        for s in data:
            T_pred = fk_from_dh(s.joints_rad, dh_cur, j_off) @ T_EE_CB
            T_err = inv_T(T_pred) @ s.T_base_CB_meas
            out.append(np.concatenate([T_err[:3, 3], LEVER_ARM_M * rot_log(T_err[:3, :3])], axis=0))
        # Small regularization: avoid fake overfitting through unused extrinsic variables.
        if len(cam_xis):
            out.append(0.001 * cam_xis)
        return np.concatenate(out, axis=0)

    sol = least_squares(
        lambda x: residuals(x, train),
        x0,
        bounds=(lb, ub),
        loss=ROBUST_LOSS,
        f_scale=ROBUST_F_SCALE,
        max_nfev=MAX_NFEV,
        verbose=0,
    )

    j_off, da, dd, cam_xis = unpack(sol.x)
    dh_cal = apply_dh_deltas(dh_nominal, da, dd)
    train_summary = evaluate_set(train, dh_cal, T_EE_CB, j_off)
    test_summary = evaluate_set(test, dh_cal, T_EE_CB, j_off)

    cam_refinement = {}
    for i, cam_id in enumerate(CAMERA_IDS_TO_REFINE if ENABLE_EXTRINSIC_REFINEMENT else []):
        xi = cam_xis[6*i:6*(i+1)]
        cam_refinement[cam_id] = {
            "xi_rad_m": xi.tolist(),
            "rot_deg": (xi[:3] * 180.0 / math.pi).tolist(),
            "trans_mm": (xi[3:] * 1000.0).tolist(),
        }

    return {
        "success": bool(sol.success),
        "status": int(sol.status),
        "message": str(sol.message),
        "cost": float(sol.cost),
        "nfev": int(sol.nfev),
        "joint_zero_offsets_rad": j_off.tolist(),
        "joint_zero_offsets_deg": (j_off * 180.0 / math.pi).tolist(),
        "delta_a_m": da.tolist(),
        "delta_d_m": dd.tolist(),
        "delta_a_mm": (da * 1000.0).tolist(),
        "delta_d_mm": (dd * 1000.0).tolist(),
        "dh_calibrated": dh_cal,
        "train_summary": train_summary,
        "test_summary": test_summary,
        "camera_refinement_request": cam_refinement,
        "parameter_saturation_warning": {
            "joint_offsets_close_to_bound": bool(np.any(np.abs(j_off) > 0.95 * j_b)),
            "dh_a_close_to_bound": bool(np.any(np.abs(da) > 0.95 * dh_b)),
            "dh_d_close_to_bound": bool(np.any(np.abs(dd) > 0.95 * dh_b)),
        },
    }


def flatten_eval_rows(summary: Dict) -> List[Dict]:
    rows = []
    for r in summary.get("per_pose", []):
        row = dict(r)
        pc = row.pop("per_cam_rmse_px", {})
        row["cam1_rmse_px"] = pc.get("cam1")
        row["cam2_rmse_px"] = pc.get("cam2")
        row["cam3_rmse_px"] = pc.get("cam3")
        row["image_triplet"] = ";".join(row.get("image_triplet", []))
        rows.append(row)
    return rows


def main() -> None:
    ensure_dir(PATHS["results_dir"])
    triang_json = load_json(PATHS["triangulated"])
    joints_json = load_json(PATHS["joints"])
    dh_nominal_json = load_json(PATHS["dh_nominal"])
    dh_calibrated_json = load_json(PATHS["dh_calibrated"])
    ee_cb_json = load_json(PATHS["ee_cb"])

    dh_nominal = dh_nominal_json["joints"]
    dh_calibrated = dh_calibrated_json["joints"]
    joint_zero_offsets = np.array(dh_calibrated_json.get("joint_zero_offsets_rad", [0.0] * 6), dtype=float)
    T_EE_CB = np.array(ee_cb_json["transformation_matrix_T_EE_CB_4x4"], dtype=float)

    all_samples = build_samples(triang_json, joints_json)
    accepted, rejected = filter_samples(all_samples, dh_nominal, T_EE_CB)
    train, test = split_samples(accepted)

    nominal_test = evaluate_set(test, dh_nominal, T_EE_CB, None)
    calibrated_test = evaluate_set(test, dh_calibrated, T_EE_CB, joint_zero_offsets)
    train_calibrated = evaluate_set(train, dh_calibrated, T_EE_CB, joint_zero_offsets)

    validation = {
        "validation_name": "UR3a_4cam_DH_validation_final",
        "source_calibrated_dh": str(PATHS["dh_calibrated"]),
        "dataset_summary": {
            "num_total_samples": len(all_samples),
            "num_accepted_samples": len(accepted),
            "num_rejected_samples": len(rejected),
            "accepted_pose_ids": [s.pose_id for s in accepted],
            "rejected_pose_ids": [int(r["pose_id"]) for r in rejected],
            "train_pose_ids": [s.pose_id for s in train],
            "test_pose_ids": [s.pose_id for s in test],
        },
        "nominal_test": nominal_test,
        "calibrated_test": calibrated_test,
        "calibrated_train": train_calibrated,
    }

    save_json(PATHS["results_dir"] / "validation_report_4cam.json", validation)
    save_csv(PATHS["results_dir"] / "validation_nominal_test_per_pose_4cam.csv", flatten_eval_rows(nominal_test))
    save_csv(PATHS["results_dir"] / "validation_calibrated_test_per_pose_4cam.csv", flatten_eval_rows(calibrated_test))

    print("\n================ 4-CAM VALIDATION SUMMARY ================")
    print(f"Accepted samples: {len(accepted)} | rejected: {[int(r['pose_id']) for r in rejected]}")
    print("Nominal TEST translation RMSE mm:   ", nominal_test["e_trans_mm"]["rmse"])
    print("Calibrated TEST translation RMSE mm:", calibrated_test["e_trans_mm"]["rmse"])
    print("Nominal TEST rotation RMSE deg:     ", nominal_test["e_rot_deg"]["rmse"])
    print("Calibrated TEST rotation RMSE deg:  ", calibrated_test["e_rot_deg"]["rmse"])
    print(f"\nSaved validation results to: {PATHS['results_dir']}")


if __name__ == "__main__":
    main()
