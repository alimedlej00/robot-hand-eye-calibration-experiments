"""
UR3e 1-camera DH calibration (cam1): produces a calibrated DH table for comparison.

Key design decisions:
- T_EE_CB is treated as fixed (no refinement), per thesis methodology.
- T_base_cam is uncertain due to manual measurement; it is searched within bounded tolerances:
    translation ±2 mm, rotation ±3 deg, with discrete resolution 0.1 mm / 0.1 deg.
- Robot calibration optimizes:
    (1) joint zero offsets (6)
    (2) DH deltas for a_i and d_i (12)
  while keeping alpha_i fixed (structural).

Outputs:
- dh_calibrated_1cam.json: updated DH (a,d), original alpha, original theta_offset kept
- calibration_report_1cam.json: metrics, chosen camera correction, joint offsets, etc.
"""

from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


# -------------------------
# Paths (Windows + fallback)
# -------------------------
WINDOWS_PATHS = {
    "pnp": r"C:\Users\kaysa\OneDrive\Pictures\Camera Roll\extrinsic_calibration\one_cam_setup\pnp_cam1_poses.json",
    "joints": r"C:\Users\kaysa\OneDrive\Pictures\Camera Roll\extrinsic_calibration\one_cam_setup\ur3e_joints_data_cam1.json",
    "dh": r"C:\Users\kaysa\OneDrive\Pictures\Camera Roll\extrinsic_calibration\one_cam_setup\ur3e_dh_table_nominal.json",
    "ee_cb": r"C:\Users\kaysa\OneDrive\Pictures\Camera Roll\extrinsic_calibration\one_cam_setup\Transformation_EE_to_CB.json",
    "cam_base": r"C:\Users\kaysa\OneDrive\Pictures\Camera Roll\extrinsic_calibration\one_cam_setup\transformation_camera_base.json",
}

SANDBOX_PATHS = {
    "pnp": "/mnt/data/pnp_cam1_poses.json",
    "joints": "/mnt/data/ur3e_joints_data_cam1.json",
    "dh": "/mnt/data/ur3e_dh_table_nominal.json",
    "ee_cb": "/mnt/data/Transformation_EE_to_CB.json",
    "cam_base": "/mnt/data/transformation_camera_base.json",
}

def resolve_paths() -> Dict[str, str]:
    paths = {}
    for k, wp in WINDOWS_PATHS.items():
        paths[k] = wp if os.path.exists(wp) else SANDBOX_PATHS[k]
    return paths


# -------------------------
# JSON loader
# -------------------------
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------
# SE(3) helpers
# -------------------------
def hat(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = w
    return np.array([[0, -wz, wy],
                     [wz, 0, -wx],
                     [-wy, wx, 0]], dtype=float)

def rodrigues(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3) + hat(w)
    k = w / theta
    K = hat(k)
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)

def rot_log(R: np.ndarray) -> np.ndarray:
    tr = float(np.trace(R))
    c = (tr - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    theta = math.acos(c)
    if theta < 1e-12:
        return np.zeros(3)
    w_hat = (R - R.T) * (0.5 / math.sin(theta))
    return theta * np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]], dtype=float)

def rot_angle(R: np.ndarray) -> float:
    tr = float(np.trace(R))
    c = (tr - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    return math.acos(c)

def se3_from_xi(xi: np.ndarray) -> np.ndarray:
    """
    xi = [wx, wy, wz, tx, ty, tz]  with w axis-angle (rad), t in meters.
    """
    w = xi[:3]
    t = xi[3:]
    T = np.eye(4)
    T[:3, :3] = rodrigues(w)
    T[:3, 3] = t
    return T

def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# -------------------------
# Standard DH FK
# -------------------------
def dh_A(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,     sa,      ca,      d],
        [0.0,    0.0,     0.0,     1.0]
    ], dtype=float)

def fk_from_dh(
    joints_rad: np.ndarray,
    dh_params: List[dict],
    joint_zero_offsets: Optional[np.ndarray] = None
) -> np.ndarray:
    if joint_zero_offsets is None:
        joint_zero_offsets = np.zeros(6)
    T = np.eye(4)
    for q, off, p in zip(joints_rad, joint_zero_offsets, dh_params):
        theta = float(q + off + p.get("theta_offset", 0.0))
        T = T @ dh_A(float(p["a"]), float(p["alpha"]), float(p["d"]), theta)
    return T


# -------------------------
# Data model
# -------------------------
@dataclass
class PoseSample:
    image: str
    joints_rad: np.ndarray
    T_cam_CB: np.ndarray     # board/CB -> camera
    reproj_rmse_px: float

def build_samples(joints_json: dict, pnp_json: dict) -> List[PoseSample]:
    j_by = {p["image"]: np.array(p["joints_rad"], dtype=float) for p in joints_json["poses"]}
    p_by = {p["image"]: p for p in pnp_json["poses"]}

    common = sorted(set(j_by.keys()) & set(p_by.keys()))
    samples: List[PoseSample] = []
    for img in common:
        pj = p_by[img]
        if not pj.get("success", True):
            continue
        T_cam_CB = np.array(pj["T_board_to_cam_4x4"], dtype=float)
        rmse = float(pj.get("reproj_rmse_px", float("nan")))
        samples.append(PoseSample(img, j_by[img], T_cam_CB, rmse))
    return samples


# -------------------------
# Evaluation + summary
# -------------------------
@dataclass
class ErrorRow:
    image: str
    reproj_rmse_px: float
    t_err_m: float
    ang_err_rad: float

def evaluate(
    samples: List[PoseSample],
    dh_params: List[dict],
    T_base_cam: np.ndarray,
    T_EE_CB: np.ndarray,
    reproj_threshold_px: Optional[float] = None,
    joint_zero_offsets: Optional[np.ndarray] = None,
) -> List[ErrorRow]:
    rows: List[ErrorRow] = []
    for s in samples:
        if reproj_threshold_px is not None and s.reproj_rmse_px > reproj_threshold_px:
            continue

        T_base_EE = fk_from_dh(s.joints_rad, dh_params, joint_zero_offsets=joint_zero_offsets)
        T_base_CB_pred = T_base_EE @ T_EE_CB
        T_base_CB_meas = T_base_cam @ s.T_cam_CB

        T_err = inv_T(T_base_CB_pred) @ T_base_CB_meas
        t_err = float(np.linalg.norm(T_err[:3, 3]))
        ang_err = float(rot_angle(T_err[:3, :3]))
        rows.append(ErrorRow(s.image, s.reproj_rmse_px, t_err, ang_err))
    return rows

def summarize(rows: List[ErrorRow]) -> dict:
    t = np.array([r.t_err_m for r in rows], dtype=float)
    a = np.array([r.ang_err_rad for r in rows], dtype=float)
    rm = np.array([r.reproj_rmse_px for r in rows], dtype=float)

    def stats(x: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "max": float(np.max(x)),
            "rmse": float(np.sqrt(np.mean(x * x))),
        }

    return {
        "N": len(rows),
        "t_err_m": stats(t),
        "t_err_mm": stats(t * 1000.0),
        "ang_err_rad": stats(a),
        "ang_err_deg": stats(a * 180.0 / math.pi),
        "reproj_rmse_px": stats(rm),
    }

def scalar_cost(summary: dict, w_t_mm=1.0, w_r_deg=10.0) -> float:
    """
    Single ranking metric for comparing 1-cam vs 2/3/4-cam (use TEST metrics).
    Keeps the comparison tied to SE(3) pose error, not reprojection.
    """
    t_rmse_mm = summary["t_err_mm"]["rmse"]
    r_rmse_deg = summary["ang_err_deg"]["rmse"]
    return float(w_t_mm * t_rmse_mm + w_r_deg * r_rmse_deg)


# -------------------------
# DH parameter update
# -------------------------
def apply_dh_deltas(dh_nominal: List[dict], delta_a: np.ndarray, delta_d: np.ndarray) -> List[dict]:
    dh_new: List[dict] = []
    for i, p in enumerate(dh_nominal):
        q = dict(p)  # shallow copy
        q["a"] = float(p["a"] + delta_a[i])
        q["d"] = float(p["d"] + delta_d[i])
        # alpha stays fixed
        # theta_offset stays as provided in the nominal json
        dh_new.append(q)
    return dh_new


# -------------------------
# Camera tolerance sampling (discrete)
# -------------------------
def quantize(x: np.ndarray, step: float) -> np.ndarray:
    return np.round(x / step) * step

def sample_camera_xi_grid(
    rng: np.random.Generator,
    n: int,
    rot_bound_deg: float,
    trans_bound_m: float,
    rot_step_deg: float,
    trans_step_m: float
) -> List[np.ndarray]:
    """
    Returns list of xi = [wx,wy,wz,tx,ty,tz] with components quantized.
    Rotation uses axis-angle vector components bounded per-component (not perfect sphere bound, but OK for tolerance study).
    """
    rot_bound = math.radians(rot_bound_deg)
    rot_step = math.radians(rot_step_deg)

    xis: List[np.ndarray] = []
    for _ in range(n):
        w = rng.uniform(-rot_bound, rot_bound, size=(3,))
        t = rng.uniform(-trans_bound_m, trans_bound_m, size=(3,))
        w = quantize(w, rot_step)
        t = quantize(t, trans_step_m)
        xi = np.concatenate([w, t], axis=0)
        xis.append(xi.astype(float))
    # Always include the zero correction as a candidate
    xis.insert(0, np.zeros(6, dtype=float))
    return xis


# -------------------------
# Main calibration routine
# -------------------------
def calibrate_dh_for_fixed_camera(
    train: List[PoseSample],
    test: List[PoseSample],
    dh_nominal: List[dict],
    T_base_cam_init: np.ndarray,
    xi_bc_fixed: np.ndarray,
    T_EE_CB_fixed: np.ndarray,
    lever_arm_m: float = 0.15,
    joff_bound_deg: float = 1.0,
    dh_bound_mm: float = 2.0,
    reproj_threshold_px: float = 2.0,
):
    """
    For one fixed camera correction xi_bc_fixed:
      minimize SE(3) residuals by optimizing [joint_offsets(6), delta_a(6), delta_d(6)].

    lever_arm_m converts rotation log residual (rad) into meters: res = [t_err, L*w_err]
    """
    try:
        from scipy.optimize import least_squares
    except Exception as e:
        raise RuntimeError("SciPy is required for calibration. Please install scipy. Error: %s" % e)

    # Filter train/test by PnP reprojection RMSE
    train_u = [s for s in train if s.reproj_rmse_px <= reproj_threshold_px]
    test_u  = [s for s in test  if s.reproj_rmse_px <= reproj_threshold_px]
    if len(train_u) < 10:
        raise RuntimeError("Not enough usable TRAIN samples after reprojection filtering.")

    T_base_cam = se3_from_xi(xi_bc_fixed) @ T_base_cam_init

    def pack(j_off: np.ndarray, delta_a: np.ndarray, delta_d: np.ndarray) -> np.ndarray:
        return np.concatenate([j_off, delta_a, delta_d], axis=0)

    def unpack(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        j_off = x[0:6]
        delta_a = x[6:12]
        delta_d = x[12:18]
        return j_off, delta_a, delta_d

    def residuals(x: np.ndarray, data: List[PoseSample]) -> np.ndarray:
        j_off, delta_a, delta_d = unpack(x)
        dh_cur = apply_dh_deltas(dh_nominal, delta_a, delta_d)

        res_list = []
        for s in data:
            T_base_EE = fk_from_dh(s.joints_rad, dh_cur, joint_zero_offsets=j_off)
            T_base_CB_pred = T_base_EE @ T_EE_CB_fixed
            T_base_CB_meas = T_base_cam @ s.T_cam_CB

            T_err = inv_T(T_base_CB_pred) @ T_base_CB_meas
            t_err = T_err[:3, 3]
            w_err = rot_log(T_err[:3, :3])

            res = np.concatenate([t_err, lever_arm_m * w_err], axis=0)
            res_list.append(res)
        return np.concatenate(res_list, axis=0)

    # Initial guess: no corrections
    x0 = pack(np.zeros(6), np.zeros(6), np.zeros(6))

    # Bounds
    joff_b = math.radians(joff_bound_deg)
    dh_b = dh_bound_mm / 1000.0

    lb = np.concatenate([
        -np.ones(6) * joff_b,
        -np.ones(6) * dh_b,   # delta_a
        -np.ones(6) * dh_b,   # delta_d
    ])
    ub = np.concatenate([
        +np.ones(6) * joff_b,
        +np.ones(6) * dh_b,
        +np.ones(6) * dh_b,
    ])

    sol = least_squares(lambda x: residuals(x, train_u), x0, bounds=(lb, ub), verbose=0, max_nfev=300)

    j_off, delta_a, delta_d = unpack(sol.x)
    dh_cal = apply_dh_deltas(dh_nominal, delta_a, delta_d)

    # Evaluate (use ALL test_u)
    rows_train = evaluate(train_u, dh_cal, T_base_cam, T_EE_CB_fixed, reproj_threshold_px=None, joint_zero_offsets=j_off)
    rows_test  = evaluate(test_u,  dh_cal, T_base_cam, T_EE_CB_fixed, reproj_threshold_px=None, joint_zero_offsets=j_off)

    train_summary = summarize(rows_train)
    test_summary  = summarize(rows_test)

    return {
        "success": bool(sol.success),
        "status": int(sol.status),
        "message": str(sol.message),
        "cost": float(sol.cost),
        "nfev": int(sol.nfev),
        "xi_bc_fixed": xi_bc_fixed,
        "T_base_cam_used": T_base_cam,
        "joint_zero_offsets_rad": j_off,
        "delta_a_m": delta_a,
        "delta_d_m": delta_d,
        "dh_calibrated": dh_cal,
        "train_summary": train_summary,
        "test_summary": test_summary,
        "train_N": int(train_summary["N"]),
        "test_N": int(test_summary["N"]),
        "E_test": scalar_cost(test_summary),
    }


def main():
    # -------------------------
    # User-tunable settings
    # -------------------------
    REPROJ_THRESHOLD_PX = 2.0

    TRAIN_RATIO = 0.7
    SEED = 7

    # Camera tolerance study (discrete candidates)
    CAM_ROT_BOUND_DEG = 3.0
    CAM_TRANS_BOUND_M = 0.002  # 2 mm
    CAM_ROT_STEP_DEG = 0.1
    CAM_TRANS_STEP_M = 0.0001  # 0.1 mm

    N_CAMERA_CANDIDATES = 120  # increase if you want more thorough search

    # Calibration bounds (keep small since UR3e is “near-nominal”)
    JOINT_OFF_BOUND_DEG = 1.0
    DH_BOUND_MM = 2.0

    # Rotation-vs-translation balancing inside residual
    LEVER_ARM_M = 0.15

    # -------------------------
    # Load data
    # -------------------------
    paths = resolve_paths()
    print("Using paths:", paths)

    pnp_json = load_json(paths["pnp"])
    joints_json = load_json(paths["joints"])
    dh_json = load_json(paths["dh"])
    ee_cb_json = load_json(paths["ee_cb"])
    cam_base_json = load_json(paths["cam_base"])

    samples = build_samples(joints_json, pnp_json)
    print(f"Loaded samples (matched by image): {len(samples)}")

    dh_nominal = dh_json["joints"]

    T_base_cam_init = np.array(cam_base_json["transform"]["T_base_from_camera_4x4"], dtype=float)
    T_EE_CB_fixed = np.array(ee_cb_json["transformation_matrix_T_EE_CB_4x4"], dtype=float)

    # -------------------------
    # Train/Test split (on usable poses)
    # -------------------------
    usable = [s for s in samples if (not math.isnan(s.reproj_rmse_px)) and (s.reproj_rmse_px <= REPROJ_THRESHOLD_PX)]
    if len(usable) < 12:
        raise RuntimeError("Not enough usable samples after reprojection filtering. Collect more images or relax threshold.")

    rng = np.random.default_rng(SEED)
    idx = np.arange(len(usable))
    rng.shuffle(idx)
    n_train = int(round(TRAIN_RATIO * len(usable)))
    train = [usable[i] for i in idx[:n_train]]
    test  = [usable[i] for i in idx[n_train:]]

    print(f"Usable poses (reproj <= {REPROJ_THRESHOLD_PX} px): {len(usable)}")
    print(f"Train: {len(train)} | Test: {len(test)}")

    # -------------------------
    # Baseline evaluation (nominal DH, nominal camera)
    # -------------------------
    rows_nom_test = evaluate(test, dh_nominal, T_base_cam_init, T_EE_CB_fixed, reproj_threshold_px=None, joint_zero_offsets=None)
    summ_nom_test = summarize(rows_nom_test)
    print("\n=== NOMINAL MODEL (TEST) ===")
    print(json.dumps(summ_nom_test, indent=2))
    print("E_nominal_test =", scalar_cost(summ_nom_test))

    # -------------------------
    # Camera tolerance candidates (discrete grid-like sampling)
    # -------------------------
    xi_candidates = sample_camera_xi_grid(
        rng=rng,
        n=N_CAMERA_CANDIDATES,
        rot_bound_deg=CAM_ROT_BOUND_DEG,
        trans_bound_m=CAM_TRANS_BOUND_M,
        rot_step_deg=CAM_ROT_STEP_DEG,
        trans_step_m=CAM_TRANS_STEP_M,
    )
    print(f"\nCamera correction candidates: {len(xi_candidates)} (includes zero)")

    # -------------------------
    # Run DH calibration for each candidate camera correction
    # -------------------------
    best = None
    for ci, xi_bc in enumerate(xi_candidates):
        try:
            result = calibrate_dh_for_fixed_camera(
                train=train,
                test=test,
                dh_nominal=dh_nominal,
                T_base_cam_init=T_base_cam_init,
                xi_bc_fixed=xi_bc,
                T_EE_CB_fixed=T_EE_CB_fixed,
                lever_arm_m=LEVER_ARM_M,
                joff_bound_deg=JOINT_OFF_BOUND_DEG,
                dh_bound_mm=DH_BOUND_MM,
                reproj_threshold_px=REPROJ_THRESHOLD_PX,
            )
        except Exception as e:
            print(f"[{ci+1}/{len(xi_candidates)}] candidate failed: {e}")
            continue

        if best is None or result["E_test"] < best["E_test"]:
            best = result

        if (ci + 1) % 20 == 0:
            print(f"[{ci+1}/{len(xi_candidates)}] best E_test so far: {best['E_test']:.4f}")

    if best is None:
        raise RuntimeError("Calibration failed for all camera candidates.")

    # -------------------------
    # Report best result
    # -------------------------
    print("\n=== BEST CALIBRATION RESULT (TEST) ===")
    print("E_best_test =", best["E_test"])
    print(json.dumps(best["test_summary"], indent=2))

    print("\nChosen camera correction xi_bc (deg, mm):")
    w_deg = best["xi_bc_fixed"][:3] * 180.0 / math.pi
    t_mm  = best["xi_bc_fixed"][3:] * 1000.0
    print("  rot (deg):", w_deg)
    print("  trans (mm):", t_mm)

    print("\nJoint zero offsets (deg):")
    print(best["joint_zero_offsets_rad"] * 180.0 / math.pi)

    # -------------------------
    # Save calibrated DH JSON
    # -------------------------
    dh_out = {
        "meta": {
            "robot": "UR3e",
            "dh_convention": dh_json.get("dh_convention", "standard"),
            "units": dh_json.get("units", {"length": "m", "angle": "rad"}),
            "source": "calibrated_from_vision_1cam",
        },
        "joints": best["dh_calibrated"],
        "joint_zero_offsets_rad": best["joint_zero_offsets_rad"].tolist(),
        "camera_correction_xi_bc": best["xi_bc_fixed"].tolist(),
        "T_base_cam_used_4x4": best["T_base_cam_used"].tolist(),
    }

    with open("dh_calibrated_1cam.json", "w", encoding="utf-8") as f:
        json.dump(dh_out, f, indent=2)

    report = {
        "settings": {
            "REPROJ_THRESHOLD_PX": REPROJ_THRESHOLD_PX,
            "TRAIN_RATIO": TRAIN_RATIO,
            "SEED": SEED,
            "CAM_ROT_BOUND_DEG": CAM_ROT_BOUND_DEG,
            "CAM_TRANS_BOUND_M": CAM_TRANS_BOUND_M,
            "CAM_ROT_STEP_DEG": CAM_ROT_STEP_DEG,
            "CAM_TRANS_STEP_M": CAM_TRANS_STEP_M,
            "N_CAMERA_CANDIDATES": N_CAMERA_CANDIDATES,
            "JOINT_OFF_BOUND_DEG": JOINT_OFF_BOUND_DEG,
            "DH_BOUND_MM": DH_BOUND_MM,
            "LEVER_ARM_M": LEVER_ARM_M,
        },
        "nominal_test_summary": summ_nom_test,
        "best_train_summary": best["train_summary"],
        "best_test_summary": best["test_summary"],
        "E_nominal_test": scalar_cost(summ_nom_test),
        "E_best_test": best["E_test"],
        "chosen_xi_bc_fixed": best["xi_bc_fixed"].tolist(),
        "joint_zero_offsets_rad": best["joint_zero_offsets_rad"].tolist(),
        "delta_a_m": best["delta_a_m"].tolist(),
        "delta_d_m": best["delta_d_m"].tolist(),
    }

    with open("calibration_report_1cam.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nWrote dh_calibrated_1cam.json")
    print("Wrote calibration_report_1cam.json")


if __name__ == "__main__":
    main()