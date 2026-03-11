"""
validate_dh_1cam.py

Validates the calibrated UR3e DH table for the 1-camera setup by comparing:
    1) nominal DH prediction vs PnP measurement
    2) calibrated DH prediction vs PnP measurement

Validation is performed on the held-out TEST split reconstructed from the same
settings used in calibrate.py.

Output:
    - validation_report_1cam.json

This script exports JSON only.
"""

from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np


ROOT = r"C:\Users\kaysa\OneDrive\Pictures\Camera Roll\extrinsic_calibration\one_cam_setup"

PATHS = {
    "pnp": os.path.join(ROOT, "pnp_cam1_poses.json"),
    "joints": os.path.join(ROOT, "ur3e_joints_data_cam1.json"),
    "dh_nominal": os.path.join(ROOT, "ur3e_dh_table_nominal.json"),
    "dh_calibrated": os.path.join(ROOT, "dh_calibrated_1cam.json"),
    "ee_cb": os.path.join(ROOT, "Transformation_EE_to_CB.json"),
    "cam_base": os.path.join(ROOT, "transformation_camera_base.json"),
    "calib_report": os.path.join(ROOT, "calibration_report_1cam.json"),
    "output_json": os.path.join(ROOT, "validation_report_1cam.json"),
}


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def rot_angle(R: np.ndarray) -> float:
    tr = float(np.trace(R))
    c = (tr - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
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


def fk_from_dh(
    joints_rad: np.ndarray,
    dh_params: List[dict],
    joint_zero_offsets: Optional[np.ndarray] = None,
) -> np.ndarray:
    if joint_zero_offsets is None:
        joint_zero_offsets = np.zeros(6, dtype=float)

    T = np.eye(4, dtype=float)
    for q, off, p in zip(joints_rad, dh_params if False else joint_zero_offsets, dh_params):
        theta = float(q + off + p.get("theta_offset", 0.0))
        T = T @ dh_A(float(p["a"]), float(p["alpha"]), float(p["d"]), theta)
    return T


@dataclass
class PoseSample:
    image: str
    joints_rad: np.ndarray
    T_cam_CB: np.ndarray
    reproj_rmse_px: float


def build_samples(joints_json: dict, pnp_json: dict) -> List[PoseSample]:
    joints_by_image = {
        p["image"]: np.array(p["joints_rad"], dtype=float)
        for p in joints_json["poses"]
    }
    pnp_by_image = {
        p["image"]: p
        for p in pnp_json["poses"]
    }

    common = sorted(set(joints_by_image.keys()) & set(pnp_by_image.keys()))
    samples: List[PoseSample] = []

    for img in common:
        pj = pnp_by_image[img]
        if not pj.get("success", True):
            continue

        samples.append(
            PoseSample(
                image=img,
                joints_rad=joints_by_image[img],
                T_cam_CB=np.array(pj["T_board_to_cam_4x4"], dtype=float),
                reproj_rmse_px=float(pj.get("reproj_rmse_px", float("nan"))),
            )
        )
    return samples


def reconstruct_train_test_split(
    samples: List[PoseSample],
    reproj_threshold_px: float,
    train_ratio: float,
    seed: int,
) -> Dict[str, List[PoseSample]]:
    usable = [
        s for s in samples
        if (not math.isnan(s.reproj_rmse_px)) and (s.reproj_rmse_px <= reproj_threshold_px)
    ]

    rng = np.random.default_rng(seed)
    idx = np.arange(len(usable))
    rng.shuffle(idx)

    n_train = int(round(train_ratio * len(usable)))
    train = [usable[i] for i in idx[:n_train]]
    test = [usable[i] for i in idx[n_train:]]

    return {"usable": usable, "train": train, "test": test}


def pose_error_against_measurement(
    sample: PoseSample,
    dh_params: List[dict],
    T_base_cam: np.ndarray,
    T_EE_CB: np.ndarray,
    joint_zero_offsets: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    T_base_EE_pred = fk_from_dh(sample.joints_rad, dh_params, joint_zero_offsets=joint_zero_offsets)
    T_base_CB_pred = T_base_EE_pred @ T_EE_CB
    T_base_CB_meas = T_base_cam @ sample.T_cam_CB
    T_err = inv_T(T_base_CB_pred) @ T_base_CB_meas

    e_trans_m = float(np.linalg.norm(T_err[:3, 3]))
    e_rot_rad = float(rot_angle(T_err[:3, :3]))

    return {
        "e_trans_m": e_trans_m,
        "e_trans_mm": e_trans_m * 1000.0,
        "e_rot_rad": e_rot_rad,
        "e_rot_deg": math.degrees(e_rot_rad),
    }


def summarize_metric(x: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "max": float(np.max(x)),
        "rmse": float(np.sqrt(np.mean(x * x))),
    }


def evaluate_set(
    samples: List[PoseSample],
    dh_params: List[dict],
    T_base_cam: np.ndarray,
    T_EE_CB: np.ndarray,
    joint_zero_offsets: Optional[np.ndarray] = None,
) -> Dict:
    per_pose = []
    for s in samples:
        err = pose_error_against_measurement(
            sample=s,
            dh_params=dh_params,
            T_base_cam=T_base_cam,
            T_EE_CB=T_EE_CB,
            joint_zero_offsets=joint_zero_offsets,
        )
        per_pose.append({
            "image": s.image,
            "reproj_rmse_px": float(s.reproj_rmse_px),
            **err,
        })

    e_trans_m = np.array([r["e_trans_m"] for r in per_pose], dtype=float)
    e_trans_mm = np.array([r["e_trans_mm"] for r in per_pose], dtype=float)
    e_rot_rad = np.array([r["e_rot_rad"] for r in per_pose], dtype=float)
    e_rot_deg = np.array([r["e_rot_deg"] for r in per_pose], dtype=float)
    reproj = np.array([r["reproj_rmse_px"] for r in per_pose], dtype=float)

    return {
        "N": int(len(per_pose)),
        "e_trans_m": summarize_metric(e_trans_m),
        "e_trans_mm": summarize_metric(e_trans_mm),
        "e_rot_rad": summarize_metric(e_rot_rad),
        "e_rot_deg": summarize_metric(e_rot_deg),
        "reproj_rmse_px": summarize_metric(reproj),
        "per_pose": per_pose,
    }


def percent_improvement(old_value: float, new_value: float) -> float:
    if abs(old_value) < 1e-12:
        return 0.0
    return float((old_value - new_value) / old_value * 100.0)


def main() -> None:
    pnp_json = load_json(PATHS["pnp"])
    joints_json = load_json(PATHS["joints"])
    dh_nominal_json = load_json(PATHS["dh_nominal"])
    dh_calibrated_json = load_json(PATHS["dh_calibrated"])
    ee_cb_json = load_json(PATHS["ee_cb"])
    cam_base_json = load_json(PATHS["cam_base"])
    calib_report_json = load_json(PATHS["calib_report"])

    dh_nominal = dh_nominal_json["joints"]
    dh_calibrated = dh_calibrated_json["joints"]
    joint_zero_offsets_cal = np.array(
        dh_calibrated_json.get("joint_zero_offsets_rad", [0.0] * 6),
        dtype=float,
    )

    T_EE_CB = np.array(ee_cb_json["transformation_matrix_T_EE_CB_4x4"], dtype=float)
    T_base_cam_nominal = np.array(cam_base_json["transform"]["T_base_from_camera_4x4"], dtype=float)
    T_base_cam_calibrated = np.array(dh_calibrated_json["T_base_cam_used_4x4"], dtype=float)

    settings = calib_report_json["settings"]
    reproj_threshold_px = float(settings["REPROJ_THRESHOLD_PX"])
    train_ratio = float(settings["TRAIN_RATIO"])
    seed = int(settings["SEED"])

    all_samples = build_samples(joints_json, pnp_json)
    split = reconstruct_train_test_split(all_samples, reproj_threshold_px, train_ratio, seed)

    usable_samples = split["usable"]
    train_samples = split["train"]
    test_samples = split["test"]

    nominal_test = evaluate_set(
        samples=test_samples,
        dh_params=dh_nominal,
        T_base_cam=T_base_cam_nominal,
        T_EE_CB=T_EE_CB,
        joint_zero_offsets=None,
    )

    calibrated_test = evaluate_set(
        samples=test_samples,
        dh_params=dh_calibrated,
        T_base_cam=T_base_cam_calibrated,
        T_EE_CB=T_EE_CB,
        joint_zero_offsets=joint_zero_offsets_cal,
    )

    nominal_usable = evaluate_set(
        samples=usable_samples,
        dh_params=dh_nominal,
        T_base_cam=T_base_cam_nominal,
        T_EE_CB=T_EE_CB,
        joint_zero_offsets=None,
    )

    calibrated_usable = evaluate_set(
        samples=usable_samples,
        dh_params=dh_calibrated,
        T_base_cam=T_base_cam_calibrated,
        T_EE_CB=T_EE_CB,
        joint_zero_offsets=joint_zero_offsets_cal,
    )

    improvement_test = {
        "e_trans_rmse_mm_percent": percent_improvement(
            nominal_test["e_trans_mm"]["rmse"], calibrated_test["e_trans_mm"]["rmse"]
        ),
        "e_trans_median_mm_percent": percent_improvement(
            nominal_test["e_trans_mm"]["median"], calibrated_test["e_trans_mm"]["median"]
        ),
        "e_rot_rmse_deg_percent": percent_improvement(
            nominal_test["e_rot_deg"]["rmse"], calibrated_test["e_rot_deg"]["rmse"]
        ),
        "e_rot_median_deg_percent": percent_improvement(
            nominal_test["e_rot_deg"]["median"], calibrated_test["e_rot_deg"]["median"]
        ),
    }

    report = {
        "validation_name": "UR3e_1cam_DH_validation",
        "purpose": (
            "Validation of the calibrated DH table using held-out test poses by comparing "
            "PnP-measured checkerboard pose against checkerboard pose predicted from forward kinematics."
        ),
        "paths_used": PATHS,
        "dataset_info": {
            "num_total_matched_samples": int(len(all_samples)),
            "reproj_threshold_px": reproj_threshold_px,
            "num_usable_samples": int(len(usable_samples)),
            "train_ratio": train_ratio,
            "seed": seed,
            "num_train_samples": int(len(train_samples)),
            "num_test_samples": int(len(test_samples)),
            "train_images": [s.image for s in train_samples],
            "test_images": [s.image for s in test_samples],
        },
        "frame_notes": {
            "pnp_pose": "T_cam_CB = checkerboard pose in camera frame from solvePnP",
            "measurement_pose_in_base": "T_base_CB_meas = T_base_cam @ T_cam_CB",
            "prediction_pose_in_base": "T_base_CB_pred = FK(q, DH) @ T_EE_CB",
            "relative_error": "T_err = inv(T_base_CB_pred) @ T_base_CB_meas",
            "validation_metrics": {
                "e_trans_m": "norm of translation component of T_err in meters",
                "e_rot_rad": "rotation angle of rotation component of T_err in radians",
            },
        },
        "models": {
            "nominal": {
                "camera_transform_used": "mechanically measured T_base_from_camera_4x4",
                "joint_zero_offsets_rad": [0.0] * 6,
            },
            "calibrated": {
                "camera_transform_used": "T_base_cam_used_4x4 from dh_calibrated_1cam.json",
                "joint_zero_offsets_rad": joint_zero_offsets_cal.tolist(),
                "camera_correction_xi_bc": dh_calibrated_json.get("camera_correction_xi_bc", None),
            },
        },
        "results": {
            "held_out_test_set": {
                "nominal": nominal_test,
                "calibrated": calibrated_test,
                "improvement_percent": improvement_test,
            },
            "all_usable_poses_reference": {
                "nominal": nominal_usable,
                "calibrated": calibrated_usable,
            },
        },
    }

    ensure_parent_dir(PATHS["output_json"])
    with open(PATHS["output_json"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Validation JSON written to: {PATHS['output_json']}")
    print("Also saving a copy in sandbox for download.")


if __name__ == "__main__":
    main()
