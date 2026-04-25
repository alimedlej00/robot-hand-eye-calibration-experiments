"""
UR3a 4-camera checkerboard triangulation pipeline.

What this script does:
1) Detect checkerboard corners in cam1, cam2, cam3, cam4 images for each pose.
2) Keep cam1 as the board-origin reference convention.
3) Try logical 0/90/180/270 corner rotations for cam2 and cam3.
4) Triangulate checkerboard corners in the robot base frame using all 4 cameras.
5) Rigid-align the known checkerboard model to triangulated 3D points -> T_base_CB.
6) Save triangulated checkerboard poses and diagnostics.

Install:
    pip install numpy opencv-python scipy

Run:
    python 01_triangulation_4cam.py

Expected repository structure, matching your current thesis repo:
robot-hand-eye-calibration-experiments/
  02_intrinsic_calibration/cam1/04_results/intrinsics_cam1.json
  02_intrinsic_calibration/cam2/04_results/intrinsics_cam2.json
  02_intrinsic_calibration/cam3/04_results/intrinsics_cam3.json
  03_extrinsic_calibration/
    images/cam1_images/img1_cam1.png ... img40_cam1.png
    images/cam2_images/img1_cam2.png ... img40_cam2.png
    images/cam3_images/img1_cam3.png ... img40_cam3.png
    shared_inputs/tf_base_to_camera.json
    shared_inputs/tf_ee_to_cb.json
    shared_inputs/robot_model_dh_nominal_ur3a.json
    shared_inputs/dataset_ur3a_joint_images_40poses.json
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# =============================================================================
# USER CONFIG
# =============================================================================
ROOT = Path(r"C:\Users\kaysa\OneDrive\Desktop\robot-hand-eye-calibration-experiments\03_extrinsic_calibration")
REPO_ROOT = ROOT.parents[0]

CAMERAS = ["cam1", "cam2", "cam3", "cam4"]
CAM_INDEX = {"cam1": 1, "cam2": 2, "cam3": 3, "cam4": 4}
PATTERN_SIZE = (9, 7)  # inner corners: cols, rows

PATHS = {
    "dataset": ROOT / "shared_inputs" / "dataset_ur3a_joint_images_40poses.json",
    "cam_base": ROOT / "shared_inputs" / "tf_base_to_camera.json",
    "ee_cb": ROOT / "shared_inputs" / "tf_ee_to_cb.json",
    "dh_nominal": ROOT / "shared_inputs" / "robot_model_dh_nominal_ur3a.json",
    "triang_results_dir": ROOT / "4cam_experiment" / "results" / "01_triangulation_4cam",
    "calib_results_dir": ROOT / "4cam_experiment" / "results" / "02_calibration_4cam",
    "valid_results_dir": ROOT / "4cam_experiment" / "results" / "03_validation_4cam",
}

INTRINSICS = {
    cam: REPO_ROOT / "02_intrinsic_calibration" / cam / "04_results" / f"intrinsics_{cam}.json"
    for cam in CAMERAS
}
IMAGE_DIRS = {
    cam: ROOT / "images" / f"{cam}_images"
    for cam in CAMERAS
}

# Conservative initial-run settings. Adjust after reading diagnostics.
REPROJ_THRESHOLD_PX = 30.0
TRAIN_RATIO = 0.70
SEED = 7
JOINT_OFF_BOUND_DEG = 1.0
DH_BOUND_MM = 2.0
LEVER_ARM_M = 0.15
MAX_NFEV = 600

# If True, a pose succeeds only when all three cameras detect the board.
REQUIRE_ALL_4_CAMS = True


# =============================================================================
# BASIC I/O AND MATH
# =============================================================================
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
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


def rot_angle(R: np.ndarray) -> float:
    tr = float(np.trace(R))
    c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    return math.acos(c)


def hat(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = w
    return np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]], dtype=float)


def rot_log(R: np.ndarray) -> np.ndarray:
    tr = float(np.trace(R))
    c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    theta = math.acos(c)
    if theta < 1e-12:
        return np.zeros(3)
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


def make_object_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    obj = np.zeros((cols * rows, 3), dtype=np.float64)
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    obj[:, 0] = xs.reshape(-1) * square_size_m
    obj[:, 1] = ys.reshape(-1) * square_size_m
    return obj


def rigid_alignment_board_to_base(obj_pts: np.ndarray, base_pts: np.ndarray) -> Tuple[np.ndarray, float]:
    """Returns T_base_CB and rigid fit RMS in mm."""
    A = obj_pts.astype(np.float64)
    B = base_pts.astype(np.float64)
    ca = A.mean(axis=0)
    cb = B.mean(axis=0)
    AA = A - ca
    BB = B - cb
    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = cb - R @ ca
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    fit = (R @ A.T).T + t
    rms_mm = float(np.sqrt(np.mean(np.sum((fit - B) ** 2, axis=1))) * 1000.0)
    return T, rms_mm


# =============================================================================
# CAMERA / IMAGE FUNCTIONS
# =============================================================================
@dataclass
class CameraInfo:
    camera_id: str
    K_calib: np.ndarray
    dist: np.ndarray
    calib_wh: Tuple[int, int]
    square_size_m: float
    cols: int
    rows: int


def load_intrinsics(path: Path) -> CameraInfo:
    d = load_json(path)
    return CameraInfo(
        camera_id=str(d.get("camera_id", path.stem)),
        K_calib=np.array(d["intrinsics_K"]["matrix_3x3"], dtype=np.float64),
        dist=np.array(d["distortion"]["coefficients"], dtype=np.float64).reshape(-1, 1),
        calib_wh=(int(d["image_size_wh"]["width"]), int(d["image_size_wh"]["height"])),
        square_size_m=float(d.get("square_size_m", 0.02)),
        cols=int(d["pattern_size_inner_corners"]["cols"]),
        rows=int(d["pattern_size_inner_corners"]["rows"]),
    )


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


def image_path_for_pose(cam: str, pose_id: int) -> Path:
    idx = CAM_INDEX[cam]
    # Your current naming is PNG. Fallbacks help if some images are JPG/JPEG.
    for ext in ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]:
        p = IMAGE_DIRS[cam] / f"img{pose_id}_{cam}.{ext}"
        if p.exists():
            return p
    return IMAGE_DIRS[cam] / f"img{pose_id}_{cam}.png"


def detect_corners(img_path: Path, pattern_size: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray, pattern_size,
        flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )
    if (not found) or corners is None or len(corners) != pattern_size[0] * pattern_size[1]:
        raise RuntimeError(f"Checkerboard not detected in {img_path.name}")
    return corners.astype(np.float32), (w, h)


def reorder_corners_assume_image_topleft(corners: np.ndarray, cols: int, rows: int) -> np.ndarray:
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


def generate_rotated_corner_orderings(corners: np.ndarray, cols: int, rows: int) -> List[np.ndarray]:
    grid = corners.reshape(rows, cols, 1, 2)[:, :, 0, :]
    outs = []
    for k in range(4):
        g = np.rot90(grid, k=k)
        g_rc = g if (g.shape[0] == rows and g.shape[1] == cols) else np.transpose(g, (1, 0, 2))
        outs.append(g_rc.reshape(-1, 1, 2).astype(np.float32))
    return outs


def triangulate_multiview_base(
    corners_by_cam: Dict[str, np.ndarray],
    K_by_cam: Dict[str, np.ndarray],
    dist_by_cam: Dict[str, np.ndarray],
    T_base_cam_by_cam: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Linear DLT triangulation using undistorted normalized image coordinates.
    Returns N x 3 points in base frame.
    """
    cams = list(corners_by_cam.keys())
    n_pts = next(iter(corners_by_cam.values())).reshape(-1, 2).shape[0]
    X = np.zeros((n_pts, 3), dtype=np.float64)

    norm_pts_by_cam = {}
    Pn_by_cam = {}
    for cam in cams:
        pts = cv2.undistortPoints(corners_by_cam[cam], K_by_cam[cam], dist_by_cam[cam]).reshape(-1, 2)
        norm_pts_by_cam[cam] = pts
        Pn_by_cam[cam] = inv_T(T_base_cam_by_cam[cam])[:3, :]  # X_cam = Pn @ X_base_h

    for i in range(n_pts):
        A = []
        for cam in cams:
            x, y = norm_pts_by_cam[cam][i]
            P = Pn_by_cam[cam]
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])
        A = np.asarray(A, dtype=np.float64)
        _, _, Vt = np.linalg.svd(A)
        Xh = Vt[-1]
        X[i, :] = Xh[:3] / Xh[3]
    return X


def project_board_points(obj_pts_board: np.ndarray, T_base_CB: np.ndarray, T_base_cam: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    T_cam_CB = inv_T(T_base_cam) @ T_base_CB
    R = T_cam_CB[:3, :3]
    t = T_cam_CB[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(obj_pts_board.astype(np.float64), rvec, t.reshape(3, 1), K, dist)
    return proj.reshape(-1, 2)


def rmse_pixels(p1: np.ndarray, p2: np.ndarray) -> float:
    d = p1.reshape(-1, 2) - p2.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


# =============================================================================
# TRIANGULATION
# =============================================================================
def run_triangulation() -> Tuple[dict, dict]:
    ensure_dir(PATHS["triang_results_dir"])
    dataset = load_json(PATHS["dataset"])
    cam_base_json = load_json(PATHS["cam_base"])
    intr = {cam: load_intrinsics(INTRINSICS[cam]) for cam in CAMERAS}

    T_base_cam = {
        cam: np.array(cam_base_json["T_base_to_camera_extrinsics"][cam]["T_4x4"], dtype=float)
        for cam in CAMERAS
    }

    cols, rows = PATTERN_SIZE
    obj_pts = make_object_points(cols, rows, intr["cam1"].square_size_m)

    poses_out = []
    joints_out = []
    successes = 0
    failures = 0

    print("\n================ TRIANGULATION: 4 CAMERAS ================")
    for pose in dataset["poses"]:
        pid = int(pose["pose_id"])
        entry = {
            "pose_id": pid,
            "images": {cam: image_path_for_pose(cam, pid).name for cam in CAMERAS},
            "success": False,
        }

        try:
            raw = {}
            K_used = {}
            scales = {}
            actual_wh = {}
            for cam in CAMERAS:
                img_path = image_path_for_pose(cam, pid)
                raw_corners, wh = detect_corners(img_path, PATTERN_SIZE)
                K_used[cam], scales[cam] = scale_camera_matrix(intr[cam].K_calib, intr[cam].calib_wh, wh)
                raw[cam] = raw_corners
                actual_wh[cam] = {"width": int(wh[0]), "height": int(wh[1])}

            c1 = reorder_corners_assume_image_topleft(raw["cam1"], cols, rows)
            candidates = {
                "cam2": generate_rotated_corner_orderings(raw["cam2"], cols, rows),
                "cam3": generate_rotated_corner_orderings(raw["cam3"], cols, rows),
                "cam4": generate_rotated_corner_orderings(raw["cam4"], cols, rows),
            }

            best = None
            for idx2, c2 in enumerate(candidates["cam2"]):
                for idx3, c3 in enumerate(candidates["cam3"]):
                    for idx4, c4 in enumerate(candidates["cam4"]):
                        corners = {"cam1": c1, "cam2": c2, "cam3": c3, "cam4": c4}
                        X = triangulate_multiview_base(
                            corners_by_cam=corners,
                            K_by_cam=K_used,
                            dist_by_cam={cam: intr[cam].dist for cam in CAMERAS},
                            T_base_cam_by_cam=T_base_cam,
                        )
                        T_base_CB, rigid_rms_mm = rigid_alignment_board_to_base(obj_pts, X)
                        reproj = {}
                        for cam in CAMERAS:
                            proj = project_board_points(obj_pts, T_base_CB, T_base_cam[cam], K_used[cam], intr[cam].dist)
                            reproj[cam] = rmse_pixels(proj, corners[cam].reshape(-1, 2))
                        mean_rmse = float(np.mean([reproj[c] for c in CAMERAS]))
                        max_rmse = float(np.max([reproj[c] for c in CAMERAS]))
                        score = mean_rmse + 0.02 * rigid_rms_mm  # small geometry penalty, mostly reprojection-based
                        if best is None or score < best["score"]:
                            best = {
                                "score": score,
                                "cam2_rotation_index": idx2,
                                "cam3_rotation_index": idx3,
                                "cam4_rotation_index": idx4,
                                "corners": corners,
                                "X": X,
                                "T_base_CB": T_base_CB,
                                "rigid_fit_rms_mm": rigid_rms_mm,
                                "reproj": reproj,
                                "mean_rmse": mean_rmse,
                                "max_rmse": max_rmse,
                            }

            assert best is not None
            T_base_CB = best["T_base_CB"]
            T_cam_CB = {cam: inv_T(T_base_cam[cam]) @ T_base_CB for cam in CAMERAS}

            entry.update({
                "success": True,
                "reproj_cam1_rmse_px": float(best["reproj"]["cam1"]),
                "reproj_cam2_rmse_px": float(best["reproj"]["cam2"]),
                "reproj_cam3_rmse_px": float(best["reproj"]["cam3"]),
                "reproj_cam4_rmse_px": float(best["reproj"]["cam4"]),
                "reproj_mean_rmse_px": float(best["mean_rmse"]),
                "reproj_max_rmse_px": float(best["max_rmse"]),
                "rigid_fit_rms_mm": float(best["rigid_fit_rms_mm"]),
                "cam2_rotation_index_selected": int(best["cam2_rotation_index"]),
                "cam3_rotation_index_selected": int(best["cam3_rotation_index"]),
                "cam4_rotation_index_selected": int(best["cam4_rotation_index"]),
                "T_base_CB_4x4": T_base_CB.tolist(),
                "T_cam1_CB_4x4": T_cam_CB["cam1"].tolist(),
                "T_cam2_CB_4x4": T_cam_CB["cam2"].tolist(),
                "T_cam3_CB_4x4": T_cam_CB["cam3"].tolist(),
                "T_cam4_CB_4x4": T_cam_CB["cam4"].tolist(),
                "triangulated_points_base_3d": best["X"].tolist(),
                "camera_matrix_used_cam1_3x3": K_used["cam1"].tolist(),
                "camera_matrix_used_cam2_3x3": K_used["cam2"].tolist(),
                "camera_matrix_used_cam3_3x3": K_used["cam3"].tolist(),
                "camera_matrix_used_cam4_3x3": K_used["cam4"].tolist(),
                "intrinsics_scale_cam1": scales["cam1"],
                "intrinsics_scale_cam2": scales["cam2"],
                "intrinsics_scale_cam3": scales["cam3"],
                "intrinsics_scale_cam4": scales["cam4"],
                "actual_image_size_wh": actual_wh,
            })

            joints_out.append({
                "pose_id": pid,
                "image_triplet": [image_path_for_pose(cam, pid).name for cam in CAMERAS],
                "joints_deg": [float(pose["joint_angles_deg"][f"joint_{i}"]) for i in range(1, 7)],
                "joints_rad": [math.radians(float(pose["joint_angles_deg"][f"joint_{i}"])) for i in range(1, 7)],
            })
            successes += 1
            print(f"pose {pid:02d}: OK | reproj mean={best['mean_rmse']:.3f}px max={best['max_rmse']:.3f}px fit={best['rigid_fit_rms_mm']:.2f}mm")

        except Exception as e:
            failures += 1
            entry["error"] = str(e)
            entry["traceback"] = traceback.format_exc(limit=2)
            print(f"pose {pid:02d}: FAIL -> {e}")

        poses_out.append(entry)

    triang_json = {
        "configuration": "4cam",
        "cameras_used": CAMERAS,
        "method": "multi_view_DLT_triangulation_followed_by_checkerboard_rigid_alignment",
        "image_directories": {cam: str(IMAGE_DIRS[cam]) for cam in CAMERAS},
        "pattern": {"cols": cols, "rows": rows, "square_size_m": intr["cam1"].square_size_m},
        "pose_definition": "Main saved measurement is T_base_CB_4x4 estimated from 4-camera triangulation and rigid alignment.",
        "origin_convention": "Cam1 uses the validated board-origin convention. Cam2, cam3 and cam4 orderings are selected among 4 logical rotations by minimum 4-view reprojection error plus a small rigid-fit penalty.",
        "summary": {"num_images_total": len(dataset["poses"]), "num_success": successes, "num_failed": failures},
        "poses": poses_out,
    }

    joints_json = {
        "robot": dataset.get("robot", "UR3a"),
        "units": {"joint_angles": "rad"},
        "source_dataset": str(PATHS["dataset"]),
        "cameras_used": CAMERAS,
        "poses": joints_out,
    }

    write_json(PATHS["triang_results_dir"] / "triangulated_checkerboard_poses_4cam.json", triang_json)
    write_json(PATHS["triang_results_dir"] / "ur3a_joints_data_4cam.json", joints_json)
    write_triangulation_csv(triang_json, PATHS["triang_results_dir"] / "triangulation_diagnostics_4cam.csv")

    print(f"\nTriangulation saved in: {PATHS['triang_results_dir']}")
    print(f"Successes: {successes}/{len(dataset['poses'])}")
    return triang_json, joints_json


def write_triangulation_csv(triang_json: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pose_id", "success", "reproj_cam1_rmse_px", "reproj_cam2_rmse_px", "reproj_cam3_rmse_px", "reproj_cam4_rmse_px",
        "reproj_mean_rmse_px", "reproj_max_rmse_px", "rigid_fit_rms_mm", "error"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in triang_json["poses"]:
            w.writerow({k: p.get(k, "") for k in fields})


# =============================================================================
# CALIBRATION + VALIDATION
# =============================================================================
@dataclass
class PoseSample:
    pose_id: int
    image_triplet: List[str]
    joints_rad: np.ndarray
    T_base_CB: np.ndarray
    reproj_mean_rmse_px: float
    reproj_max_rmse_px: float
    rigid_fit_rms_mm: float
    per_cam_rmse: Dict[str, float]


@dataclass
class ErrorRow:
    pose_id: int
    image_triplet: List[str]
    reproj_mean_rmse_px: float
    reproj_max_rmse_px: float
    rigid_fit_rms_mm: float
    t_err_m: float
    ang_err_rad: float
    per_cam_rmse: Dict[str, float]


def build_samples(joints_json: dict, triang_json: dict) -> List[PoseSample]:
    j_by = {int(p["pose_id"]): p for p in joints_json["poses"]}
    t_by = {int(p["pose_id"]): p for p in triang_json["poses"]}
    out: List[PoseSample] = []
    for pid in sorted(set(j_by) & set(t_by)):
        tj = t_by[pid]
        if not tj.get("success", True):
            continue
        jj = j_by[pid]
        out.append(PoseSample(
            pose_id=pid,
            image_triplet=list(jj["image_triplet"]),
            joints_rad=np.array(jj["joints_rad"], dtype=float),
            T_base_CB=np.array(tj["T_base_CB_4x4"], dtype=float),
            reproj_mean_rmse_px=float(tj.get("reproj_mean_rmse_px", float("nan"))),
            reproj_max_rmse_px=float(tj.get("reproj_max_rmse_px", float("nan"))),
            rigid_fit_rms_mm=float(tj.get("rigid_fit_rms_mm", float("nan"))),
            per_cam_rmse={
                "cam1": float(tj.get("reproj_cam1_rmse_px", float("nan"))),
                "cam2": float(tj.get("reproj_cam2_rmse_px", float("nan"))),
                "cam3": float(tj.get("reproj_cam3_rmse_px", float("nan"))),
            "cam4": float(tj.get("reproj_cam4_rmse_px", float("nan"))),
            },
        ))
    return out


def evaluate(samples: List[PoseSample], dh_params: List[dict], T_EE_CB: np.ndarray, joint_zero_offsets: Optional[np.ndarray] = None) -> List[ErrorRow]:
    rows = []
    for s in samples:
        T_pred = fk_from_dh(s.joints_rad, dh_params, joint_zero_offsets) @ T_EE_CB
        T_err = inv_T(T_pred) @ s.T_base_CB
        rows.append(ErrorRow(
            pose_id=s.pose_id,
            image_triplet=s.image_triplet,
            reproj_mean_rmse_px=s.reproj_mean_rmse_px,
            reproj_max_rmse_px=s.reproj_max_rmse_px,
            rigid_fit_rms_mm=s.rigid_fit_rms_mm,
            t_err_m=float(np.linalg.norm(T_err[:3, 3])),
            ang_err_rad=float(rot_angle(T_err[:3, :3])),
            per_cam_rmse=s.per_cam_rmse,
        ))
    return rows


def summarize_metric(x: np.ndarray) -> Dict[str, float]:
    if len(x) == 0:
        return {"mean": float("nan"), "median": float("nan"), "max": float("nan"), "rmse": float("nan")}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "max": float(np.max(x)),
        "rmse": float(np.sqrt(np.mean(x * x))),
    }


def summarize(rows: List[ErrorRow]) -> dict:
    t = np.array([r.t_err_m for r in rows], dtype=float)
    a = np.array([r.ang_err_rad for r in rows], dtype=float)
    mean_r = np.array([r.reproj_mean_rmse_px for r in rows], dtype=float)
    max_r = np.array([r.reproj_max_rmse_px for r in rows], dtype=float)
    fit = np.array([r.rigid_fit_rms_mm for r in rows], dtype=float)
    return {
        "N": len(rows),
        "t_err_m": summarize_metric(t),
        "t_err_mm": summarize_metric(t * 1000.0),
        "ang_err_rad": summarize_metric(a),
        "ang_err_deg": summarize_metric(a * 180.0 / math.pi),
        "reproj_mean_rmse_px": summarize_metric(mean_r),
        "reproj_max_rmse_px": summarize_metric(max_r),
        "rigid_fit_rms_mm": summarize_metric(fit),
    }


def scalar_cost(summary: dict, w_t_mm: float = 1.0, w_r_deg: float = 10.0) -> float:
    return float(summary["t_err_mm"]["rmse"] + w_r_deg * summary["ang_err_deg"]["rmse"])


def apply_dh_deltas(dh_nominal: List[dict], delta_a: np.ndarray, delta_d: np.ndarray) -> List[dict]:
    out = []
    for i, p in enumerate(dh_nominal):
        q = dict(p)
        q["a"] = float(p["a"] + delta_a[i])
        q["d"] = float(p["d"] + delta_d[i])
        out.append(q)
    return out


def calibrate_dh(train: List[PoseSample], test: List[PoseSample], dh_nominal: List[dict], T_EE_CB: np.ndarray) -> dict:
    try:
        from scipy.optimize import least_squares
    except Exception as e:
        raise RuntimeError(f"SciPy is required. Install it with: pip install scipy. Original error: {e}")

    train_u = [s for s in train if s.reproj_mean_rmse_px <= REPROJ_THRESHOLD_PX]
    test_u = [s for s in test if s.reproj_mean_rmse_px <= REPROJ_THRESHOLD_PX]
    if len(train_u) < 10:
        raise RuntimeError("Not enough usable TRAIN samples after reprojection filtering. Relax threshold or collect more usable images.")

    def pack(j: np.ndarray, a: np.ndarray, d: np.ndarray) -> np.ndarray:
        return np.concatenate([j, a, d], axis=0)

    def unpack(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return x[0:6], x[6:12], x[12:18]

    def residuals(x: np.ndarray, data: List[PoseSample]) -> np.ndarray:
        j_off, delta_a, delta_d = unpack(x)
        dh_cur = apply_dh_deltas(dh_nominal, delta_a, delta_d)
        out = []
        for s in data:
            T_pred = fk_from_dh(s.joints_rad, dh_cur, j_off) @ T_EE_CB
            T_err = inv_T(T_pred) @ s.T_base_CB
            out.append(np.concatenate([T_err[:3, 3], LEVER_ARM_M * rot_log(T_err[:3, :3])], axis=0))
        return np.concatenate(out, axis=0)

    x0 = pack(np.zeros(6), np.zeros(6), np.zeros(6))
    joff_b = math.radians(JOINT_OFF_BOUND_DEG)
    dh_b = DH_BOUND_MM / 1000.0
    lb = np.concatenate([-np.ones(6) * joff_b, -np.ones(6) * dh_b, -np.ones(6) * dh_b])
    ub = np.concatenate([+np.ones(6) * joff_b, +np.ones(6) * dh_b, +np.ones(6) * dh_b])

    sol = least_squares(lambda x: residuals(x, train_u), x0, bounds=(lb, ub), verbose=0, max_nfev=MAX_NFEV)
    j_off, delta_a, delta_d = unpack(sol.x)
    dh_cal = apply_dh_deltas(dh_nominal, delta_a, delta_d)
    train_rows = evaluate(train_u, dh_cal, T_EE_CB, j_off)
    test_rows = evaluate(test_u, dh_cal, T_EE_CB, j_off)
    train_summary = summarize(train_rows)
    test_summary = summarize(test_rows)

    return {
        "success": bool(sol.success),
        "status": int(sol.status),
        "message": str(sol.message),
        "cost": float(sol.cost),
        "nfev": int(sol.nfev),
        "joint_zero_offsets_rad": j_off,
        "delta_a_m": delta_a,
        "delta_d_m": delta_d,
        "dh_calibrated": dh_cal,
        "train_summary": train_summary,
        "test_summary": test_summary,
        "E_test": scalar_cost(test_summary),
        "train_rows": train_rows,
        "test_rows": test_rows,
    }


def split_samples(samples: List[PoseSample]) -> Tuple[List[PoseSample], List[PoseSample], List[PoseSample]]:
    usable = [s for s in samples if (not math.isnan(s.reproj_mean_rmse_px)) and s.reproj_mean_rmse_px <= REPROJ_THRESHOLD_PX]
    if len(usable) < 12:
        raise RuntimeError("Not enough usable samples after 4-camera reprojection filtering.")
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(usable))
    rng.shuffle(idx)
    n_train = int(round(TRAIN_RATIO * len(usable)))
    train = [usable[i] for i in idx[:n_train]]
    test = [usable[i] for i in idx[n_train:]]
    return usable, train, test


def rows_to_list(rows: List[ErrorRow]) -> List[dict]:
    out = []
    for r in rows:
        out.append({
            "pose_id": r.pose_id,
            "image_triplet": r.image_triplet,
            "reproj_mean_rmse_px": r.reproj_mean_rmse_px,
            "reproj_max_rmse_px": r.reproj_max_rmse_px,
            "rigid_fit_rms_mm": r.rigid_fit_rms_mm,
            "e_trans_m": r.t_err_m,
            "e_trans_mm": r.t_err_m * 1000.0,
            "e_rot_rad": r.ang_err_rad,
            "e_rot_deg": math.degrees(r.ang_err_rad),
            "per_cam_rmse_px": r.per_cam_rmse,
        })
    return out


def percent_improvement(old_value: float, new_value: float) -> float:
    if abs(old_value) < 1e-12:
        return 0.0
    return float((old_value - new_value) / old_value * 100.0)


def make_diagnostic_advice(triang_json: dict, nominal_summary: dict, calibrated_summary: dict, calib_result: dict) -> dict:
    ok_poses = [p for p in triang_json["poses"] if p.get("success")]
    cam_means = {}
    for cam in CAMERAS:
        key = f"reproj_{cam}_rmse_px"
        vals = [float(p[key]) for p in ok_poses if key in p]
        cam_means[cam] = float(np.mean(vals)) if vals else float("nan")

    worst_reproj = sorted(
        [{"pose_id": p["pose_id"], "reproj_mean_rmse_px": p.get("reproj_mean_rmse_px"), "reproj_max_rmse_px": p.get("reproj_max_rmse_px"), "rigid_fit_rms_mm": p.get("rigid_fit_rms_mm")} for p in ok_poses],
        key=lambda x: float(x["reproj_mean_rmse_px"]), reverse=True
    )[:8]

    joint_offsets_deg = calib_result["joint_zero_offsets_rad"] * 180.0 / math.pi
    delta_a_mm = calib_result["delta_a_m"] * 1000.0
    delta_d_mm = calib_result["delta_d_m"] * 1000.0

    saturated = {
        "joint_offsets_near_bound": [int(i + 1) for i, v in enumerate(joint_offsets_deg) if abs(v) > 0.90 * JOINT_OFF_BOUND_DEG],
        "delta_a_near_bound": [int(i + 1) for i, v in enumerate(delta_a_mm) if abs(v) > 0.90 * DH_BOUND_MM],
        "delta_d_near_bound": [int(i + 1) for i, v in enumerate(delta_d_mm) if abs(v) > 0.90 * DH_BOUND_MM],
    }

    advice = []
    worst_cam = max(cam_means, key=lambda c: cam_means[c] if not math.isnan(cam_means[c]) else -1)
    if cam_means[worst_cam] > 1.5 * min(cam_means.values()):
        advice.append(f"{worst_cam} has noticeably higher average reprojection RMSE. Check its intrinsic calibration, focus, motion blur, and T_base_to_camera value first.")
    if calibrated_summary["t_err_mm"]["rmse"] > nominal_summary["t_err_mm"]["rmse"]:
        advice.append("Calibration worsened held-out translation RMSE. This usually means overfitting, incorrect corner ordering for some poses, wrong camera extrinsics, or train/test split too small.")
    if saturated["joint_offsets_near_bound"] or saturated["delta_a_near_bound"] or saturated["delta_d_near_bound"]:
        advice.append("Some optimized parameters hit the bounds. If results improve but bounds saturate, run a second controlled experiment with slightly wider bounds and report it clearly.")
    if calibrated_summary["reproj_mean_rmse_px"]["mean"] > 3.0:
        advice.append("Mean triangulation reprojection is above 3 px. Improve corner detection/images/camera transforms before trying more aggressive DH calibration.")
    if not advice:
        advice.append("Initial run looks numerically stable. Next improvement step: inspect worst poses, remove only physically justified outliers, then compare 1cam/2cam/4cam under the same train/test split.")

    return {
        "camera_average_reprojection_rmse_px": cam_means,
        "worst_reprojection_poses": worst_reproj,
        "parameter_saturation_check": saturated,
        "joint_zero_offsets_deg": joint_offsets_deg.tolist(),
        "delta_a_mm": delta_a_mm.tolist(),
        "delta_d_mm": delta_d_mm.tolist(),
        "advice": advice,
    }


def write_validation_csv(rows: List[ErrorRow], path: Path) -> None:
    fields = [
        "pose_id", "e_trans_mm", "e_rot_deg", "reproj_mean_rmse_px", "reproj_max_rmse_px", "rigid_fit_rms_mm",
        "cam1_rmse_px", "cam2_rmse_px", "cam3_rmse_px", "cam4_rmse_px", "image_triplet"
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                "pose_id": r.pose_id,
                "e_trans_mm": r.t_err_m * 1000.0,
                "e_rot_deg": math.degrees(r.ang_err_rad),
                "reproj_mean_rmse_px": r.reproj_mean_rmse_px,
                "reproj_max_rmse_px": r.reproj_max_rmse_px,
                "rigid_fit_rms_mm": r.rigid_fit_rms_mm,
                "cam1_rmse_px": r.per_cam_rmse.get("cam1"),
                "cam2_rmse_px": r.per_cam_rmse.get("cam2"),
                "cam3_rmse_px": r.per_cam_rmse.get("cam3"),
                "cam4_rmse_px": r.per_cam_rmse.get("cam4"),
                "image_triplet": " | ".join(r.image_triplet),
            })


def run_calibration_and_validation(triang_json: dict, joints_json: dict) -> Tuple[dict, dict]:
    print("\n================ DH CALIBRATION + VALIDATION: 4 CAMERAS ================")
    ensure_dir(PATHS["calib_results_dir"])
    ensure_dir(PATHS["valid_results_dir"])

    dh_json = load_json(PATHS["dh_nominal"])
    ee_cb_json = load_json(PATHS["ee_cb"])
    dh_nominal = dh_json["joints"]
    T_EE_CB = np.array(ee_cb_json["transformation_matrix_T_EE_CB_4x4"], dtype=float)

    samples = build_samples(joints_json, triang_json)
    usable, train, test = split_samples(samples)
    print(f"Loaded matched successful 4-cam samples: {len(samples)}")
    print(f"Usable poses (mean reproj <= {REPROJ_THRESHOLD_PX}px): {len(usable)}")
    print(f"Train: {len(train)} | Test: {len(test)}")

    nominal_test_rows = evaluate(test, dh_nominal, T_EE_CB, None)
    nominal_test_summary = summarize(nominal_test_rows)
    print("\n=== NOMINAL MODEL (4-CAM TEST) ===")
    print(json.dumps(nominal_test_summary, indent=2))
    print("E_nominal_test =", scalar_cost(nominal_test_summary))

    best = calibrate_dh(train, test, dh_nominal, T_EE_CB)
    print("\n=== CALIBRATED MODEL (4-CAM TEST) ===")
    print("E_best_test =", best["E_test"])
    print(json.dumps(best["test_summary"], indent=2))
    print("\nJoint zero offsets (deg):")
    print(best["joint_zero_offsets_rad"] * 180.0 / math.pi)
    print("Delta a (mm):")
    print(best["delta_a_m"] * 1000.0)
    print("Delta d (mm):")
    print(best["delta_d_m"] * 1000.0)

    dh_out = {
        "meta": {
            "robot": dh_json.get("robot_name", dh_json.get("robot", "UR3a")),
            "dh_convention": dh_json.get("dh_convention", "standard"),
            "units": dh_json.get("units", {"length": "m", "angle": "rad"}),
            "source": "calibrated_from_vision_4cam_triangulation",
        },
        "joints": best["dh_calibrated"],
        "joint_zero_offsets_rad": best["joint_zero_offsets_rad"].tolist(),
    }

    calibrated_test_rows = best["test_rows"]
    calibrated_test_summary = best["test_summary"]
    usable_nominal_rows = evaluate(usable, dh_nominal, T_EE_CB, None)
    usable_calibrated_rows = evaluate(usable, best["dh_calibrated"], T_EE_CB, best["joint_zero_offsets_rad"])

    improvement_test = {
        "e_trans_rmse_mm_percent": percent_improvement(nominal_test_summary["t_err_mm"]["rmse"], calibrated_test_summary["t_err_mm"]["rmse"]),
        "e_trans_median_mm_percent": percent_improvement(nominal_test_summary["t_err_mm"]["median"], calibrated_test_summary["t_err_mm"]["median"]),
        "e_rot_rmse_deg_percent": percent_improvement(nominal_test_summary["ang_err_deg"]["rmse"], calibrated_test_summary["ang_err_deg"]["rmse"]),
        "e_rot_median_deg_percent": percent_improvement(nominal_test_summary["ang_err_deg"]["median"], calibrated_test_summary["ang_err_deg"]["median"]),
    }

    calibration_report = {
        "settings": {
            "REPROJ_THRESHOLD_PX": REPROJ_THRESHOLD_PX,
            "TRAIN_RATIO": TRAIN_RATIO,
            "SEED": SEED,
            "JOINT_OFF_BOUND_DEG": JOINT_OFF_BOUND_DEG,
            "DH_BOUND_MM": DH_BOUND_MM,
            "LEVER_ARM_M": LEVER_ARM_M,
            "MAX_NFEV": MAX_NFEV,
        },
        "nominal_test_summary": nominal_test_summary,
        "best_train_summary": best["train_summary"],
        "best_test_summary": calibrated_test_summary,
        "E_nominal_test": scalar_cost(nominal_test_summary),
        "E_best_test": best["E_test"],
        "joint_zero_offsets_rad": best["joint_zero_offsets_rad"].tolist(),
        "delta_a_m": best["delta_a_m"].tolist(),
        "delta_d_m": best["delta_d_m"].tolist(),
        "solver": {"success": best["success"], "status": best["status"], "message": best["message"], "nfev": best["nfev"], "cost": best["cost"]},
    }

    validation_report = {
        "validation_name": "UR3a_4cam_DH_validation_triangulation",
        "purpose": "Held-out validation comparing 4-camera triangulation-measured checkerboard pose against FK(q,DH) @ T_EE_CB.",
        "paths_used": {k: str(v) for k, v in PATHS.items()},
        "dataset_info": {
            "num_total_matched_samples": len(samples),
            "reproj_threshold_px": REPROJ_THRESHOLD_PX,
            "num_usable_samples": len(usable),
            "train_ratio": TRAIN_RATIO,
            "seed": SEED,
            "num_train_samples": len(train),
            "num_test_samples": len(test),
            "train_pose_ids": [s.pose_id for s in train],
            "test_pose_ids": [s.pose_id for s in test],
        },
        "frame_notes": {
            "measurement_pose_in_base": "T_base_CB_meas comes directly from 4-camera triangulation followed by rigid alignment.",
            "prediction_pose_in_base": "T_base_CB_pred = FK(q, DH) @ T_EE_CB",
            "relative_error": "T_err = inv(T_base_CB_pred) @ T_base_CB_meas",
            "translation_metric": "norm of translation component of T_err",
            "rotation_metric": "rotation angle of rotation component of T_err",
        },
        "results": {
            "held_out_test_set": {
                "nominal": {**nominal_test_summary, "per_pose": rows_to_list(nominal_test_rows)},
                "calibrated": {**calibrated_test_summary, "per_pose": rows_to_list(calibrated_test_rows)},
                "improvement_percent": improvement_test,
            },
            "all_usable_poses_reference": {
                "nominal": {**summarize(usable_nominal_rows), "per_pose": rows_to_list(usable_nominal_rows)},
                "calibrated": {**summarize(usable_calibrated_rows), "per_pose": rows_to_list(usable_calibrated_rows)},
            },
        },
        "diagnostics_and_next_steps": make_diagnostic_advice(triang_json, nominal_test_summary, calibrated_test_summary, best),
    }

    write_json(PATHS["calib_results_dir"] / "dh_calibrated_4cam.json", dh_out)
    write_json(PATHS["calib_results_dir"] / "calibration_report_4cam.json", calibration_report)
    write_json(PATHS["valid_results_dir"] / "validation_report_4cam.json", validation_report)
    write_validation_csv(nominal_test_rows, PATHS["valid_results_dir"] / "validation_nominal_test_per_pose_4cam.csv")
    write_validation_csv(calibrated_test_rows, PATHS["valid_results_dir"] / "validation_calibrated_test_per_pose_4cam.csv")

    print(f"\nWrote: {PATHS['calib_results_dir'] / 'dh_calibrated_4cam.json'}")
    print(f"Wrote: {PATHS['calib_results_dir'] / 'calibration_report_4cam.json'}")
    print(f"Wrote: {PATHS['valid_results_dir'] / 'validation_report_4cam.json'}")
    return calibration_report, validation_report


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    ensure_dir(PATHS["triang_results_dir"])
    triang_json, joints_json = run_triangulation()

    print("\n================ 4-CAM TRIANGULATION SUMMARY ================")
    tri_sum = triang_json["summary"]
    print(f"Triangulation successes: {tri_sum['num_success']}/{tri_sum['num_images_total']}")
    print(f"Saved triangulation results to: {PATHS['triang_results_dir']}")
    print("\nNext run:")
    print("  python 02_calibration_4cam.py")


if __name__ == "__main__":
    main()
