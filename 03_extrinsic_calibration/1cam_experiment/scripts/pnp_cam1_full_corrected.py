"""
Corrected 1-camera PnP pipeline for the thesis dataset.

Key corrections:
- Uses the ORIGINAL image for solvePnP (no pre-undistortion mismatch).
- If image resolution differs from intrinsic-calibration resolution,
  the intrinsic matrix K is scaled to the actual image size.
- Preserves the original thesis board convention used in the old pipeline:
  board origin is chosen as the detected corner closest to the image top-left
  by logical 0/90/180/270 grid rotation.

Outputs:
- pnp_cam1_poses_corrected.json
- ur3a_joints_data_cam1.json
- visualization images
"""

from __future__ import annotations
import os
import re
import json
import glob
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(r"C:\Users\kaysa\OneDrive\Desktop\robot-hand-eye-calibration-experiments\03_extrinsic_calibration")
REPO_ROOT = ROOT.parents[0]

IMAGES_DIR = ROOT / "images" / "cam1_images"
INTRINSICS_JSON = REPO_ROOT / "02_intrinsic_calibration" / "cam1" / "04_results" / "intrinsics_cam1.json"
DATASET_JSON = ROOT / "shared_inputs" / "dataset_ur3a_joint_images_40poses.json"
OUT_ROOT = ROOT / "1cam_experiment" / "results" / "01_pnp_cam1"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_intrinsics(json_path: Path):
    data = load_json(json_path)
    K = np.array(data["intrinsics_K"]["matrix_3x3"], dtype=np.float64)
    dist = np.array(data["distortion"]["coefficients"], dtype=np.float64).reshape(-1, 1)
    cols = int(data["pattern_size_inner_corners"]["cols"])
    rows = int(data["pattern_size_inner_corners"]["rows"])
    square_size_m = float(data.get("square_size_m", 0.02))
    image_w = int(data["image_size_wh"]["width"])
    image_h = int(data["image_size_wh"]["height"])
    return data, K, dist, (cols, rows), square_size_m, (image_w, image_h)


def scale_intrinsics(K: np.ndarray, calib_size_wh, image_size_wh):
    calib_w, calib_h = calib_size_wh
    img_w, img_h = image_size_wh
    sx = img_w / calib_w
    sy = img_h / calib_h
    K_scaled = K.copy()
    K_scaled[0, 0] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[0, 2] *= sx
    K_scaled[1, 2] *= sy
    return K_scaled, sx, sy


def reorder_corners_assume_image_topleft(corners, cols, rows):
    grid = corners.reshape(rows, cols, 1, 2)[:, :, 0, :]
    best = None
    best_score = float("inf")
    for k in range(4):
        g = np.rot90(grid, k=k)
        if g.shape[0] == rows and g.shape[1] == cols:
            g_rc = g
        else:
            g_rc = np.transpose(g, (1, 0, 2))
        origin = g_rc[0, 0]
        score = float(origin[0] + origin[1])
        if score < best_score:
            best_score = score
            best = g_rc.reshape(-1, 1, 2).astype(np.float32)
    return best


def make_object_points(cols, rows, square_size_m):
    objp = np.zeros((cols * rows, 3), dtype=np.float64)
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    objp[:, 0] = xs.reshape(-1) * square_size_m
    objp[:, 1] = ys.reshape(-1) * square_size_m
    return objp


def reprojection_rmse(objp, imgp, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    imgp2 = imgp.reshape(-1, 2)
    err = proj - imgp2
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def parse_index_from_name(name: str):
    m = re.search(r"img(\d+)_cam1\.(png|jpg|jpeg)$", name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def main():
    out_vis = ensure_dir(OUT_ROOT / "visualizations")
    ensure_dir(OUT_ROOT)

    dataset_json = load_json(DATASET_JSON)
    intr_data, K, dist, pattern_size, square_size_m, calib_size = load_intrinsics(INTRINSICS_JSON)
    cols, rows = pattern_size
    objp = make_object_points(cols, rows, square_size_m)

    dataset_by_pose = {}
    for p in dataset_json["poses"]:
        pose_id = int(p["pose_id"])
        img_base = p["images"]["cam1"]
        img_name = f"{Path(img_base).name}.png"
        joints_deg_dict = p["joint_angles_deg"]
        joints_deg = [
            float(joints_deg_dict["joint_1"]),
            float(joints_deg_dict["joint_2"]),
            float(joints_deg_dict["joint_3"]),
            float(joints_deg_dict["joint_4"]),
            float(joints_deg_dict["joint_5"]),
            float(joints_deg_dict["joint_6"]),
        ]
        joints_rad = np.deg2rad(np.array(joints_deg, dtype=float))
        dataset_by_pose[pose_id] = {
            "image": img_name,
            "joints_deg": joints_deg,
            "joints_rad": joints_rad.tolist(),
        }

    img_paths = sorted(glob.glob(os.path.join(str(IMAGES_DIR), "img*_cam1.png")))
    if not img_paths:
        raise FileNotFoundError(f"No images found in: {IMAGES_DIR}")

    poses = []
    joints_out = []
    num_ok = 0

    for p in img_paths:
        img_name = os.path.basename(p)
        pose_id = parse_index_from_name(img_name)
        if pose_id is None or pose_id not in dataset_by_pose:
            continue

        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            poses.append({"pose_id": pose_id, "image": img_name, "success": False})
            continue

        h, w = img.shape[:2]
        K_used, sx, sy = scale_intrinsics(K, calib_size, (w, h))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCornersSB(
            gray, (cols, rows),
            flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )

        entry = {"pose_id": pose_id, "image": img_name, "success": False}
        vis = img.copy()
        if found and corners is not None and len(corners) == cols * rows:
            corners = reorder_corners_assume_image_topleft(corners, cols, rows)
            ok, rvec, tvec = cv2.solvePnP(objp, corners, K_used, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                R, _ = cv2.Rodrigues(rvec)
                rmse = reprojection_rmse(objp, corners, rvec, tvec, K_used, dist)
                cv2.drawChessboardCorners(vis, (cols, rows), corners, True)
                origin = corners[0, 0, :]
                cv2.drawMarker(vis, (int(origin[0]), int(origin[1])), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=28, thickness=2)
                rvec = rvec.reshape(3)
                tvec = tvec.reshape(3)
                T = np.eye(4, dtype=float)
                T[:3, :3] = R.astype(float)
                T[:3, 3] = tvec.astype(float)
                entry.update({
                    "pose_id": pose_id,
                    "image": img_name,
                    "success": True,
                    "rvec_rad": rvec.astype(float).tolist(),
                    "tvec_m": tvec.astype(float).tolist(),
                    "R_3x3": R.astype(float).tolist(),
                    "reproj_rmse_px": float(rmse),
                    "T_board_to_cam_4x4": T.tolist(),
                    "camera_matrix_used_3x3": K_used.astype(float).tolist(),
                    "intrinsics_scale": {"sx": float(sx), "sy": float(sy)},
                })
                num_ok += 1

        cv2.imwrite(str(out_vis / img_name), vis)
        poses.append(entry)
        joints_info = dataset_by_pose[pose_id]
        joints_out.append({
            "pose_id": pose_id,
            "image": img_name,
            "joints_deg": joints_info["joints_deg"],
            "joints_rad": joints_info["joints_rad"],
        })
        print(f"{img_name}: {'OK' if entry['success'] else 'NO'}")

    poses_sorted = sorted(poses, key=lambda d: d["pose_id"])
    joints_sorted = sorted(joints_out, key=lambda d: d["pose_id"])

    pnp_out = {
        "camera_id": intr_data.get("camera_id", "cam1"),
        "image_directory": str(IMAGES_DIR),
        "intrinsics_used": {
            "K_3x3": K.astype(float).tolist(),
            "dist_coeffs": dist.reshape(-1).astype(float).tolist(),
            "image_size_wh": {"width": calib_size[0], "height": calib_size[1]},
        },
        "pattern": {"cols": cols, "rows": rows, "square_size_m": square_size_m},
        "pose_definition": "solvePnP returns board->camera: X_cam = R*X_board + t; rvec is Rodrigues(R).",
        "origin_convention": "board (0,0) is forced to the detected corner closest to image top-left via 0/90/180/270 logical grid rotation.",
        "important_note": "If image resolution differs from the intrinsic-calibration resolution, the camera matrix is scaled to the actual image size before solvePnP.",
        "summary": {
            "num_images_total": len(img_paths),
            "num_success": num_ok,
            "num_failed": len(img_paths) - num_ok,
            "num_missing_in_dataset": 0,
        },
        "poses": poses_sorted,
    }

    joints_json = {
        "robot": dataset_json.get("robot", "UR3a"),
        "units": {"joint_angles": "rad"},
        "source_dataset": str(DATASET_JSON),
        "camera_used": "cam1",
        "poses": joints_sorted,
    }

    pnp_path = OUT_ROOT / "pnp_cam1_poses_corrected.json"
    joints_path = OUT_ROOT / "ur3a_joints_data_cam1.json"
    with open(pnp_path, "w", encoding="utf-8") as f:
        json.dump(pnp_out, f, indent=2)
    with open(joints_path, "w", encoding="utf-8") as f:
        json.dump(joints_json, f, indent=2)

    print("\nDone.")
    print(f"Saved corrected PnP JSON: {pnp_path}")
    print(f"Saved joints JSON: {joints_path}")
    print(f"PnP successes: {num_ok}/{len(img_paths)}")


if __name__ == "__main__":
    main()
