import os
import re
import json
import glob
from pathlib import Path

import numpy as np
import cv2


def load_intrinsics(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    K = np.array(data["intrinsics_K"]["matrix_3x3"], dtype=np.float64)
    dist = np.array(data["distortion"]["coefficients"], dtype=np.float64).reshape(-1, 1)

    cols = int(data["pattern_size_inner_corners"]["cols"])
    rows = int(data["pattern_size_inner_corners"]["rows"])
    square_size_m = float(data.get("square_size_m", 0.02))

    image_w = int(data["image_size_wh"]["width"])
    image_h = int(data["image_size_wh"]["height"])

    return data, K, dist, (cols, rows), square_size_m, (image_w, image_h)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def reorder_corners_assume_image_topleft(corners, cols, rows):
    """
    Enforce your convention:
      board corner (0,0) := whichever detected corner is closest to the IMAGE top-left (min x+y),
      by rotating the (rows x cols) corner grid in 90-degree steps.
    """
    grid = corners.reshape(rows, cols, 1, 2)[:, :, 0, :]  # (rows, cols, 2)

    best = None
    best_score = float("inf")

    for k in range(4):
        g = np.rot90(grid, k=k)
        if g.shape[0] == rows and g.shape[1] == cols:
            g_rc = g
        else:
            g_rc = np.transpose(g, (1, 0, 2))  # (cols, rows, 2) -> (rows, cols, 2)

        origin = g_rc[0, 0]
        score = float(origin[0] + origin[1])

        if score < best_score:
            best_score = score
            best = g_rc.reshape(-1, 1, 2).astype(np.float32)

    return best


def make_object_points(cols, rows, square_size_m):
    """
    Board/object frame:
      Z = 0 plane
      (0,0,0) at chosen origin corner
      x increases along columns, y increases along rows
    """
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
    m = re.search(r"cam1_img(\d+)\.jpg$", name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def main():
    # ========= USER PATHS =========
    images_dir = r"C:\Users\kaysa\OneDrive\Pictures\Camera Roll\extrinsic_calibration\one_cam_setup\raw_images"
    intrinsics_json = r"C:\Users\kaysa\OneDrive\Pictures\Camera Roll\extrinsic_calibration\one_cam_setup\intrinsics_cam1.json"
    # ==============================

    out_root = Path(images_dir) / "_pnp_cam1"
    out_vis = ensure_dir(out_root / "visualizations")     # optional debug images
    out_undist = ensure_dir(out_root / "undistorted")     # optional undistorted images

    data, K, dist, pattern_size, square_size_m, expected_size = load_intrinsics(intrinsics_json)
    cols, rows = pattern_size
    patternSize = (cols, rows)

    objp = make_object_points(cols, rows, square_size_m)

    img_paths = sorted(glob.glob(os.path.join(images_dir, "cam1_img*.jpg")))
    if not img_paths:
        raise FileNotFoundError(f"No images found matching cam1_img*.jpg in: {images_dir}")

    # Precompute undistortion map for expected size
    newK = K.copy()
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, R=None, newCameraMatrix=newK,
        size=expected_size, m1type=cv2.CV_16SC2
    )

    poses = []
    num_ok = 0

    for p in img_paths:
        img_name = os.path.basename(p)
        idx = parse_index_from_name(img_name)

        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Could not read: {p}")
            poses.append({"index": idx, "image": img_name, "success": False})
            continue

        h, w = img.shape[:2]

        # Undistort (handles size mismatch)
        if (w, h) != expected_size:
            map1_i, map2_i = cv2.initUndistortRectifyMap(
                K, dist, R=None, newCameraMatrix=newK,
                size=(w, h), m1type=cv2.CV_16SC2
            )
            undist = cv2.remap(img, map1_i, map2_i, interpolation=cv2.INTER_LINEAR)
        else:
            undist = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(str(out_undist / img_name), undist)

        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCornersSB(
            gray, patternSize,
            flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )

        success = bool(found and corners is not None and len(corners) == cols * rows)

        entry = {
            "index": idx,
            "image": img_name,
            "success": False,
            # If success:
            # "rvec_rad": [rx, ry, rz],
            # "tvec_m": [tx, ty, tz],
            # "R_3x3": [[...],[...],[...]],
            # "reproj_rmse_px": value
        }

        vis = undist.copy()

        if success:
            corners = reorder_corners_assume_image_topleft(corners, cols, rows)

            ok, rvec, tvec = cv2.solvePnP(
                objp, corners, K, dist,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if ok:
                R, _ = cv2.Rodrigues(rvec)
                rmse = reprojection_rmse(objp, corners, rvec, tvec, K, dist)

                # Optional visualization
                cv2.drawChessboardCorners(vis, patternSize, corners, True)
                origin = corners[0, 0, :]
                cv2.drawMarker(vis, (int(origin[0]), int(origin[1])),
                               (0, 0, 255), markerType=cv2.MARKER_CROSS,
                               markerSize=28, thickness=2)

                rvec = rvec.reshape(3)
                tvec = tvec.reshape(3)

                entry["success"] = True
                entry["rvec_rad"] = rvec.astype(float).tolist()
                entry["tvec_m"] = tvec.astype(float).tolist()
                entry["R_3x3"] = R.astype(float).tolist()
                entry["reproj_rmse_px"] = float(rmse)

                T = np.eye(4, dtype=float)
                T[:3, :3] = R.astype(float)
                T[:3, 3] = tvec.astype(float)
                entry["T_board_to_cam_4x4"] = T.tolist()

                num_ok += 1

        cv2.imwrite(str(out_vis / img_name), vis)
        poses.append(entry)

        print(f"{img_name}: {'OK' if entry['success'] else 'NO'}")

    # Sort by image index if available
    poses_sorted = sorted(poses, key=lambda d: (d["index"] is None, d["index"] if d["index"] is not None else 10**9))

    out = {
        "camera_id": data.get("camera_id", "cam1"),
        "intrinsics_used": {
            "K_3x3": K.astype(float).tolist(),
            "dist_coeffs": dist.reshape(-1).astype(float).tolist(),
            "image_size_wh": {"width": expected_size[0], "height": expected_size[1]},
        },
        "pattern": {"cols": cols, "rows": rows, "square_size_m": square_size_m},
        "pose_definition": "solvePnP returns board->camera: X_cam = R*X_board + t; rvec is Rodrigues(R).",
        "origin_convention": "board (0,0) forced to the detected corner closest to image top-left (min x+y) via 0/90/180/270 grid rotation",
        "summary": {"num_images_total": len(img_paths), "num_success": num_ok},
        "poses": poses_sorted,
    }

    out_path = out_root / "pnp_cam1_poses.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nDone.")
    print(f"Saved poses JSON: {out_path}")
    print(f"PnP OK: {num_ok}/{len(img_paths)}")


if __name__ == "__main__":
    main()