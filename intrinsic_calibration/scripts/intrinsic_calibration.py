import json
import re
from pathlib import Path

import cv2
import numpy as np

# =========================================================
# USER SETTINGS
# =========================================================

SQUARE_SIZE_M = 0.02
NPZ_GLOB = "*.npz"

# Optional:
# If True, save JSON in the same folder as the script
# If False, you can change OUTPUT_DIR below
SAVE_JSON_NEXT_TO_SCRIPT = True

# If SAVE_JSON_NEXT_TO_SCRIPT = False, this will be used
OUTPUT_DIR = None

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def compute_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, K, dist):
    """Compute per-image and overall mean reprojection RMSE in pixels."""
    per_image = []
    total_squared_error = 0.0
    total_points = 0

    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], projected, cv2.NORM_L2)

        n = len(projected)
        rmse = np.sqrt((err * err) / n)
        per_image.append(float(rmse))

        total_squared_error += err * err
        total_points += n

    mean_rmse = float(np.sqrt(total_squared_error / total_points)) if total_points > 0 else float("nan")
    return per_image, mean_rmse


def fov_from_intrinsics(image_size_wh, fx, fy):
    """Compute horizontal and vertical field of view in degrees."""
    W, H = image_size_wh
    fov_h = 2.0 * np.arctan(W / (2.0 * fx))
    fov_v = 2.0 * np.arctan(H / (2.0 * fy))
    return float(np.degrees(fov_h)), float(np.degrees(fov_v))


def extract_camera_id(npz_path: Path):
    """
    Try to infer camera ID from filename like:
    detected_corners_cam1.npz -> cam1
    corners_cam2.npz -> cam2
    If not found, use parent folder name.
    """
    match = re.search(r"(cam\d+)", npz_path.stem.lower())
    if match:
        return match.group(1)
    return npz_path.parent.name.lower()


def find_single_npz(script_dir: Path):
    """Find exactly one NPZ file in the script folder."""
    npz_files = sorted(script_dir.glob(NPZ_GLOB))

    if len(npz_files) == 0:
        raise FileNotFoundError(
            f"No NPZ file found in: {script_dir}\n"
            f"Expected something like: detected_corners_cam1.npz"
        )

    if len(npz_files) > 1:
        raise RuntimeError(
            f"Multiple NPZ files found in: {script_dir}\n"
            f"Please keep only one NPZ file there or modify NPZ_GLOB.\n"
            f"Found: {[p.name for p in npz_files]}"
        )

    return npz_files[0]


def build_object_points(pattern_size, square_size_m):
    """Create checkerboard object points in meters."""
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size_m)
    return objp


# =========================================================
# MAIN
# =========================================================

def main():
    script_dir = Path(__file__).resolve().parent

    # Automatically find NPZ in the same folder as the script
    npz_path = find_single_npz(script_dir)

    # Infer camera ID
    camera_id = extract_camera_id(npz_path)

    # Decide output directory
    if SAVE_JSON_NEXT_TO_SCRIPT:
        output_dir = script_dir
    else:
        output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR is not None else script_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f"intrinsics_{camera_id}.json"

    print(f"Using NPZ file: {npz_path.name}")
    print(f"Inferred camera ID: {camera_id}")

    data = np.load(str(npz_path), allow_pickle=True)

    pattern_size = tuple(data["pattern_size"].tolist())   # (cols, rows)
    image_size = tuple(data["image_size"].tolist())       # (W, H)
    used_images = data["used_images"].tolist()
    rejected_images = data["rejected_images"].tolist()
    image_points = data["image_points"]

    n_images = len(image_points)
    if n_images == 0:
        raise RuntimeError("No image points found inside the NPZ file.")

    # Build checkerboard 3D points
    cols, rows = pattern_size
    objp = build_object_points(pattern_size, SQUARE_SIZE_M)
    objpoints = [objp.copy() for _ in range(n_images)]

    # Ensure correct OpenCV format
    imgpoints = [np.array(pts, dtype=np.float32) for pts in image_points]

    # Run intrinsic calibration
    flags = 0
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=flags
    )

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    dist_list = dist.reshape(-1).astype(float).tolist()

    # Quality metrics
    per_image_rmse, mean_rmse = compute_reprojection_errors(
        objpoints, imgpoints, rvecs, tvecs, K, dist
    )

    # Field of view
    fov_h_deg, fov_v_deg = fov_from_intrinsics(image_size, fx, fy)

    # JSON export
    result = {
        "camera_id": camera_id,
        "source_file": npz_path.name,
        "image_size_wh": {
            "width": int(image_size[0]),
            "height": int(image_size[1])
        },
        "pattern_size_inner_corners": {
            "cols": int(cols),
            "rows": int(rows)
        },
        "square_size_m": float(SQUARE_SIZE_M),
        "num_images_used": int(n_images),
        "used_images": used_images,
        "rejected_images": rejected_images,
        "intrinsics_K": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "matrix_3x3": K.astype(float).tolist()
        },
        "distortion": {
            "model": "opencv_standard",
            "coefficients": dist_list
        },
        "quality": {
            "rms_reported_by_opencv": float(rms),
            "mean_reprojection_rmse_px": float(mean_rmse),
            "per_image_reprojection_rmse_px": per_image_rmse
        },
        "field_of_view_deg": {
            "horizontal": float(fov_h_deg),
            "vertical": float(fov_v_deg)
        }
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Console summary
    print("\n=== Intrinsic Calibration Result ===")
    print(f"Camera ID: {camera_id}")
    print(f"Images used: {n_images}")
    print(f"Image size (W,H): {image_size}")
    print(f"Pattern (cols,rows): {pattern_size} inner corners")
    print(f"Square size: {SQUARE_SIZE_M} m")

    print("\nK =")
    print(K)

    print("\nDistortion coefficients =")
    print(dist.reshape(-1))

    print("\nQuality:")
    print(f"OpenCV RMS: {float(rms):.4f} px")
    print(f"Mean reprojection RMSE: {float(mean_rmse):.4f} px")

    print("\nFoV (deg):")
    print(f"Horizontal: {fov_h_deg:.2f}°, Vertical: {fov_v_deg:.2f}°")

    print(f"\nSaved JSON: {output_json}")


if __name__ == "__main__":
    main()