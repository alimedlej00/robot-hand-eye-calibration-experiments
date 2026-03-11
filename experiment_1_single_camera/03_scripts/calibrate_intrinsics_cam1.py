import json
import numpy as np
import cv2
from pathlib import Path

# -----------------------------
# Configuration (edit if needed)
# -----------------------------
NPZ_PATH = Path("detected_corners_cam1.npz")   # in the same folder as this script
SQUARE_SIZE_M = 0.02                           # 2 cm squares -> 0.02 meters
OUTPUT_JSON = Path("intrinsics_cam1.json")

# Termination criteria for corner refinement (not used here; corners already refined in your pipeline)
# Included for completeness.
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)

def compute_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, K, dist):
    """Compute per-image and mean reprojection error in pixels."""
    per_image = []
    total_err = 0.0
    total_points = 0

    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], projected, cv2.NORM_L2)

        n = len(projected)
        rmse = np.sqrt((err * err) / n)  # RMSE per image
        per_image.append(float(rmse))

        total_err += err * err
        total_points += n

    mean_rmse = float(np.sqrt(total_err / total_points)) if total_points > 0 else float("nan")
    return per_image, mean_rmse

def fov_from_intrinsics(image_size_wh, fx, fy):
    """Compute FoV in degrees from calibrated fx, fy and image resolution."""
    W, H = image_size_wh
    fov_h = 2.0 * np.arctan(W / (2.0 * fx))
    fov_v = 2.0 * np.arctan(H / (2.0 * fy))
    return float(np.degrees(fov_h)), float(np.degrees(fov_v))

def main():
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"Could not find: {NPZ_PATH.resolve()}")

    data = np.load(str(NPZ_PATH), allow_pickle=True)

    pattern_size = tuple(data["pattern_size"].tolist())  # (cols, rows) = (9,7)
    image_size = tuple(data["image_size"].tolist())      # (W, H) from your inspect script
    used_images = data["used_images"].tolist()
    rejected_images = data["rejected_images"].tolist()

    image_points = data["image_points"]  # object array: list-like
    n_images = len(image_points)

    if n_images == 0:
        raise RuntimeError("No image points found inside the NPZ file.")

    # -----------------------------
    # Build object points (3D) grid
    # -----------------------------
    cols, rows = pattern_size  # inner corners
    objp = np.zeros((rows * cols, 3), np.float32)

    # OpenCV uses pattern as (cols, rows) but points are typically generated row-major:
    # x increases along cols, y increases along rows.
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(SQUARE_SIZE_M)

    # Same object points repeated for every image
    objpoints = [objp.copy() for _ in range(n_images)]

    # Ensure imgpoints are in the exact format OpenCV expects: list of (N,1,2) float32 arrays
    imgpoints = [np.array(pts, dtype=np.float32) for pts in image_points]

    # -----------------------------
    # Run intrinsic calibration
    # -----------------------------
    # Flags kept default for clarity; you can add flags later if you have a justified constraint.
    flags = 0

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=flags
    )

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # Distortion vector shape varies; store as flat list
    dist_list = dist.reshape(-1).astype(float).tolist()

    # Compute reprojection error independently (mean + per-image)
    per_image_rmse, mean_rmse = compute_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, K, dist)

    # Compute effective FoV from intrinsics
    fov_h_deg, fov_v_deg = fov_from_intrinsics(image_size, fx, fy)

    # Prepare JSON-safe export
    result = {
        "camera_id": "cam1",
        "image_size_wh": {"width": int(image_size[0]), "height": int(image_size[1])},
        "pattern_size_inner_corners": {"cols": int(cols), "rows": int(rows)},
        "square_size_m": float(SQUARE_SIZE_M),
        "num_images_used": int(n_images),
        "used_images": used_images,
        "rejected_images": rejected_images,

        "intrinsics_K": {
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
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

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Print concise summary for your records
    print("=== Intrinsic Calibration Result (cam1) ===")
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
    print(f"\nSaved JSON: {OUTPUT_JSON.resolve()}")

if __name__ == "__main__":
    main()
