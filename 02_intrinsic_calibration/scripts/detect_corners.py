import os
import re
import glob
from pathlib import Path

import cv2
import numpy as np

# =========================================================
# USER SETTINGS
# =========================================================

PATTERN_SIZE = (9, 7)   # inner corners: (cols, rows)

# Relative paths from the script folder
INPUT_DIR = "../01_input_images"
OUTPUT_PREVIEW_DIR = "../04_detection_check"
OUTPUT_RESULTS_DIR = "../03_results"

# Sub-pixel refinement settings
SUBPIX_WIN = (11, 11)
SUBPIX_ZERO_ZONE = (-1, -1)
SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    1e-4
)

# Chessboard detection flags
FIND_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    + cv2.CALIB_CB_NORMALIZE_IMAGE
    + cv2.CALIB_CB_FAST_CHECK
)

# =========================================================
# HELPERS
# =========================================================

def natural_key(path: str):
    """Sort files like img_2.jpg before img_10.jpg."""
    base = os.path.basename(path)
    digits = "".join([c if c.isdigit() else " " for c in base]).split()
    return [int(digits[-1])] if digits else [base]


def list_images(folder: Path):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for ext in exts:
        files.extend(folder.glob(ext))
    return sorted(files, key=lambda p: natural_key(str(p)))


def infer_camera_id(script_dir: Path):
    """
    Infer camera id from parent folders, e.g. cam1, cam2, ...
    Falls back to 'camera' if not found.
    """
    for part in [script_dir.name, script_dir.parent.name, script_dir.parent.parent.name]:
        match = re.search(r"(cam\d+)", part.lower())
        if match:
            return match.group(1)
    return "camera"


# =========================================================
# MAIN
# =========================================================

def main():
    script_dir = Path(__file__).resolve().parent
    input_dir = (script_dir / INPUT_DIR).resolve()
    preview_dir = (script_dir / OUTPUT_PREVIEW_DIR).resolve()
    results_dir = (script_dir / OUTPUT_RESULTS_DIR).resolve()

    preview_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    camera_id = infer_camera_id(script_dir)
    output_npz_path = results_dir / f"detected_corners_{camera_id}.npz"

    image_paths = list_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(
            f"No images found in '{input_dir}'. "
            f"Put calibration images there."
        )

    all_img_points = []   # list of (N,1,2) float32 arrays
    used_images = []
    rejected_images = []
    image_size = None

    for idx, path in enumerate(image_paths, start=1):
        img = cv2.imread(str(path))
        if img is None:
            rejected_images.append(path.name)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (width, height)

        found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, FIND_FLAGS)

        if not found or corners is None:
            rejected_images.append(path.name)
            continue

        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            SUBPIX_WIN,
            SUBPIX_ZERO_ZONE,
            SUBPIX_CRITERIA
        )

        all_img_points.append(corners_refined.astype(np.float32))
        used_images.append(path.name)

        # Save preview image
        vis = img.copy()
        cv2.drawChessboardCorners(vis, PATTERN_SIZE, corners_refined, True)
        out_name = f"detected_{path.name}"
        cv2.imwrite(str(preview_dir / out_name), vis)

    if image_size is None:
        raise RuntimeError("Could not determine image size from input images.")

    np.savez(
        str(output_npz_path),
        camera_id=np.array(camera_id, dtype=object),
        pattern_size=np.array(PATTERN_SIZE, dtype=np.int32),
        image_size=np.array(image_size, dtype=np.int32),
        image_points=np.array(all_img_points, dtype=object),
        used_images=np.array(used_images, dtype=object),
        rejected_images=np.array(rejected_images, dtype=object),
    )

    print("=== Corner Detection Summary ===")
    print(f"Camera ID: {camera_id}")
    print(f"Input folder: {input_dir}")
    print(f"Pattern (inner corners): {PATTERN_SIZE[0]} x {PATTERN_SIZE[1]}")
    print(f"Total images found: {len(image_paths)}")
    print(f"Images accepted (corners found): {len(used_images)}")
    print(f"Images rejected: {len(rejected_images)}")
    print(f"Preview overlays saved to: {preview_dir}")
    print(f"Detected corners saved to: {output_npz_path}")

    if rejected_images:
        print("\nRejected files:")
        for name in rejected_images:
            print(f" - {name}")


if __name__ == "__main__":
    main()