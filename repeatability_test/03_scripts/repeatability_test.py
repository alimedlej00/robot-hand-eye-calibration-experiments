import cv2
import numpy as np
import glob
import os
import json
import matplotlib.pyplot as plt

# =========================================================
# USER SETTINGS
# =========================================================

CHECKERBOARD = (9, 7)   # inner corners: (cols, rows)
SQUARE_SIZE_M = 0.02
IMAGE_GLOB = "rep_*.*"

# =========================================================
# CAM1 INTRINSICS
# =========================================================

camera_matrix = np.array([
    [1572.9338514293024, 0.0, 952.936772202578],
    [0.0, 1582.3983865805571, 578.2012480790197],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

dist_coeffs = np.array([
    -0.38114960202973464,
     0.09091853311094379,
    -0.00470872493609046,
    -0.0012806888098315411,
     0.07953303549259323
], dtype=np.float64)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def create_object_points(checkerboard, square_size_m):
    cols, rows = checkerboard
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_m
    return objp

def rotation_matrix_to_angle_deg(R):
    trace_val = np.trace(R)
    cos_theta = (trace_val - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    return float(np.degrees(theta_rad))

def summary_stats(arr):
    arr = np.array(arr, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr))
    }

def resize_to_height(img, height):
    scale = height / img.shape[0]
    width = int(img.shape[1] * scale)
    return cv2.resize(img, (width, height))

# =========================================================
# MAIN
# =========================================================

def main():
    image_files = sorted(glob.glob(IMAGE_GLOB))

    if not image_files:
        print("No images found. Make sure files are named like rep_001.jpg")
        return

    print(f"Found {len(image_files)} image(s).")

    objp = create_object_points(CHECKERBOARD, SQUARE_SIZE_M)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )

    used_images = []
    all_centers = []
    all_rvecs = []
    all_tvecs = []
    per_image_results = []

    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read: {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not found:
            print(f"Checkerboard NOT found in {img_path}")
            continue

        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        success, rvec, tvec = cv2.solvePnP(
            objp,
            corners_refined,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            print(f"solvePnP failed for {img_path}")
            continue

        center_xy = np.mean(corners_refined.reshape(-1, 2), axis=0)

        used_images.append(img_path)
        all_centers.append(center_xy)
        all_rvecs.append(rvec)
        all_tvecs.append(tvec)

        # Save checkerboard detection visualization
        vis = img.copy()
        cv2.drawChessboardCorners(vis, CHECKERBOARD, corners_refined, found)
        cv2.circle(vis, tuple(np.round(center_xy).astype(int)), 6, (0, 0, 255), -1)
        cv2.imwrite(f"detected_{os.path.basename(img_path)}", vis)

    if len(used_images) < 2:
        print("Not enough valid images with checkerboard detection.")
        return

    # First valid image is the reference
    ref_center = all_centers[0]
    ref_rvec = all_rvecs[0]
    ref_tvec = all_tvecs[0]
    R_ref, _ = cv2.Rodrigues(ref_rvec)

    center_shift_pixels = []
    translation_error_mm = []
    rotation_error_deg = []

    for i in range(len(used_images)):
        img_name = os.path.basename(used_images[i])
        center = all_centers[i]
        rvec = all_rvecs[i]
        tvec = all_tvecs[i]

        # Center shift in pixels
        center_shift = float(np.linalg.norm(center - ref_center))

        # Translation difference in mm
        t_error_m = float(np.linalg.norm((tvec - ref_tvec).reshape(3)))
        t_error_mm = t_error_m * 1000.0

        # Rotation difference in degrees
        R_i, _ = cv2.Rodrigues(rvec)
        R_rel = R_i @ R_ref.T
        r_error_deg = rotation_matrix_to_angle_deg(R_rel)

        center_shift_pixels.append(center_shift)
        translation_error_mm.append(t_error_mm)
        rotation_error_deg.append(r_error_deg)

        per_image_results.append({
            "image_name": img_name,
            "checkerboard_center_px": {
                "x": float(center[0]),
                "y": float(center[1])
            },
            "center_shift_px": center_shift,
            "translation_error_mm": t_error_mm,
            "rotation_error_deg": r_error_deg,
            "tvec_m": {
                "x": float(tvec[0]),
                "y": float(tvec[1]),
                "z": float(tvec[2])
            },
            "rvec_rad": {
                "x": float(rvec[0]),
                "y": float(rvec[1]),
                "z": float(rvec[2])
            }
        })

    # =========================================================
    # SAVE JSON
    # =========================================================

    results_json = {
        "experiment_type": "robot_repeatability_test",
        "description": "Repeatability evaluation using checkerboard pose estimation from repeated returns to the same robot pose.",
        "image_pattern_used": IMAGE_GLOB,
        "num_input_images_found": len(image_files),
        "num_valid_images_used": len(used_images),
        "reference_image": os.path.basename(used_images[0]),
        "checkerboard": {
            "inner_corners": {
                "cols": CHECKERBOARD[0],
                "rows": CHECKERBOARD[1]
            },
            "square_size_m": SQUARE_SIZE_M
        },
        "camera_intrinsics": {
            "camera_matrix_3x3": camera_matrix.tolist(),
            "distortion_coefficients": dist_coeffs.tolist()
        },
        "summary": {
            "center_shift_px": summary_stats(center_shift_pixels),
            "translation_error_mm": summary_stats(translation_error_mm),
            "rotation_error_deg": summary_stats(rotation_error_deg)
        },
        "per_image_results": per_image_results
    }

    with open("repeatability_results.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)

    # =========================================================
    # PRINT SUMMARY
    # =========================================================

    print("\n================ REPEATABILITY SUMMARY ================\n")
    print(f"Valid images used: {len(used_images)}")
    print("\nCenter shift [px]:")
    print(results_json["summary"]["center_shift_px"])
    print("\nTranslation error [mm]:")
    print(results_json["summary"]["translation_error_mm"])
    print("\nRotation error [deg]:")
    print(results_json["summary"]["rotation_error_deg"])

    # =========================================================
    # PLOTS
    # =========================================================

    # Plot 1: translation error
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(translation_error_mm) + 1), translation_error_mm, marker='o')
    plt.xlabel("Image index")
    plt.ylabel("Translation error [mm]")
    plt.title("Repeatability Test - Translation Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("repeatability_translation_mm.png", dpi=300)
    plt.close()

    # Plot 2: rotation error
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(rotation_error_deg) + 1), rotation_error_deg, marker='o')
    plt.xlabel("Image index")
    plt.ylabel("Rotation error [deg]")
    plt.title("Repeatability Test - Rotation Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("repeatability_rotation_deg.png", dpi=300)
    plt.close()

    # Plot 3: checkerboard center spread
    centers_np = np.array(all_centers)
    plt.figure(figsize=(6, 6))
    plt.scatter(centers_np[:, 0], centers_np[:, 1], marker='o')
    plt.scatter(ref_center[0], ref_center[1], marker='x', s=120)
    plt.xlabel("Center X [px]")
    plt.ylabel("Center Y [px]")
    plt.title("Checkerboard Center Spread")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("repeatability_center_spread.png", dpi=300)
    plt.close()

    # =========================================================
    # COMBINE THE 3 PLOTS INTO ONE HORIZONTAL IMAGE
    # =========================================================

    img1 = cv2.imread("repeatability_translation_mm.png")
    img2 = cv2.imread("repeatability_rotation_deg.png")
    img3 = cv2.imread("repeatability_center_spread.png")

    if img1 is None or img2 is None or img3 is None:
        print("One or more plot images not found. Skipping combined image.")
    else:
        target_height = min(img1.shape[0], img2.shape[0], img3.shape[0])

        img1_r = resize_to_height(img1, target_height)
        img2_r = resize_to_height(img2, target_height)
        img3_r = resize_to_height(img3, target_height)

        # Small white separator between figures
        separator_width = 30
        separator = np.ones((target_height, separator_width, 3), dtype=np.uint8) * 255

        combined = np.hstack((img1_r, separator, img2_r, separator, img3_r))
        cv2.imwrite("repeatability_combined.png", combined)

        print("Saved combined image: repeatability_combined.png")

    # =========================================================
    # FINAL OUTPUT MESSAGE
    # =========================================================

    print("\nSaved outputs:")
    print("  repeatability_results.json")
    print("  repeatability_translation_mm.png")
    print("  repeatability_rotation_deg.png")
    print("  repeatability_center_spread.png")
    print("  repeatability_combined.png")
    print("  detected_rep_*.jpg / detected_rep_*.png")

if __name__ == "__main__":
    main()