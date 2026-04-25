from __future__ import annotations
"""
Standalone experimental patch:
Apply image-based refined cam1/cam2/cam3 extrinsics directly into:

C:\\Users\\kaysa\\OneDrive\\Desktop\\robot-hand-eye-calibration-experiments\\03_extrinsic_calibration\\shared_inputs\\tf_base_to_camera.json

No external refined JSON is needed.
All refined extrinsic matrices are embedded inside this file.

After running this:
    python run_3cam_robust_calibrate_validate.py

Important:
- This is experimental.
- The script creates a timestamped backup before editing.
- cam4 is not changed.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(r"C:\Users\kaysa\OneDrive\Desktop\robot-hand-eye-calibration-experiments\03_extrinsic_calibration")
SHARED_TF = ROOT / "shared_inputs" / "tf_base_to_camera.json"


# ============================================================
# IMAGE-BASED REFINED EXTRINSICS FROM 3-CAM EXPERIMENT
# Direction: base_T_camera
# Units: meters
# Source: image_based_extrinsic_plus_dh_experiment_3cam
# ============================================================

REFINED_EXTRINSICS = {
    "cam1": {
        "T_4x4": [
            [0.9997486107717183, 0.007714184193785511, -0.021052579584438822, 0.13764690921807266],
            [0.021171335285188372, -0.01565053405419336, 0.999653358074088, -1.1267418957492414],
            [0.0073820254727726044, -0.9998477657153279, -0.015809916750110144, 0.37823514115435064],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "source": "image_based_extrinsic_plus_dh_experiment_3cam_EXPERIMENTAL_RERUN",
        "note": "Refined by direct image-based optimization. Experimental; compare against robust triangulation result before accepting.",
    },

    "cam2": {
        "T_4x4": [
            [0.999779568145083, 0.01567875671030066, -0.013963609609406794, -0.11397026835573672],
            [0.014133137028124004, -0.010753967705954728, 0.9998422872113019, -1.1403084363560574],
            [0.015526123472759374, -0.9998192463040506, -0.010973185981588278, 0.36196440828064474],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "source": "image_based_extrinsic_plus_dh_experiment_3cam_EXPERIMENTAL_RERUN",
        "note": "Refined by direct image-based optimization. Experimental; compare against robust triangulation result before accepting.",
    },

    "cam3": {
        "T_4x4": [
            [0.9986895436294616, 0.036368397250183614, -0.03600743154725281, 0.019313919539737844],
            [0.04127625061602423, -0.15642918718137203, 0.9868263173086025, -1.1219193026523662],
            [0.030256678275389384, -0.9870193762430542, -0.1577253446358937, 0.6452247320675701],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "source": "image_based_extrinsic_plus_dh_experiment_3cam_EXPERIMENTAL_RERUN",
        "note": "Refined by direct image-based optimization after cam3 correction. Experimental; compare before accepting.",
    },
}


# Optional human-readable metadata from the experiment.
CAMERA_REFINEMENT_SUMMARY = {
    "cam1": {
        "xi_rot_deg": [0.07814640583732144, 0.319233218508385, 1.0846779146914023],
        "xi_trans_mm": [0.12477706535178072, -9.999999999817378, -1.449027436500502],
        "rotation_update_norm_deg": 1.1333767631126916,
        "translation_update_norm_mm": 10.105209044058247,
    },
    "cam2": {
        "xi_rot_deg": [0.6038952468066888, 0.1030575131771996, 1.2349186151640361],
        "xi_trans_mm": [1.3004446626432218, -9.999999999999913, 8.480941934094059],
        "rotation_update_norm_deg": 1.3785261354855194,
        "translation_update_norm_mm": 13.176400586278127,
    },
    "cam3": {
        "xi_rot_deg": [-0.2737023272665477, 0.0824789226299739, 0.9911083096970118],
        "xi_trans_mm": [-2.135282036838924, -3.567180404100634, 0.4285573954583137],
        "rotation_update_norm_deg": 1.0315092913686206,
        "translation_update_norm_mm": 4.179457722414304,
    },
}


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    if not SHARED_TF.exists():
        raise FileNotFoundError(f"Shared transform file not found:\n{SHARED_TF}")

    data = load_json(SHARED_TF)

    if "T_base_to_camera_extrinsics" not in data:
        raise KeyError("Could not find key: T_base_to_camera_extrinsics")

    extr = data["T_base_to_camera_extrinsics"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = SHARED_TF.with_name(
        f"tf_base_to_camera_BACKUP_before_embedded_image_based_extrinsics_{ts}.json"
    )
    shutil.copy2(SHARED_TF, backup_path)

    old_blocks = {}
    for cam_id in ["cam1", "cam2", "cam3"]:
        if cam_id not in extr:
            raise KeyError(f"Camera {cam_id} not found in tf_base_to_camera.json")

        old_blocks[cam_id] = extr[cam_id]
        extr[cam_id] = REFINED_EXTRINSICS[cam_id]

    update_entry = {
        "update": "cam1_cam2_cam3 replaced with embedded image-based refined extrinsics",
        "timestamp": ts,
        "backup_file": str(backup_path),
        "cameras_updated": ["cam1", "cam2", "cam3"],
        "cam4_changed": False,
        "source": "embedded_values_from_image_based_extrinsic_plus_dh_experiment_3cam",
        "warning": "Experimental rerun only. Compare results before using in thesis.",
        "camera_refinement_summary": CAMERA_REFINEMENT_SUMMARY,
        "old_cameras": old_blocks,
        "new_cameras": REFINED_EXTRINSICS,
    }

    data.setdefault("update_history", []).append(update_entry)

    save_json(SHARED_TF, data)

    print("\nDONE.")
    print("Updated cam1, cam2, cam3 using embedded image-based refined extrinsics.")
    print(f"Backup created:\n  {backup_path}")
    print("\nNow run:")
    print("  python run_3cam_robust_calibrate_validate.py")
    print("\nThen send:")
    print("  robust_calibration_report_3cam.json")
    print("  robust_validation_report_3cam.json")


if __name__ == "__main__":
    main()
