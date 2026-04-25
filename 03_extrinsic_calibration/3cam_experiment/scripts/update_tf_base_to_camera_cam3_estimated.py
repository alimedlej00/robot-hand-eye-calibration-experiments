from pathlib import Path
import json
import shutil
from datetime import datetime

ROOT = Path(r"C:\Users\kaysa\OneDrive\Desktop\robot-hand-eye-calibration-experiments\03_extrinsic_calibration")
TF_PATH = ROOT / "shared_inputs" / "tf_base_to_camera.json"

ESTIMATED_CAM3 = {
    "T_4x4": [
        [0.999208114085529, 0.03511921953109053, -0.018702544350085594, 0.0011506975039146075],
        [0.023847517194553518, -0.1523307397123962, 0.9880418218184541, -1.121615133385428],
        [0.031850285230988526, -0.9877054146648185, -0.15304761733690392, 0.6394621278178275],
        [0.0, 0.0, 0.0, 1.0],
    ],
    "source": "estimated_from_cam3_PnP_robot_FK_fixed_T_EE_CB",
    "note": "Updated after cam3 transform estimation. Original file was backed up before replacement."
}


def main():
    if not TF_PATH.exists():
        raise FileNotFoundError(f"Could not find: {TF_PATH}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = TF_PATH.with_name(f"tf_base_to_camera_BACKUP_before_cam3_update_{timestamp}.json")
    shutil.copy2(TF_PATH, backup_path)

    with open(TF_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "T_base_to_camera_extrinsics" not in data:
        raise KeyError("Expected key not found: T_base_to_camera_extrinsics")
    if "cam3" not in data["T_base_to_camera_extrinsics"]:
        raise KeyError("Expected cam3 block not found in T_base_to_camera_extrinsics")

    old_cam3 = data["T_base_to_camera_extrinsics"]["cam3"]
    data["T_base_to_camera_extrinsics"]["cam3"] = ESTIMATED_CAM3

    data.setdefault("update_history", []).append({
        "update": "cam3 transform replaced with estimated transform",
        "backup_file": str(backup_path),
        "old_cam3": old_cam3,
        "new_cam3": ESTIMATED_CAM3,
    })

    with open(TF_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Done. cam3 transform updated safely.")
    print(f"Original backup saved as:\n{backup_path}")
    print("Now rerun:")
    print(r"python run_3cam_triangulate_calibrate_validate_v2.py")


if __name__ == "__main__":
    main()
