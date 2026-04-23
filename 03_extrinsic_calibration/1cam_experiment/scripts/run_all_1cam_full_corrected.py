from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    ROOT / "pnp_cam1_full_corrected.py",
    ROOT / "calibrate_dh_1cam_full_corrected.py",
    ROOT / "validate_dh_1cam_full_corrected.py",
]

for script in SCRIPTS:
    print("\n" + "="*80)
    print(f"RUNNING: {script.name}")
    print("="*80)
    runpy.run_path(str(script), run_name="__main__")
