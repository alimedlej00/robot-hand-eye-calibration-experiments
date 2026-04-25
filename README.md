# Robot Hand–Eye Calibration Using Multi-Camera Fusion

This repository contains the full experimental framework, datasets, and results for the master thesis:

**“Fusion of Multiple Fixed Cameras for Automated Self-Calibration of Industrial Robot Arms”**

---

## 1. Overview

This work investigates how multiple fixed cameras can be fused to improve the accuracy and robustness of vision-based robot calibration.

A unified calibration pipeline is developed and evaluated across four configurations:

* Single-camera (monocular baseline)
* Two-camera (stereo configuration)
* Three-camera (multi-view redundancy)
* Four-camera (full configuration)

All experiments are conducted under identical conditions to ensure fair comparison.

---

## 2. Methodology

The calibration framework integrates geometric vision and robot kinematics:

* Perspective-n-Point (PnP) for checkerboard pose estimation
* Multi-view triangulation for 3D reconstruction
* Forward kinematics (FK) using the robot DH model
* Optimization-based calibration including:

  * Joint zero offset estimation
  * Denavit–Hartenberg (DH) parameter refinement (aᵢ, dᵢ)
  * Optional camera extrinsic refinement

Validation is performed by comparing:

* Vision-based pose estimation
* Kinematics-based prediction

Errors are evaluated in both translation (mm) and rotation (degrees).

---

## 3. Repository Structure

```
robot-hand-eye-calibration-experiments/

├── 02_intrinsic_calibration/
├── 03_extrinsic_calibration/
│   ├── shared_inputs/
│   │   ├── tf_base_to_camera.json
│   │   ├── transformation_ee_to_cb.json
│   │   ├── nominal_dh_table_ur3a.json
│   │
│   ├── 1cam_experiment/
│   ├── 2cam_experiment/
│   ├── 3cam_experiment/
│   ├── 4cam_experiment/
│
├── images/
│   ├── cam1_images/
│   ├── cam2_images/
│   ├── cam3_images/
│   ├── cam4_images/
```

Each experiment contains:

* Measurement data (PnP / triangulation outputs)
* Calibration results
* Validation reports
* Execution scripts

---

## 4. Experimental Setup

* Robot: UR3a
* Number of poses: 40
* Checkerboard: 9 × 7 inner corners
* Square size: 0.02 m
* Camera resolution: 1920×1080 (scaled during processing)

---

## 5. Experiments Description

### 5.1 Single Camera

Monocular setup without depth information.
Serves as a baseline for evaluating calibration limitations.

### 5.2 Two Cameras

Stereo configuration enabling depth estimation through triangulation.
Significant improvement is expected in translation accuracy.

### 5.3 Three Cameras

Adds redundancy and an additional viewpoint.
Improves robustness but introduces inter-camera dependency.

### 5.4 Four Cameras

Full multi-view configuration.
Provides maximum coverage but increases calibration sensitivity and system complexity.

---

## 6. Results and Quantitative Comparison

The performance of each configuration is evaluated using root mean square error (RMSE) on a held-out test set.

### 6.1 Translation Error (RMSE)

| Configuration | Translation RMSE (mm) | Improvement             |
| ------------- | --------------------- | ----------------------- |
| 1 Camera      | 8.04                  | Baseline                |
| 2 Cameras     | ~5–6                  | Significant improvement |
| 3 Cameras     | ~4–5                  | Moderate improvement    |
| 4 Cameras     | ~7–40 (unstable)      | Degraded performance    |

---

### 6.2 Rotation Error (RMSE)

| Configuration | Rotation RMSE (deg) | Observation           |
| ------------- | ------------------- | --------------------- |
| 1 Camera      | ~2.5                | Baseline              |
| 2 Cameras     | ~2.4–2.6            | No significant change |
| 3 Cameras     | ~1.7–2.0            | Slight improvement    |
| 4 Cameras     | ~2.7–3.0            | Slight degradation    |

---

### 6.3 Key Observations

* The transition from one to two cameras provides the largest improvement due to depth recovery.
* Additional cameras introduce redundancy but also increase sensitivity to calibration errors.
* The four-camera setup shows instability due to:

  * Higher reprojection errors
  * Increased number of rejected poses
* Rotation error is less sensitive to camera count compared to translation error.

---

### 6.4 Interpretation

The results demonstrate that:

* Depth information is the dominant factor in improving calibration accuracy.
* Increasing the number of cameras does not guarantee better performance.
* There exists a trade-off between:

  * Measurement redundancy
  * Calibration sensitivity and system complexity

A well-calibrated multi-camera system with minimal redundancy may outperform a larger but less stable configuration.

---

## 7. Reproducibility

To reproduce the experiments:

```bash
git clone https://github.com/<your-username>/robot-hand-eye-calibration-experiments.git
cd robot-hand-eye-calibration-experiments
pip install -r requirements.txt
```

Run a specific experiment:

```bash
cd 03_extrinsic_calibration/3cam_experiment
python run_pipeline.py
```

---

## 8. Data and Availability

* All private system paths have been removed
* Data is structured for reproducibility
* Experiments follow a consistent and modular pipeline

---

## 9. Author

Ali Lhadi Medlej
M.Sc. Mechatronics Engineering
Technische Hochschule Deggendorf

---

## 10. Notes

This repository is intended for:

* Academic evaluation
* Reproducible robotics research
* Demonstration of multi-camera calibration methodologies
