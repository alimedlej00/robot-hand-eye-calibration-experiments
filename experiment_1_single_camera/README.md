# Experiment 1 – Single Camera Hand–Eye Calibration

This directory contains the datasets, scripts, input parameters, and results used for the **single-camera calibration experiment** performed in the master thesis:

**"Experimental Investigation of Hand-to-Eye Configuration Effects on Vision-Based Robot Arm Calibration"**

The experiment evaluates the estimation of the camera-to-base transformation in a hand–eye configuration using a UR3e robot and a checkerboard calibration target observed by a fixed external camera.

---

# Overview of the Calibration Pipeline

The experiment follows a multi-stage calibration workflow:

1. Raw images of a checkerboard attached to the robot end-effector are captured using a fixed camera.
2. Checkerboard corners are detected and the **Perspective-n-Point (PnP)** algorithm estimates the camera-to-checkerboard pose for each image.
3. Robot joint configurations corresponding to each captured image are recorded.
4. The robot forward kinematics are computed using the **nominal UR3e DH parameters**.
5. A calibration optimization procedure estimates refined transformation parameters.
6. The resulting calibration is validated using independent robot poses.

---

# Folder Structure

## 01_raw_images
Contains the raw calibration images captured during the experiment.

These images are used for checkerboard detection and pose estimation.

---

## 02_checkerboard_pose_visualization
Contains processed images where:

- checkerboard corners are detected
- pose estimation results are visualized
- the checkerboard origin and coordinate frame are displayed

These visualizations help verify correct detection and pose estimation.

---

## 03_scripts
Contains the Python scripts used to execute the calibration pipeline.

Main scripts:

- **calibrate_intrinsics_cam1.py**  
  Detects checkerboard corners and computes camera intrinsic parameters like distortion coefficients.
  
- **pnp_cam1_from_images.py**  
  Detects checkerboard corners and computes camera-to-checkerboard poses using the PnP algorithm.

- **calibrate.py**  
  Performs calibration optimization using robot kinematics, PnP poses, and known transformations.

- **validate_dh_1cam.py**  
  Evaluates the calibration results by comparing predicted and observed poses.

---

## 04_input_parameters
Contains input configuration files required for the calibration.

Files include:

- `ur3e_dh_table_nominal.json`  
  Nominal Denavit–Hartenberg parameters of the UR3e robot.

- `Transformation_EE_to_CB.json`  
  Fixed transformation between the robot end-effector and the checkerboard.

- `transformation_camera_base.json`  
  Initial estimate of the camera-to-base transformation.

- `ur3e_joints_data_cam1.json`  
  Recorded robot joint configurations corresponding to captured images.

---

## 05_results
Contains the outputs produced by the calibration pipeline.

Files include:

- `intrinsics_cam1.json`  
  Camera intrinsic parameters and checkerboard configuration.
  
- `pnp_cam1_poses.json`  
  Estimated camera-to-checkerboard poses obtained from image observations.

- `calibration_report_1cam.json`  
  Summary of calibration performance metrics.

- `dh_calibrated_1cam.json`  
  Updated robot kinematic parameters after calibration.

- `validation_report_1cam.json`  
  Quantitative validation results of the calibrated model.

---

# Reproducibility

All scripts, parameters, and datasets required to reproduce the experiment are provided in this repository.

The repository structure separates:

- raw data
- processing scripts
- input parameters
- calibration results

to facilitate transparency and reproducibility of the experimental evaluation.
