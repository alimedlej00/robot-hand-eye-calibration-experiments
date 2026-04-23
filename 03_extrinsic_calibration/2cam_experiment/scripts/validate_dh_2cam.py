from __future__ import annotations
import json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

ROOT = Path(r"C:\Users\kaysa\OneDrive\Desktop\robot-hand-eye-calibration-experiments\03_extrinsic_calibration")
PATHS = {
    'triangulated': ROOT/'2cam_experiment'/'results'/'01_triangulation_2cam'/'triangulated_checkerboard_poses_2cam.json',
    'joints': ROOT/'2cam_experiment'/'results'/'01_triangulation_2cam'/'ur3a_joints_data_2cam.json',
    'dh_nominal': ROOT/'shared_inputs'/'robot_model_dh_nominal_ur3a.json',
    'dh_calibrated': ROOT/'2cam_experiment'/'results'/'02_calibration_2cam'/'dh_calibrated_2cam.json',
    'ee_cb': ROOT/'shared_inputs'/'tf_ee_to_cb.json',
    'calib_report': ROOT/'2cam_experiment'/'results'/'02_calibration_2cam'/'calibration_report_2cam.json',
    'output_json': ROOT/'2cam_experiment'/'results'/'03_validation_2cam'/'validation_report_2cam.json',
}
def load_json(path: Path)->dict:
    with open(path,'r',encoding='utf-8') as f: return json.load(f)
def ensure_parent_dir(path: Path)->None: path.parent.mkdir(parents=True, exist_ok=True)
def inv_T(T): R,t=T[:3,:3],T[:3,3]; Ti=np.eye(4); Ti[:3,:3]=R.T; Ti[:3,3]=-R.T@t; return Ti
def rot_angle(R): tr=float(np.trace(R)); c=max(-1.0,min(1.0,(tr-1.0)/2.0)); return math.acos(c)
def dh_A(a,alpha,d,theta):
    ca,sa=math.cos(alpha),math.sin(alpha); ct,st=math.cos(theta),math.sin(theta)
    return np.array([[ct,-st*ca,st*sa,a*ct],[st,ct*ca,-ct*sa,a*st],[0.0,sa,ca,d],[0.0,0.0,0.0,1.0]],dtype=float)
def fk_from_dh(joints_rad, dh_params, joint_zero_offsets=None):
    if joint_zero_offsets is None: joint_zero_offsets=np.zeros(6,dtype=float)
    T=np.eye(4,dtype=float)
    for q,off,p in zip(joints_rad,joint_zero_offsets,dh_params): T=T@dh_A(float(p['a']),float(p['alpha']),float(p['d']),float(q+off+p.get('theta_offset',0.0)))
    return T
@dataclass
class PoseSample:
    pose_id:int; image_pair:list; joints_rad:np.ndarray; T_base_CB:np.ndarray; reproj_mean_rmse_px:float
def build_samples(joints_json, triang_json):
    j_by={int(p['pose_id']):p for p in joints_json['poses']}; t_by={int(p['pose_id']):p for p in triang_json['poses']}; out=[]
    for pid in sorted(set(j_by)&set(t_by)):
        tj=t_by[pid]
        if not tj.get('success',True): continue
        jj=j_by[pid]
        out.append(PoseSample(pid,list(jj['image_pair']),np.array(jj['joints_rad'],dtype=float),np.array(tj['T_base_CB_4x4'],dtype=float),float(tj.get('reproj_mean_rmse_px',float('nan')))))
    return out
def reconstruct_train_test_split(samples,reproj_threshold_px,train_ratio,seed):
    usable=[s for s in samples if (not math.isnan(s.reproj_mean_rmse_px)) and s.reproj_mean_rmse_px<=reproj_threshold_px]
    rng=np.random.default_rng(seed); idx=np.arange(len(usable)); rng.shuffle(idx); n_train=int(round(train_ratio*len(usable)))
    return {'usable':usable,'train':[usable[i] for i in idx[:n_train]],'test':[usable[i] for i in idx[n_train:]]}
def pose_error_against_measurement(sample, dh_params, T_EE_CB, joint_zero_offsets=None):
    T_err=inv_T(fk_from_dh(sample.joints_rad,dh_params,joint_zero_offsets) @ T_EE_CB) @ sample.T_base_CB
    e_trans_m=float(np.linalg.norm(T_err[:3,3])); e_rot_rad=float(rot_angle(T_err[:3,:3]))
    return {'e_trans_m':e_trans_m,'e_trans_mm':e_trans_m*1000.0,'e_rot_rad':e_rot_rad,'e_rot_deg':math.degrees(e_rot_rad)}
def summarize_metric(x): return {'mean':float(np.mean(x)),'median':float(np.median(x)),'max':float(np.max(x)),'rmse':float(np.sqrt(np.mean(x*x)))}
def evaluate_set(samples, dh_params, T_EE_CB, joint_zero_offsets=None):
    per=[]
    for s in samples:
        err=pose_error_against_measurement(s,dh_params,T_EE_CB,joint_zero_offsets)
        per.append({'pose_id':s.pose_id,'image_pair':s.image_pair,'reproj_mean_rmse_px':float(s.reproj_mean_rmse_px),**err})
    e_trans_m=np.array([r['e_trans_m'] for r in per],dtype=float); e_trans_mm=np.array([r['e_trans_mm'] for r in per],dtype=float); e_rot_rad=np.array([r['e_rot_rad'] for r in per],dtype=float); e_rot_deg=np.array([r['e_rot_deg'] for r in per],dtype=float); reproj=np.array([r['reproj_mean_rmse_px'] for r in per],dtype=float)
    return {'N':int(len(per)),'e_trans_m':summarize_metric(e_trans_m),'e_trans_mm':summarize_metric(e_trans_mm),'e_rot_rad':summarize_metric(e_rot_rad),'e_rot_deg':summarize_metric(e_rot_deg),'reproj_mean_rmse_px':summarize_metric(reproj),'per_pose':per}
def percent_improvement(old_value,new_value): return 0.0 if abs(old_value)<1e-12 else float((old_value-new_value)/old_value*100.0)
def main():
    triang_json=load_json(PATHS['triangulated']); joints_json=load_json(PATHS['joints']); dh_nominal_json=load_json(PATHS['dh_nominal']); dh_calibrated_json=load_json(PATHS['dh_calibrated']); ee_cb_json=load_json(PATHS['ee_cb']); calib_report_json=load_json(PATHS['calib_report'])
    dh_nominal=dh_nominal_json['joints']; dh_calibrated=dh_calibrated_json['joints']; joint_zero_offsets_cal=np.array(dh_calibrated_json.get('joint_zero_offsets_rad',[0.0]*6),dtype=float); T_EE_CB=np.array(ee_cb_json['transformation_matrix_T_EE_CB_4x4'],dtype=float)
    settings=calib_report_json['settings']; reproj_threshold_px=float(settings['REPROJ_THRESHOLD_PX']); train_ratio=float(settings['TRAIN_RATIO']); seed=int(settings['SEED'])
    all_samples=build_samples(joints_json,triang_json); split=reconstruct_train_test_split(all_samples,reproj_threshold_px,train_ratio,seed); usable_samples, train_samples, test_samples = split['usable'], split['train'], split['test']
    nominal_test=evaluate_set(test_samples,dh_nominal,T_EE_CB,None); calibrated_test=evaluate_set(test_samples,dh_calibrated,T_EE_CB,joint_zero_offsets_cal); nominal_usable=evaluate_set(usable_samples,dh_nominal,T_EE_CB,None); calibrated_usable=evaluate_set(usable_samples,dh_calibrated,T_EE_CB,joint_zero_offsets_cal)
    improvement_test={'e_trans_rmse_mm_percent':percent_improvement(nominal_test['e_trans_mm']['rmse'], calibrated_test['e_trans_mm']['rmse']),'e_trans_median_mm_percent':percent_improvement(nominal_test['e_trans_mm']['median'], calibrated_test['e_trans_mm']['median']),'e_rot_rmse_deg_percent':percent_improvement(nominal_test['e_rot_deg']['rmse'], calibrated_test['e_rot_deg']['rmse']),'e_rot_median_deg_percent':percent_improvement(nominal_test['e_rot_deg']['median'], calibrated_test['e_rot_deg']['median'])}
    report={'validation_name':'UR3a_2cam_DH_validation_triangulation','purpose':'Validation of the calibrated DH table using held-out test poses by comparing triangulation-measured checkerboard pose against checkerboard pose predicted from forward kinematics.','paths_used':{k:str(v) for k,v in PATHS.items()},'dataset_info':{'num_total_matched_samples':int(len(all_samples)),'reproj_threshold_px':reproj_threshold_px,'num_usable_samples':int(len(usable_samples)),'train_ratio':train_ratio,'seed':seed,'num_train_samples':int(len(train_samples)),'num_test_samples':int(len(test_samples)),'train_pose_ids':[s.pose_id for s in train_samples],'test_pose_ids':[s.pose_id for s in test_samples]},'frame_notes':{'measurement_pose_in_base':'T_base_CB_meas comes directly from 2-camera triangulation followed by rigid alignment.','prediction_pose_in_base':'T_base_CB_pred = FK(q, DH) @ T_EE_CB','relative_error':'T_err = inv(T_base_CB_pred) @ T_base_CB_meas','validation_metrics':{'e_trans_m':'norm of translation component of T_err in meters','e_rot_rad':'rotation angle of rotation component of T_err in radians'}},'models':{'nominal':{'joint_zero_offsets_rad':[0.0]*6},'calibrated':{'joint_zero_offsets_rad':joint_zero_offsets_cal.tolist()}},'results':{'held_out_test_set':{'nominal':nominal_test,'calibrated':calibrated_test,'improvement_percent':improvement_test},'all_usable_poses_reference':{'nominal':nominal_usable,'calibrated':calibrated_usable}}}
    ensure_parent_dir(PATHS['output_json'])
    with open(PATHS['output_json'],'w',encoding='utf-8') as f: json.dump(report,f,indent=2)
    print(f"Validation JSON written to: {PATHS['output_json']}")
if __name__=='__main__': main()
