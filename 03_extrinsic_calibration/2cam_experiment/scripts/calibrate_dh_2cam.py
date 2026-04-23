from __future__ import annotations
import json, math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

ROOT = Path(r"C:\Users\kaysa\OneDrive\Desktop\robot-hand-eye-calibration-experiments\03_extrinsic_calibration")
PATHS = {
    'triangulated': ROOT/'2cam_experiment'/'results'/'01_triangulation_2cam'/'triangulated_checkerboard_poses_2cam.json',
    'joints': ROOT/'2cam_experiment'/'results'/'01_triangulation_2cam'/'ur3a_joints_data_2cam.json',
    'dh': ROOT/'shared_inputs'/'robot_model_dh_nominal_ur3a.json',
    'ee_cb': ROOT/'shared_inputs'/'tf_ee_to_cb.json',
    'results_dir': ROOT/'2cam_experiment'/'results'/'02_calibration_2cam',
}
def load_json(path: Path)->dict:
    with open(path,'r',encoding='utf-8') as f: return json.load(f)
def ensure_dir(path: Path)->None: path.mkdir(parents=True, exist_ok=True)
def inv_T(T): R,t=T[:3,:3],T[:3,3]; Ti=np.eye(4); Ti[:3,:3]=R.T; Ti[:3,3]=-R.T@t; return Ti
def hat(w): wx,wy,wz=w; return np.array([[0,-wz,wy],[wz,0,-wx],[-wy,wx,0]],dtype=float)
def rodrigues(w):
    theta=float(np.linalg.norm(w))
    if theta<1e-12:
        return np.eye(3)+hat(w)
    k=w/theta; K=hat(k)
    return np.eye(3)+math.sin(theta)*K+(1-math.cos(theta))*(K@K)
def rot_log(R):
    tr=float(np.trace(R)); c=max(-1.0,min(1.0,(tr-1.0)/2.0)); theta=math.acos(c)
    if theta<1e-12:
        return np.zeros(3)
    w_hat=(R-R.T)*(0.5/math.sin(theta))
    return theta*np.array([w_hat[2,1],w_hat[0,2],w_hat[1,0]],dtype=float)
def rot_angle(R): tr=float(np.trace(R)); c=max(-1.0,min(1.0,(tr-1.0)/2.0)); return math.acos(c)
def dh_A(a,alpha,d,theta):
    ca,sa=math.cos(alpha),math.sin(alpha); ct,st=math.cos(theta),math.sin(theta)
    return np.array([[ct,-st*ca,st*sa,a*ct],[st,ct*ca,-ct*sa,a*st],[0.0,sa,ca,d],[0.0,0.0,0.0,1.0]],dtype=float)
def fk_from_dh(joints_rad, dh_params, joint_zero_offsets=None):
    if joint_zero_offsets is None: joint_zero_offsets=np.zeros(6)
    T=np.eye(4)
    for q,off,p in zip(joints_rad,joint_zero_offsets,dh_params):
        T = T @ dh_A(float(p['a']), float(p['alpha']), float(p['d']), float(q+off+p.get('theta_offset',0.0)))
    return T
@dataclass
class PoseSample:
    pose_id:int; image_pair:list; joints_rad:np.ndarray; T_base_CB:np.ndarray; reproj_mean_rmse_px:float
def build_samples(joints_json, triang_json):
    j_by={int(p['pose_id']):p for p in joints_json['poses']}; t_by={int(p['pose_id']):p for p in triang_json['poses']}; out=[]
    for pid in sorted(set(j_by)&set(t_by)):
        tj=t_by[pid]
        if not tj.get('success',True): continue
        jj=j_by[pid]; out.append(PoseSample(pid, list(jj['image_pair']), np.array(jj['joints_rad'],dtype=float), np.array(tj['T_base_CB_4x4'],dtype=float), float(tj.get('reproj_mean_rmse_px',float('nan')))))
    return out
@dataclass
class ErrorRow:
    pose_id:int; image_pair:list; reproj_mean_rmse_px:float; t_err_m:float; ang_err_rad:float
def evaluate(samples, dh_params, T_EE_CB, reproj_threshold_px=None, joint_zero_offsets=None):
    rows=[]
    for s in samples:
        if reproj_threshold_px is not None and s.reproj_mean_rmse_px>reproj_threshold_px: continue
        T_pred=fk_from_dh(s.joints_rad, dh_params, joint_zero_offsets) @ T_EE_CB; T_err=inv_T(T_pred) @ s.T_base_CB
        rows.append(ErrorRow(s.pose_id, s.image_pair, s.reproj_mean_rmse_px, float(np.linalg.norm(T_err[:3,3])), float(rot_angle(T_err[:3,:3]))))
    return rows
def summarize(rows):
    t=np.array([r.t_err_m for r in rows],dtype=float); a=np.array([r.ang_err_rad for r in rows],dtype=float); rm=np.array([r.reproj_mean_rmse_px for r in rows],dtype=float)
    stats=lambda x:{'mean':float(np.mean(x)),'median':float(np.median(x)),'max':float(np.max(x)),'rmse':float(np.sqrt(np.mean(x*x)))}
    return {'N':len(rows),'t_err_m':stats(t),'t_err_mm':stats(t*1000.0),'ang_err_rad':stats(a),'ang_err_deg':stats(a*180.0/math.pi),'reproj_mean_rmse_px':stats(rm)}
def scalar_cost(summary): return float(summary['t_err_mm']['rmse'] + 10.0*summary['ang_err_deg']['rmse'])
def apply_dh_deltas(dh_nominal, delta_a, delta_d):
    out=[]
    for i,p in enumerate(dh_nominal):
        q=dict(p); q['a']=float(p['a']+delta_a[i]); q['d']=float(p['d']+delta_d[i]); out.append(q)
    return out
def calibrate_dh(train,test,dh_nominal,T_EE_CB,lever_arm_m=0.15,joff_bound_deg=1.0,dh_bound_mm=2.0,reproj_threshold_px=2.0):
    from scipy.optimize import least_squares
    train_u=[s for s in train if s.reproj_mean_rmse_px<=reproj_threshold_px]; test_u=[s for s in test if s.reproj_mean_rmse_px<=reproj_threshold_px]
    if len(train_u)<10: raise RuntimeError('Not enough usable TRAIN samples.')
    pack=lambda j,a,d: np.concatenate([j,a,d],axis=0)
    unpack=lambda x:(x[0:6],x[6:12],x[12:18])
    def residuals(x,data):
        j_off,delta_a,delta_d=unpack(x); dh_cur=apply_dh_deltas(dh_nominal,delta_a,delta_d); out=[]
        for s in data:
            T_pred=fk_from_dh(s.joints_rad, dh_cur, j_off) @ T_EE_CB; T_err=inv_T(T_pred) @ s.T_base_CB
            out.append(np.concatenate([T_err[:3,3], lever_arm_m*rot_log(T_err[:3,:3])],axis=0))
        return np.concatenate(out,axis=0)
    x0=pack(np.zeros(6),np.zeros(6),np.zeros(6)); joff_b=math.radians(joff_bound_deg); dh_b=dh_bound_mm/1000.0
    lb=np.concatenate([-np.ones(6)*joff_b,-np.ones(6)*dh_b,-np.ones(6)*dh_b]); ub=np.concatenate([np.ones(6)*joff_b,np.ones(6)*dh_b,np.ones(6)*dh_b])
    sol=least_squares(lambda x: residuals(x,train_u), x0, bounds=(lb,ub), verbose=0, max_nfev=400)
    j_off,delta_a,delta_d=unpack(sol.x); dh_cal=apply_dh_deltas(dh_nominal,delta_a,delta_d)
    train_summary=summarize(evaluate(train_u,dh_cal,T_EE_CB,None,j_off)); test_summary=summarize(evaluate(test_u,dh_cal,T_EE_CB,None,j_off))
    return {'success':bool(sol.success),'status':int(sol.status),'message':str(sol.message),'cost':float(sol.cost),'nfev':int(sol.nfev),'joint_zero_offsets_rad':j_off,'delta_a_m':delta_a,'delta_d_m':delta_d,'dh_calibrated':dh_cal,'train_summary':train_summary,'test_summary':test_summary,'E_test':scalar_cost(test_summary)}
def main():
    REPROJ_THRESHOLD_PX=6.0; TRAIN_RATIO=0.7; SEED=7; JOINT_OFF_BOUND_DEG=1.0; DH_BOUND_MM=2.0; LEVER_ARM_M=0.15
    ensure_dir(PATHS['results_dir'])
    triang_json=load_json(PATHS['triangulated']); joints_json=load_json(PATHS['joints']); dh_json=load_json(PATHS['dh']); ee_cb_json=load_json(PATHS['ee_cb'])
    samples=build_samples(joints_json,triang_json); print(f'Loaded matched successful 2-cam samples: {len(samples)}')
    dh_nominal=dh_json['joints']; T_EE_CB=np.array(ee_cb_json['transformation_matrix_T_EE_CB_4x4'],dtype=float)
    usable=[s for s in samples if (not math.isnan(s.reproj_mean_rmse_px)) and s.reproj_mean_rmse_px<=REPROJ_THRESHOLD_PX]
    if len(usable)<12: raise RuntimeError('Not enough usable samples after triangulation reprojection filtering.')
    rng=np.random.default_rng(SEED); idx=np.arange(len(usable)); rng.shuffle(idx); n_train=int(round(TRAIN_RATIO*len(usable))); train=[usable[i] for i in idx[:n_train]]; test=[usable[i] for i in idx[n_train:]]
    print(f'Usable poses (triang reproj <= {REPROJ_THRESHOLD_PX} px): {len(usable)}'); print(f'Train: {len(train)} | Test: {len(test)}')
    summ_nom_test=summarize(evaluate(test,dh_nominal,T_EE_CB,None,None)); print('\n=== NOMINAL MODEL (2-CAM TEST) ==='); print(json.dumps(summ_nom_test,indent=2)); print('E_nominal_test =', scalar_cost(summ_nom_test))
    best=calibrate_dh(train,test,dh_nominal,T_EE_CB,LEVER_ARM_M,JOINT_OFF_BOUND_DEG,DH_BOUND_MM,REPROJ_THRESHOLD_PX)
    print('\n=== BEST CALIBRATION RESULT (2-CAM TEST) ==='); print('E_best_test =', best['E_test']); print(json.dumps(best['test_summary'],indent=2)); print('\nJoint zero offsets (deg):'); print(best['joint_zero_offsets_rad']*180.0/math.pi)
    dh_out={'meta':{'robot':dh_json.get('robot_name',dh_json.get('robot','UR3a')),'dh_convention':dh_json.get('dh_convention','standard'),'units':dh_json.get('units',{'length':'m','angle':'rad'}),'source':'calibrated_from_vision_2cam_triangulation'},'joints':best['dh_calibrated'],'joint_zero_offsets_rad':best['joint_zero_offsets_rad'].tolist()}
    report={'settings':{'REPROJ_THRESHOLD_PX':REPROJ_THRESHOLD_PX,'TRAIN_RATIO':TRAIN_RATIO,'SEED':SEED,'JOINT_OFF_BOUND_DEG':JOINT_OFF_BOUND_DEG,'DH_BOUND_MM':DH_BOUND_MM,'LEVER_ARM_M':LEVER_ARM_M},'nominal_test_summary':summ_nom_test,'best_train_summary':best['train_summary'],'best_test_summary':best['test_summary'],'E_nominal_test':scalar_cost(summ_nom_test),'E_best_test':best['E_test'],'joint_zero_offsets_rad':best['joint_zero_offsets_rad'].tolist(),'delta_a_m':best['delta_a_m'].tolist(),'delta_d_m':best['delta_d_m'].tolist()}
    dh_out_path=PATHS['results_dir']/'dh_calibrated_2cam.json'; report_path=PATHS['results_dir']/'calibration_report_2cam.json'
    with open(dh_out_path,'w',encoding='utf-8') as f: json.dump(dh_out,f,indent=2)
    with open(report_path,'w',encoding='utf-8') as f: json.dump(report,f,indent=2)
    print(f'\nWrote {dh_out_path}'); print(f'Wrote {report_path}')
if __name__=='__main__': main()
