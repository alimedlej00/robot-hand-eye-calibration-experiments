from __future__ import annotations
import json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import cv2, numpy as np

INTRINSICS_CAM1 = Path(r"C:\Users\kaysa\OneDrive\Desktop\robot-hand-eye-calibration-experiments\02_intrinsic_calibration\cam1\04_results\intrinsics_cam1.json")
INTRINSICS_CAM2 = Path(r"C:\Users\kaysa\OneDrive\Desktop\robot-hand-eye-calibration-experiments\02_intrinsic_calibration\cam2\04_results\intrinsics_cam2.json")
CAM_BASE_JSON = Path(r"C:\Users\kaysa\OneDrive\Desktop\robot-hand-eye-calibration-experiments\03_extrinsic_calibration\shared_inputs\tf_base_to_camera.json")
DATASET_JSON = Path(r"C:\Users\kaysa\OneDrive\Desktop\robot-hand-eye-calibration-experiments\03_extrinsic_calibration\shared_inputs\dataset_ur3a_joint_images_40poses.json")
CAM1_IMAGES = Path(r"C:\Users\kaysa\OneDrive\Desktop\robot-hand-eye-calibration-experiments\03_extrinsic_calibration\images\cam1_images")
CAM2_IMAGES = Path(r"C:\Users\kaysa\OneDrive\Desktop\robot-hand-eye-calibration-experiments\03_extrinsic_calibration\images\cam2_images")
RESULTS_DIR = Path(r"C:\Users\kaysa\OneDrive\Desktop\robot-hand-eye-calibration-experiments\03_extrinsic_calibration\2cam_experiment\results\01_triangulation_2cam")
PATTERN_SIZE = (9, 7)

def load_json(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f: return json.load(f)
def ensure_dir(path: Path) -> None: path.mkdir(parents=True, exist_ok=True)
def inv_T(T: np.ndarray) -> np.ndarray:
    R,t=T[:3,:3],T[:3,3]; Ti=np.eye(4); Ti[:3,:3]=R.T; Ti[:3,3]=-R.T@t; return Ti
def make_object_points(cols:int, rows:int, s:float)->np.ndarray:
    obj=np.zeros((cols*rows,3),dtype=np.float64); xs,ys=np.meshgrid(np.arange(cols),np.arange(rows)); obj[:,0]=xs.reshape(-1)*s; obj[:,1]=ys.reshape(-1)*s; return obj
def reorder_corners_assume_image_topleft(corners: np.ndarray, cols: int, rows: int) -> np.ndarray:
    grid=corners.reshape(rows,cols,1,2)[:,:,0,:]; best=None; best_score=float('inf')
    for k in range(4):
        g=np.rot90(grid,k=k); g_rc=g if (g.shape[0]==rows and g.shape[1]==cols) else np.transpose(g,(1,0,2)); origin=g_rc[0,0]; score=float(origin[0]+origin[1])
        if score<best_score: best_score=score; best=g_rc.reshape(-1,1,2).astype(np.float32)
    return best
def generate_rotated_corner_orderings(corners: np.ndarray, cols: int, rows: int) -> List[np.ndarray]:
    grid=corners.reshape(rows,cols,1,2)[:,:,0,:]; outs=[]
    for k in range(4):
        g=np.rot90(grid,k=k); g_rc=g if (g.shape[0]==rows and g.shape[1]==cols) else np.transpose(g,(1,0,2)); outs.append(g_rc.reshape(-1,1,2).astype(np.float32))
    return outs
def rigid_alignment_board_to_base(obj_pts: np.ndarray, base_pts: np.ndarray) -> np.ndarray:
    A=obj_pts.astype(np.float64); B=base_pts.astype(np.float64); ca=A.mean(0); cb=B.mean(0); AA=A-ca; BB=B-cb; H=AA.T@BB; U,_,Vt=np.linalg.svd(H); R=Vt.T@U.T
    if np.linalg.det(R)<0: Vt[-1,:]*=-1.0; R=Vt.T@U.T
    t=cb-R@ca; T=np.eye(4); T[:3,:3]=R; T[:3,3]=t; return T
def project_board_points(obj_pts_board, T_base_CB, T_base_cam, K, dist):
    T_cam_CB=inv_T(T_base_cam)@T_base_CB; R=T_cam_CB[:3,:3]; t=T_cam_CB[:3,3]; rvec,_=cv2.Rodrigues(R); proj,_=cv2.projectPoints(obj_pts_board.astype(np.float64), rvec, t.reshape(3,1), K, dist); return proj.reshape(-1,2)
def rmse_pixels(p1,p2)->float:
    d=p1.reshape(-1,2)-p2.reshape(-1,2); return float(np.sqrt(np.mean(np.sum(d*d,axis=1))))
def scale_camera_matrix(K, calib_wh, actual_wh):
    cw,ch=calib_wh; aw,ah=actual_wh; sx=aw/float(cw); sy=ah/float(ch); K2=K.copy().astype(np.float64); K2[0,0]*=sx; K2[0,2]*=sx; K2[1,1]*=sy; K2[1,2]*=sy; return K2, {'sx':sx,'sy':sy}
def detect_corners(img_path: Path, pattern_size):
    img=cv2.imread(str(img_path),cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(f'Could not read image: {img_path}')
    h,w=img.shape[:2]; gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    found,corners=cv2.findChessboardCornersSB(gray,pattern_size,flags=cv2.CALIB_CB_EXHAUSTIVE|cv2.CALIB_CB_ACCURACY)
    if (not found) or corners is None or len(corners)!=pattern_size[0]*pattern_size[1]: raise RuntimeError(f'Checkerboard not detected in: {img_path.name}')
    return corners.astype(np.float32),(w,h)
def triangulate_points_base(corners1,corners2,K1,K2,dist1,dist2,T_base_cam1,T_base_cam2):
    P1=K1@inv_T(T_base_cam1)[:3,:]; P2=K2@inv_T(T_base_cam2)[:3,:]
    pts1=cv2.undistortPoints(corners1,K1,dist1,P=K1).reshape(-1,2); pts2=cv2.undistortPoints(corners2,K2,dist2,P=K2).reshape(-1,2)
    X4=cv2.triangulatePoints(P1,P2,pts1.T,pts2.T).T; return (X4[:,:3]/X4[:,3:4]).astype(np.float64)
@dataclass
class CameraInfo:
    camera_id:str; K_calib:np.ndarray; dist:np.ndarray; calib_wh:Tuple[int,int]; square_size_m:float; cols:int; rows:int
def load_intrinsics(path: Path)->CameraInfo:
    d=load_json(path); return CameraInfo(str(d.get('camera_id',path.stem)), np.array(d['intrinsics_K']['matrix_3x3'],dtype=np.float64), np.array(d['distortion']['coefficients'],dtype=np.float64).reshape(-1,1), (int(d['image_size_wh']['width']),int(d['image_size_wh']['height'])), float(d['square_size_m']), int(d['pattern_size_inner_corners']['cols']), int(d['pattern_size_inner_corners']['rows']))
def image_path_for_pose(cam_dir: Path, pose_id: int, cam_idx: int) -> Path: return cam_dir / f'img{pose_id}_cam{cam_idx}.png'

def main():
    ensure_dir(RESULTS_DIR)
    intr1, intr2 = load_intrinsics(INTRINSICS_CAM1), load_intrinsics(INTRINSICS_CAM2)
    dataset=load_json(DATASET_JSON); cam_base=load_json(CAM_BASE_JSON)
    T_base_cam1=np.array(cam_base['T_base_to_camera_extrinsics']['cam1']['T_4x4'],dtype=float)
    T_base_cam2=np.array(cam_base['T_base_to_camera_extrinsics']['cam2']['T_4x4'],dtype=float)
    obj_pts=make_object_points(PATTERN_SIZE[0],PATTERN_SIZE[1],intr1.square_size_m)
    poses_out=[]; joints_out=[]; successes=0; failures=0
    for pose in dataset['poses']:
        pid=int(pose['pose_id']); img1=image_path_for_pose(CAM1_IMAGES,pid,1); img2=image_path_for_pose(CAM2_IMAGES,pid,2)
        entry={'pose_id':pid,'image_cam1':img1.name,'image_cam2':img2.name,'success':False}
        try:
            c1raw, wh1=detect_corners(img1,PATTERN_SIZE); c2raw, wh2=detect_corners(img2,PATTERN_SIZE)
            K1, s1=scale_camera_matrix(intr1.K_calib,intr1.calib_wh,wh1); K2, s2=scale_camera_matrix(intr2.K_calib,intr2.calib_wh,wh2)
            c1=reorder_corners_assume_image_topleft(c1raw,*PATTERN_SIZE); cand2=generate_rotated_corner_orderings(c2raw,*PATTERN_SIZE)
            best=None
            for c2 in cand2:
                X=triangulate_points_base(c1,c2,K1,K2,intr1.dist,intr2.dist,T_base_cam1,T_base_cam2)
                T_base_CB=rigid_alignment_board_to_base(obj_pts,X)
                err1=rmse_pixels(project_board_points(obj_pts,T_base_CB,T_base_cam1,K1,intr1.dist), c1.reshape(-1,2))
                err2=rmse_pixels(project_board_points(obj_pts,T_base_CB,T_base_cam2,K2,intr2.dist), c2.reshape(-1,2))
                total=0.5*(err1+err2)
                if best is None or total<best['total']: best={'c2':c2,'X':X,'T_base_CB':T_base_CB,'err1':err1,'err2':err2,'total':total}
            T_base_CB=best['T_base_CB']; T_cam1_CB=inv_T(T_base_cam1)@T_base_CB; T_cam2_CB=inv_T(T_base_cam2)@T_base_CB; rvec1,_=cv2.Rodrigues(T_cam1_CB[:3,:3])
            entry.update({'success':True,'reproj_cam1_rmse_px':float(best['err1']),'reproj_cam2_rmse_px':float(best['err2']),'reproj_mean_rmse_px':float(best['total']),'T_base_CB_4x4':T_base_CB.tolist(),'T_cam1_CB_4x4':T_cam1_CB.tolist(),'T_cam2_CB_4x4':T_cam2_CB.tolist(),'triangulated_points_base_3d':best['X'].tolist(),'rvec_cam1_rad':rvec1.reshape(3).tolist(),'tvec_cam1_m':T_cam1_CB[:3,3].tolist(),'camera_matrix_used_cam1_3x3':K1.tolist(),'camera_matrix_used_cam2_3x3':K2.tolist(),'intrinsics_scale_cam1':s1,'intrinsics_scale_cam2':s2})
            joints_out.append({'pose_id':pid,'image_pair':[img1.name,img2.name],'joints_deg':[float(pose['joint_angles_deg'][f'joint_{i}']) for i in range(1,7)],'joints_rad':[math.radians(float(pose['joint_angles_deg'][f'joint_{i}'])) for i in range(1,7)]})
            successes+=1; print(f'pose {pid:02d}: OK')
        except Exception as e:
            entry['error']=str(e); failures+=1; print(f'pose {pid:02d}: FAIL -> {e}')
        poses_out.append(entry)
    out={'configuration':'2cam','cameras_used':['cam1','cam2'],'method':'triangulation_followed_by_rigid_alignment','image_directories':{'cam1':str(CAM1_IMAGES),'cam2':str(CAM2_IMAGES)},'pattern':{'cols':PATTERN_SIZE[0],'rows':PATTERN_SIZE[1],'square_size_m':intr1.square_size_m},'pose_definition':'Main saved measurement is T_base_CB_4x4 estimated from multi-view triangulation and rigid alignment. T_cam1_CB_4x4 and T_cam2_CB_4x4 are derived via known camera extrinsics.','origin_convention':'Cam1 uses the validated 1-camera board-origin convention (image-top-left logical rotation). Cam2 ordering is selected among 4 logical rotations by minimum total reprojection error after triangulation + rigid alignment.','summary':{'num_images_total':len(dataset['poses']),'num_success':successes,'num_failed':failures},'poses':poses_out}
    joints={'robot':'UR3a','units':{'joint_angles':'rad'},'source_dataset':str(DATASET_JSON),'cameras_used':['cam1','cam2'],'poses':joints_out}
    out_path=RESULTS_DIR/'triangulated_checkerboard_poses_2cam.json'; joints_path=RESULTS_DIR/'ur3a_joints_data_2cam.json'
    with open(out_path,'w',encoding='utf-8') as f: json.dump(out,f,indent=2)
    with open(joints_path,'w',encoding='utf-8') as f: json.dump(joints,f,indent=2)
    print('\nDone.'); print(f'Saved triangulation JSON: {out_path}'); print(f'Saved joints JSON: {joints_path}'); print(f'Successes: {successes}/{len(dataset["poses"])}')
if __name__=='__main__': main()
