import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
import torch
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pyrender
import trimesh
import glob
from PIL import Image
#Interal
from human_body_prior.body_model.body_model import BodyModel
from body_visualizer.tools.vis_tools import colors
from debug.rendering import init_mesh_viewer, extract_from_xml, world2im, triangulate, plot3d
from debug.yolo_utils import init_yolo, run_yolo

# Overwrite log file every time the script runs
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def __main():

    # 
    # --- Init pathes ---
    # 
    dataroot = os.path.join('..', 'AGRoL', 'amass')
    dataroot_subset = 'BioMotionLab_NTroje'
    phase = 'train'
    split_file = os.path.join("..", "OpenMPLPoser", "data", "data_split", dataroot_subset, phase+"_split.txt")
    support_dir = os.path.join('..', "AGRoL", "support_data")
    available_cameras = ['Camera_1.xml','Camera_2.xml','Camera_3.xml']

    # 
    # --- init parameters ---
    # 
    num_betas = 16
    num_dmpls = 8
    file_id = 12
    frame_to_render = 0
    yolo_model = 'yolov8n-pose'
    #
    # --- Init device ---
    #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)


    # 
    # --- Init YOLO ---
    yolo_model = init_yolo(f'{yolo_model}.pt',)
    
    #
    # --- Get motion ---
    #  
    with open(split_file, 'r') as f:
        filepaths = [line.rstrip('\n') for line in f]

    bm_fname_male = os.path.join(support_dir, 'body_models/smplh/male/model.npz')
    dmpl_fname_male = os.path.join(support_dir, 'body_models/dmpls/male/model.npz')

    bm_fname_female = os.path.join(support_dir, 'body_models/smplh/female/model.npz')
    dmpl_fname_female = os.path.join(support_dir, 'body_models/dmpls/female/model.npz')

    bm_male = BodyModel(bm_fname=bm_fname_male, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname_male).to(device)
    bm_female = BodyModel(bm_fname=bm_fname_female, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname_female).to(device)

    assert 0 <= file_id < len(filepaths), "Invalid value for file_id" 
    filepath = filepaths[file_id]
    bdata = np.load(os.path.join(dataroot, filepath), allow_pickle=True)
    framerate = bdata["mocap_framerate"]

    if framerate == 120:
        stride = 2
    elif framerate == 60:
        stride = 1

    bdata_poses = bdata["poses"][::stride, ...]
    bdata_trans = bdata["trans"][::stride, ...]
    subject_gender = bdata["gender"]

    bm = bm_male if subject_gender == 'male' else bm_female

    body_parms = {
        'root_orient': torch.tensor(bdata_poses[:, :3], device=device, dtype=torch.float32),
        'pose_body': torch.tensor(bdata_poses[:, 3:66], device=device, dtype=torch.float32),
        'trans': torch.tensor(bdata_trans, device=device, dtype=torch.float32),
    }

    
    body_pose_world = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body','root_orient','trans']})
    #ground truth in coco format
    jreg_path = os.path.join('data', 'J_regressor_coco.npy') 
    jregressor = torch.tensor(np.load(jreg_path), dtype=body_pose_world.v.dtype, device=device)
    joints_coco = torch.einsum('bik,ji->bjk', [body_pose_world.v, jregressor])
    positions_gt = joints_coco[frame_to_render].cpu().numpy()
    
    camera_files = [c for c in glob.glob(os.path.join("data", "virtual_cameras", "*.xml")) if os.path.basename(c) in available_cameras]
    
    mv = init_mesh_viewer(camera_path = os.path.join("data", 'virtual_cameras'))  

    camera_poses = []
    Ks = []

    for camera_file in camera_files:
        #get camera param for projection
        K, camera_pose, size = extract_from_xml(camera_file)
        camera_pose = np.vstack((camera_pose, [0, 0, 0, 1]))
        camera_poses.append(camera_pose)
        Ks.append(K)

    vcam_kp = run_yolo(
            yolo_model=yolo_model, 
            mv=mv, 
            bm=bm, 
            body_pose_world=body_pose_world,
            frame_to_render=frame_to_render,
            nb_frames=1,
            camera_files=camera_files
        )

    vcam_kp = np.vstack(vcam_kp)
    
    positions_triang = triangulate(
        Ks=Ks, 
        camera_poses=camera_poses, 
        yolo_keypoints=vcam_kp, 
        image_sizes=size,
        mv=mv
    )

    positions_triang = np.vstack(positions_triang)

    plot3d(positions_gt=positions_gt, positions_triang=positions_triang)

if __name__ == '__main__':
    __main()