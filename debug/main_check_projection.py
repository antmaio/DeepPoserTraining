import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
import torch
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
from debug.rendering import init_mesh_viewer, extract_from_xml, render

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
    cam_id = 0
    
    #
    # --- Init device ---
    #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)


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

    #
    # --- Rendering ---
    #
    camera_files = [c for c in glob.glob(os.path.join("virtual_cameras", "*.xml")) if os.path.basename(c) in available_cameras]
    assert 0 <= cam_id < len(camera_files), "invalid value for cam_id"

    mv = init_mesh_viewer(camera_path = os.path.join('virtual_cameras'))  

    #get camera param for projection
    K, camera_pose, size = extract_from_xml(camera_files[cam_id])
    projection_matrix = K @ camera_pose
    camera_pose = np.vstack((camera_pose, [0, 0, 0, 1]))
    mv.updateCam(camera_pose, K)

    verts = body_pose_world.v[frame_to_render].cpu().numpy()
    faces = bm.f.cpu().numpy()
    joints = body_pose_world.Jtr[frame_to_render].cpu().numpy()

    render(mv=mv, verts=verts, faces=faces, savepath='temp.png', keypoints=joints, camera_pose=camera_pose, K=K, size=size)


if __name__ == '__main__':
    __main()