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
from debug.rendering import init_mesh_viewer, extract_from_xml, world2im, triangulate, plot3d

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
    
    mv = init_mesh_viewer(camera_path = os.path.join('virtual_cameras'))  

    
    body_pose_world = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body','root_orient','trans']})
    positions_gt = body_pose_world.Jtr[frame_to_render,:22].cpu().numpy()
    njoints = positions_gt.shape[0]

    # 
    # --- Get 2D projection in cams ---
    #
    camera_files = [c for c in glob.glob(os.path.join("virtual_cameras", "*.xml")) if os.path.basename(c) in available_cameras]

    keypoints_im = np.empty((len(camera_files), njoints, 2))
    camera_poses = []
    Ks = []
    for cam_id, camera_file in enumerate(camera_files):
        #get camera param for projection
        K, camera_pose, size = extract_from_xml(camera_file)
        camera_pose = np.vstack((camera_pose, [0, 0, 0, 1]))
        camera_poses.append(camera_pose)
        Ks.append(K)

        mv.updateCam(camera_pose, K)

        #get 2D projection in image plane 
        keypoints_im_in_cam = world2im(positions_gt, camera_pose, K, size)
        
        keypoints_im[cam_id] = keypoints_im_in_cam  

    positions_triang = triangulate(
        Ks=Ks, 
        camera_poses=camera_poses, 
        yolo_keypoints=keypoints_im, 
        image_sizes=size,
        joint_id=0,
        mv=mv
    )
    positions_triang = np.vstack(positions_triang)

    plot3d(positions_gt=positions_gt, positions_triang=positions_triang)
    
    #
    # --- Triangulation ---
    #
    
if __name__ == '__main__':
    __main()