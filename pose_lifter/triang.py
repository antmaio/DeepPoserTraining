#External
import argparse
import pickle
import glob 
import torch 
import re
from typing import List, Tuple
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyrender
from pyrender import Viewer
import logging
import copy
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'  # Optional: controls time format
)
#Internal
from data.camera_config import CAMERA_PATH_PROTOCOL_1, CAMERA_PATH_PROTOCOL_2, CAMERA_PATH_PROTOCOL_3
from data.yolo_data_gen import extract_from_xml
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.tools.vis_tools import colors

class MeshViewer2(MeshViewer):

    def __init__(self, width=1200, height=800, intrinsic=np.array([[1, 0, 0], [0, 1, 0]]), use_offscreen=True):
        # super().__init__()

        self.width, self.height = width, height
        self.use_offscreen = use_offscreen
        self.render_wireframe = False

        self.mat_constructor = pyrender.MetallicRoughnessMaterial
        self.trimesh_to_pymesh = pyrender.Mesh.from_trimesh

        self.scene = pyrender.Scene(bg_color=colors['white'], ambient_light=(0.3, 0.3, 0.3))

        # pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=float(width) / height)
        K = intrinsic
        pc = pyrender.IntrinsicsCamera(intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2])
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 0, 3.0])
        self.camera_node = self.scene.add(pc, pose=camera_pose, name='pc-camera')

        self.figsize = (width, height)

        if self.use_offscreen:
            self.viewer = pyrender.OffscreenRenderer(*self.figsize)
            self.use_raymond_lighting(4.)
        else:
            self.viewer = Viewer(self.scene, use_raymond_lighting=True, viewport_size=self.figsize, cull_faces=False,
                                 run_in_thread=True)

    def updateCam(self, camera_pose, intrinsic):
        self.camera_node.camera.fx = intrinsic[0, 0]
        self.camera_node.camera.fy = intrinsic[1, 1]
        self.camera_node.camera.cx = intrinsic[0, 2]
        self.camera_node.camera.cy = intrinsic[1, 2]
        self.scene.set_pose(self.camera_node, pose=camera_pose)

    def get_camera(self):
        return self.camera_node.camera

    def get_projection_matrix(self):
        return self.camera_node.camera.get_projection_matrix(width=self.width, height=self.height)

    def remove_mesh(self):
        for node in self.scene.get_nodes():
            if node.name is not None and 'mesh' in node.name:
                self.scene.remove_node(node)

def animate_2d_keypoints(yolo_keypoints: dict, 
                        cam_keys_for_triang: List[str], 
                        output_dir: str = "./",
                        fps: int = 60,
                        point_size: int = 30,
                        point_color: str = 'red'):
    """
    Save scatter-only animations of 2D keypoints as AVI files
    
    Args:
        yolo_keypoints: Dictionary with camera keys and keypoints (nframes, njoints, 2)
        cam_keys_for_triang: List of camera keys to process
        output_dir: Directory to save AVI files
        fps: Frames per second for output video
        point_size: Size of scatter points
        point_color: Color of scatter points
    """
    
    for cam_key in cam_keys_for_triang:
        if cam_key not in yolo_keypoints:
            continue
            
        points2d = yolo_keypoints[cam_key].cpu().numpy()
        nframes, njoints, _ = points2d.shape
        
        # Create figure with dark background
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.set_title(f'Camera: {cam_key}', color='white')
        
        # Set bounds with padding
        x_min, x_max = np.min(points2d[..., 0]), np.max(points2d[..., 0])
        y_min, y_max = np.min(points2d[..., 1]), np.max(points2d[..., 1])
        padding = max((x_max - x_min) * 0.1, (y_max - y_min) * 0.1, 50)
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.invert_yaxis()  # Match image coordinates
        
        # Clean up axes
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Initialize scatter plot
        scatter = ax.scatter([], [], c=point_color, s=point_size, alpha=0.8)
        
        # Animation update function
        def update(frame):
            points = points2d[frame]
            scatter.set_offsets(points)
            return [scatter]
        
        # Create animation
        anim = FuncAnimation(
            fig,
            update,
            frames=nframes,
            interval=1000/fps,  # Convert fps to interval
            blit=True
        )
        
        # Save as AVI file
        output_path = f"{output_dir}/{cam_key}_2d_keypoints.avi"
        
        # Using 'ffmpeg' writer with AVI-specific codec
        writer = 'ffmpeg'
        codec = 'mpeg4'  # Common AVI codec
        anim.save(output_path,
                 writer=writer,
                 codec=codec,
                 fps=fps,
                 dpi=100,
                 bitrate=2000,
                 extra_args=['-vcodec', codec])
        
        print(f"Saved AVI animation for {cam_key} to {output_path}")
        
        # Clean up
        plt.close(fig)

        # Example usage:
        # animate_2d_keypoints(yolo_keypoints, 
        #                     ['cam1', 'cam2'], 
        #                     output_dir='animations',
        #                     fps=30,
        #                     point_size=40,
        #                     point_color='cyan')

def set_mesh_viewer(camera_files:List[str]) -> MeshViewer2:
    # all cameras share the same intrinsic
    K, _, (image_width, image_height) = extract_from_xml(camera_files[0])
    mv = MeshViewer2(width=image_width, height=image_height, intrinsic=K, use_offscreen=True)
    return mv

def triangulate(
    Ks:np.ndarray, 
    camera_poses:np.ndarray, 
    image_sizes:List[Tuple[int, int]], 
    yolo_keypoints:dict, 
    cam_keys_for_triang:List[str],
    frame_id:int,
    joint_id:int,
    mv:MeshViewer2
):
    
    """
    Triangulate a 3D point from 2D keypoints observed by multiple cameras.

    Parameters
    ----------
    Ks : np.ndarray
        Array of camera intrinsic matrices with shape (n_cams, 3, 3)
    camera_poses : np.ndarray
        Array of camera extrinsic matrices with shape (n_cams, 4, 4)
    image_sizes : List[Tuple[int, int]]
        List of (width, height) tuples for each camera
    yolo_keypoints : torch.Tensor
        Dictionary-like object containing 2D keypoints for each camera
        with keys like 'vcam0' and values of shape (nframes, njoints, 2)
    cam_keys_for_triang : List[str]
        List of camera keys to use for triangulation (e.g., ['vcam0', 'vcam1'])
    frame_id : int
        Frame index to triangulate
    joint_id : int
        Joint index to triangulate

    Returns
    -------
    np.ndarray
        Triangulated 3D point as a numpy array of shape (3,)

    Notes
    -----
    - Currently only supports triangulation from exactly 2 camera views
    - Normalizes 2D points to [0,1] range using image dimensions
    - Uses OpenCV's triangulatePoints function
    """

    """ Method 1 """
    '''
    proj_matrices, points2d = [], []

    for cam_id, cam_key in enumerate(cam_keys_for_triang):
        image_width, image_height = image_sizes[cam_id]
        point2d = yolo_keypoints[cam_key][frame_id, joint_id].cpu().numpy()
        point2d_normalized = np.array([
            point2d[0] / image_width,
            point2d[1] / image_height
        ], dtype=np.float32)
        points2d.append(point2d_normalized)

        #Get projection matrix (K @ [R|t])
        K = Ks[cam_id]
        R = camera_poses[cam_id][:3,:3]
        t = camera_poses[cam_id][:3, 3]
        P = K @ np.hstack((R, t.reshape(3, 1)))
        proj_matrices.append(P)

    #Triangulate
    points2d = np.array(points2d, dtype=np.float32).T
    points3d_hom = cv2.triangulatePoints(
        proj_matrices[0], proj_matrices[1],
        points2d[:, 0:1], points2d[:, 1:2]
    )

    points3d = points3d_hom[:3] / points3d_hom[3].flatten()
    '''
    proj_matrices, points2d = [], []
    KMats, proj_matrices = [], []

    for cam_id, cam_key in enumerate(cam_keys_for_triang):
        image_width, image_height = image_sizes[cam_id]
        point2d = yolo_keypoints[cam_key][frame_id, joint_id].cpu().numpy()
        points2d.append(point2d)

        #get camera parameters
        camera_pose = camera_poses[cam_id]
        mv.updateCam(camera_pose, Ks[cam_id])

        KMats.append(mv.viewer._renderer._get_camera_matrices(mv.scene))
        proj_matrices.append(mv.get_camera().get_projection_matrix(image_width, image_height))
        #from homogeneous to 3D

    mat0, mat1 = (KMats[0][1] @ KMats[0][0])[:3,:], (KMats[1][1] @ KMats[1][0])[:3,:]
    #pmat0, pmat1 = proj_matrices[0][:3], proj_matrices[1][:3]

    kp0, kp1 = np.transpose(copy.copy(points2d[0])), np.transpose(copy.copy(points2d[1]))
    # keypoints at cam 0
    kp0[0] = (kp0[0] - (image_width/2)) / (image_width/2)
    kp0[1] = ((image_height / 2) - kp0[1]) / (image_height/2)
    # keypoints at cam 1
    kp1[0] = (kp1[0] - (image_width/2)) / (image_width/2)
    kp1[1] = ((image_height / 2) - kp1[1]) / (image_height/2)

    #triangulation
    points3d = cv2.triangulatePoints(mat0,mat1,kp0,kp1)
    points3d = points3d[:3] / points3d[3] #homogeneous to heterogeneous
    return points3d.squeeze()

def get_cam_params_from_key(key: List[str], camera_files:List[str])->Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extract camera parameters from XML files based on camera keys.

    Parameters
    ----------
    key : List[str]
        List of camera keys (e.g., ['vcam0', 'vcam1'])
    camera_files : List[str]
        List of paths to camera XML configuration files

    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]
        A tuple containing:
        - List of intrinsic matrices (3x3 numpy arrays)
        - List of extrinsic matrices (4x4 numpy arrays)
        - List of image sizes as (width, height) tuples

    Raises
    ------
    ValueError
        If no number can be extracted from a camera key
    TypeError
        If input is not a list of strings
    """
    Ks, camera_poses, image_sizes = [],[],[]
    def extract_number(s: str) -> int:
        match = re.search(r'\d+', s)
        if match:
            return int(match.group())
        raise ValueError(f"No number found in string: {s}")
    
    if isinstance(key, list):
        kIds = list(map(extract_number, key))  # Using map
    else:
        raise TypeError("Input must be a string or list of string")
    
    for kId in kIds:
        K, camera_pose, image_size = extract_from_xml(camera_files[kId])
        camera_pose = np.vstack((camera_pose, [0, 0, 0, 1]))
        Ks.append(K)
        camera_poses.append(camera_pose)
        image_sizes.append(image_size)
    return Ks, camera_poses, image_sizes

def check_if_shape_matches(points3d, pkl_filename:str):
    with open(pkl_filename, 'rb') as f:
        data = pickle.load(f)
    key = 'yolo_keypoints' if any('yolo' in _key for _key in data.keys()) else 'pose_estimation_keypoints'
    gt = data[key]['ground_truth']

    assert gt.shape == points3d.shape, \
        f"points3d and gt shapes do not match {points3d.shape} | {gt.shape}"

def __main():
    """
    Main execution function for multi-view 3D pose triangulation.

    Processes YOLO keypoints data and performs triangulation using:
    - Camera parameters from XML files
    - 2D keypoints from pickle files
    - Confidence scores to select best camera pairs

    Command-line Arguments
    ---------------------
    --dataset_type : str
        Dataset split identifier (default: 'amass_p1')
    --dataroot : str
        Path to directory containing pickle files (default: './data/keypoints/yolov8n-pose_protocol_1')

    Outputs
    -------
    points3d : np.ndarray
        Array of triangulated 3D points with shape (nframes, njoints, 3)
    """
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset_type', default='amass_p1', type=str, help="Dataset split as in AvatarJLM")
    parse.add_argument('--dataroot', default='./data/keypoints/yolov8n-pose_protocol_1', type=str, help='Path to pkl files')
    parse.add_argument('--output_dir', default='triang', type=str, help='relative output path to 3D keypoints')
    args = parse.parse_args()
    
    assert args.dataset_type in ('amass_p1', 'amass_p2', 'amass_p3'), f"{args.dataset_type} not supported for --dataset_type"

    if args.dataset_type in ('amass_p1', 'amass_p2'):
        camera_path = CAMERA_PATH_PROTOCOL_1
    else:  # args.dataset_type == 'amass_p3'
        camera_path = CAMERA_PATH_PROTOCOL_3
    # Get all camera XML files (any naming pattern)
    camera_files = glob.glob(os.path.join(camera_path, '*.xml'))

    def get_cam_id(path):
        filename = os.path.basename(path)
        _match = re.search(r'Camera_(\d+)\.xml', filename)
        return int(_match.group(1)) if _match else float('inf')

    camera_files.sort(key=get_cam_id)

    # meshviewer for cam update  
    mv = set_mesh_viewer(camera_files)

    dataset_type = args.dataset_type
    dataroot = args.dataroot
    phases = ['train', 'test']

    for phase in phases:
        if dataset_type == 'amass_p1':
            filename_list = glob.glob(f'./{dataroot}/*/{phase}/preprocessed/*.pkl')
        elif dataset_type == 'amass_p2':
            if phase == 'train':
                filename_list = glob.glob(f'./{dataroot}/MPI_HDM05/*/*.pkl') + glob.glob(f'./{dataroot}/BioMotionLab_NTroje/*/*.pkl')
            else:
                filename_list = glob.glob(f'./{dataroot}/CMU/*/*.pkl')
        elif dataset_type == 'amass_p3':
            if phase == 'train':
                datasets = ['ACCAD', 'BioMotionLab_NTroje', 'BMLmovi', 'CMU','EKUT', 'Eyes_Japan_Dataset', 'KIT', 'MPI_HDM05', 'MPI_mosh', 'SFU', 'TotalCapture']
                filename_list = [
                    f for dataset in datasets
                    for f in glob.glob(f'./{dataroot}/{dataset}/*/*/*.pkl')]
                
            else:
                filename_list = glob.glob(f'./{dataroot}/HumanEva/*/*/*.pkl') + glob.glob(f'./{dataroot}/Transitions_mocap/*/*/*.pkl')

        print('-------------------------------number of {} data is {}'.format(phase, len(filename_list)))
    
        # create .../triang
        if len(filename_list) > 0:
            input_dir = os.path.dirname(filename_list[0])
            output_dir = os.path.join(str(input_dir).replace('/preprocessed', '/'), args.output_dir)
            os.makedirs(output_dir, exist_ok=True)

        for filename in sorted(filename_list):
            # Construct output path
            output_path = filename.replace("/preprocessed/", f"/{args.output_dir}/").replace(".pkl", ".npz")

            # Skip if already exists
            if os.path.exists(output_path):
                logging.info(f"{output_path} already exists!")
                continue

            with open(filename, 'rb') as f:
                logging.info(f'Loading file {filename}...')
                data = pickle.load(f)
            
            key = 'yolo_keypoints' if any('yolo' in _key for _key in data.keys()) else 'pose_estimation_keypoints'
            pose_estimation_keypoints = data[key]
            confidences = pose_estimation_keypoints['confidences']
            nframes, njoints, _ = confidences.shape

            top2_conf, cams_max_conf = torch.topk(confidences, k=2, dim=-1)
            points3d = np.zeros((nframes, njoints, 3))

            check_if_shape_matches(points3d, filename)

            for f in range(nframes):
                for j in range(njoints):
                    cam_keys_for_triang = [f'vcam{k}' for k in cams_max_conf[f, j]]
                    Ks, camera_poses, image_sizes = get_cam_params_from_key(cam_keys_for_triang, camera_files)

                    point3d = triangulate(
                        Ks=Ks, 
                        camera_poses=camera_poses, 
                        yolo_keypoints=pose_estimation_keypoints, 
                        cam_keys_for_triang=cam_keys_for_triang,
                        image_sizes=image_sizes,
                        frame_id=f,
                        joint_id=j,
                        mv=mv
                    )
                    points3d[f, j] = point3d

            check_if_shape_matches(points3d, filename)

            # Save to the same output_path used earlier
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.savez(output_path, points3d=points3d, conf=top2_conf)
            logging.info(f'{output_path} successfully saved!')


if __name__ == '__main__':
    __main()
