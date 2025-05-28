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
#Internal
from data.camera_config import CAMERA_PATH
from data.yolo_data_gen import extract_from_xml

def triangulate(
    Ks:np.ndarray, 
    camera_poses:np.ndarray, 
    image_sizes:List[Tuple[int, int]], 
    yolo_keypoints:torch.Tensor, 
    cam_keys_for_triang:List[str],
    frame_id:int,
    joint_id:int
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

    proj_matrices, points2d = [], []

    for cam_id, cam_key in enumerate(cam_keys_for_triang):
        image_width, image_height = image_sizes[cam_id]
        Ks[cam_id], camera_poses[cam_id], yolo_keypoints[cam_key]
        point2d = yolo_keypoints[cam_key][frame_id, joint_id].cpu().numpy()
        point2d_normalized = np.array([
            point2d[0] / image_width,
            point2d[1] / image_height
        ], dtype=np.float32)
        points2d.append(point2d_normalized)  # <-- FIX: Append to points2d

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
    
    #from homogeneous to 3D
    return (points3d_hom[:3] / points3d_hom[3]).flatten()


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
    args = parse.parse_args()

    
    # Get all camera XML files (any naming pattern)
    camera_files = glob.glob(os.path.join(CAMERA_PATH, '*.xml'))

    dataset_type = args.dataset_type
    dataroot = args.dataroot
    phases = ['train', 'test']

    for phase in phases:
        if dataset_type == 'amass_p1':
            filename_list = glob.glob(f'./{dataroot}/*/{phase}/*.pkl')
        elif dataset_type == 'amass_p2':
            if phase == 'train':
                filename_list = glob.glob(f'./{dataroot}/MPI_HDM05/*/*.pkl') + glob.glob(f'./{dataroot}/BioMotionLab_NTroje/*/*.pkl')
            else:
                filename_list = glob.glob(f'./{dataroot}/CMU/*/*.pkl')
        
        print('-------------------------------number of {} data is {}'.format(phase, len(filename_list)))
    
        for filename in filename_list:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        
        yolo_keypoints = data['yolo_keypoints'] 
        confidences = yolo_keypoints['confidences']
        nframes, njoints, _ = confidences.shape
                
        # Get the indices of the top-2 cameras with highest confidence per joint
        top2_conf, cams_max_conf = torch.topk(confidences, k=2, dim=-1)
        
        # `cams_max_conf` now contains the camera IDs (0, 1, or 2) of the top-2 confidences
        points3d = np.zeros((nframes, njoints, 3))

        for f in range(nframes):
            for j in range(njoints):
                cam_keys_for_triang = [f'vcam{k}' for k in cams_max_conf[f,j]]
                Ks, camera_poses, image_sizes = get_cam_params_from_key(cam_keys_for_triang, camera_files)
        
                #Get 3D world positions from 2D yolo keypoints 
                point3d = triangulate(
                    Ks=Ks, 
                    camera_poses=camera_poses, 
                    yolo_keypoints=yolo_keypoints, 
                    cam_keys_for_triang=cam_keys_for_triang,
                    image_sizes=image_sizes,
                    frame_id=f,
                    joint_id=j
                )

                points3d[f, j] = point3d
        
        np.savez(os.path.splitext(filename)[0] + ".npz", points3d=points3d)



if __name__ == '__main__':
    __main()
