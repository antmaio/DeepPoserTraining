# External
import argparse
import copy
import glob
import logging
import os
import pickle
import re
import tomli
from typing import List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

# Internal
from data.camera_config import (
    CAMERA_PATH_PROTOCOL_1,
    CAMERA_PATH_PROTOCOL_3,
)
from data.rendering import MeshViewer2
from data.yolo_data_gen import extract_from_xml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CAMERA_PATHS = {
    'amass_p1': CAMERA_PATH_PROTOCOL_1,
    'amass_p2': CAMERA_PATH_PROTOCOL_1,
    'amass_p3': CAMERA_PATH_PROTOCOL_3,
}

# Per dataset-type and phase, which dataset folders to glob
_DATASET_FOLDERS: dict[str, dict[str, list[str]]] = {
    'amass_p1': {
        'train': [],   # handled by wildcard glob — see _build_filename_list
        'test':  [],
    },
    'amass_p2': {
        'train': ['MPI_HDM05', 'BioMotionLab_NTroje'],
        'test':  ['CMU'],
    },
    'amass_p3': {
        'train': [
            'ACCAD', 'BioMotionLab_NTroje', 'BMLmovi', 'CMU', 'EKUT',
            'Eyes_Japan_Dataset', 'KIT', 'MPI_HDM05', 'MPI_mosh',
            'SFU', 'TotalCapture',
        ],
        'test': ['HumanEva', 'Transitions_mocap'],
    },
}

_POSE_KEY_ALIASES = ('yolo_keypoints', 'pose_estimation_keypoints')


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

class Config:
    """Simple attribute container populated from TOML + CLI overrides."""
    dataset_type: str
    dataroot: str
    output_dir: str


def _load_config(args: argparse.Namespace) -> Config:
    with open(args.config, 'rb') as f:
        toml = tomli.load(f)

    cfg = Config()
    dat_cfg = toml.get('data_path', {})
    dst_cfg = toml.get('dataset',   {})

    cfg.dataset_type = args.dataset_type or dst_cfg.get('type', 'amass_p1')
    cfg.dataroot     = args.dataroot     or dat_cfg.get('dataroot', './data/keypoints/yolov8n-pose_protocol_1')
    cfg.output_dir   = args.output_dir   or dat_cfg.get('output_dir', 'triang')

    return cfg


def log_config(cfg: Config, args: argparse.Namespace):
    """Log the configuration state in cyan, highlighting CLI overrides."""
    CYAN = "\033[36m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    logging.info(f"{CYAN}--- Configuration Summary ---{RESET}")
    for key in ['dataset_type', 'dataroot', 'output_dir']:
        val = getattr(cfg, key)
        is_overridden = getattr(args, key, None) is not None
        
        label = f"{key:15}"
        if is_overridden:
            logging.info(f"{CYAN}{label}: {BOLD}{val}{RESET}{CYAN} [CLI OVERRIDE]{RESET}")
        else:
            logging.info(f"{CYAN}{label}: {val}{RESET}")
    logging.info(f"{CYAN}-----------------------------{RESET}")


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _sorted_camera_files(camera_path: str) -> List[str]:
    """Return Camera_*.xml files from *camera_path* sorted by camera index."""
    def _cam_id(path: str) -> float:
        m = re.search(r'Camera_(\d+)\.xml', os.path.basename(path))
        return int(m.group(1)) if m else float('inf')

    return sorted(glob.glob(os.path.join(camera_path, '*.xml')), key=_cam_id)


def _extract_cam_index(key: str) -> int:
    """Extract the trailing integer from a camera key string (e.g. 'vcam3' → 3)."""
    m = re.search(r'\d+', key)
    if not m:
        raise ValueError(f"No number found in camera key: '{key}'")
    return int(m.group())


def get_cam_params(cam_keys: List[str],
                   camera_files: List[str],
                   ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]:
    """
    Extract intrinsics, extrinsics and image sizes for the requested cameras.

    Args:
        cam_keys:      Camera key strings, e.g. ['vcam0', 'vcam2'].
        camera_files:  Sorted list of camera XML paths (index matches vcam index).

    Returns:
        Ks:            List of (3, 3) intrinsic matrices.
        camera_poses:  List of (4, 4) extrinsic matrices.
        image_sizes:   List of (width, height) tuples.
    """
    Ks, camera_poses, image_sizes = [], [], []
    for key in cam_keys:
        idx = _extract_cam_index(key)
        K, pose, size = extract_from_xml(camera_files[idx])
        camera_poses.append(np.vstack((pose, [0, 0, 0, 1])))
        Ks.append(K)
        image_sizes.append(size)
    return Ks, camera_poses, image_sizes


def init_mesh_viewer(camera_files: List[str]) -> MeshViewer2:
    """Instantiate a MeshViewer2 using intrinsics from the first camera file."""
    K, _, (width, height) = extract_from_xml(camera_files[0])
    return MeshViewer2(width=width, height=height, intrinsic=K, use_offscreen=True)


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------

def _normalise_keypoint(point2d: np.ndarray, width: int, height: int) -> np.ndarray:
    """Map pixel coordinates to the [-1, 1] NDC range expected by pyrender."""
    kp = point2d.copy()
    kp[0] = (kp[0] - width  / 2) / (width  / 2)
    kp[1] = (height / 2 - kp[1]) / (height / 2)
    return kp


def triangulate(
    Ks: List[np.ndarray],
    camera_poses: List[np.ndarray],
    image_sizes: List[Tuple[int, int]],
    yolo_keypoints: dict,
    cam_keys: List[str],
    frame_id: int,
    joint_id: int,
    mv: MeshViewer2,
) -> np.ndarray:
    """
    Triangulate a single 3-D joint from two camera views.

    Args:
        Ks:             List of (3, 3) intrinsic matrices, one per camera.
        camera_poses:   List of (4, 4) extrinsic matrices, one per camera.
        image_sizes:    List of (width, height), one per camera.
        yolo_keypoints: Dict mapping camera key → (T, J, 2) tensor.
        cam_keys:       Two camera keys to use for triangulation.
        frame_id:       Frame index.
        joint_id:       Joint index.
        mv:             MeshViewer2 used to retrieve combined projection matrices.

    Returns:
        (3,) ndarray — triangulated world-space 3-D point.
    """
    proj_matrices, norm_points = [], []

    for cam_id, cam_key in enumerate(cam_keys):
        width, height = image_sizes[cam_id]
        point2d = yolo_keypoints[cam_key][frame_id, joint_id].cpu().numpy()

        mv.updateCam(camera_poses[cam_id], Ks[cam_id])
        K_gl  = mv.viewer._renderer._get_camera_matrices(mv.scene)
        P_mat = (K_gl[1] @ K_gl[0])[:3, :]

        proj_matrices.append(P_mat)
        norm_points.append(_normalise_keypoint(point2d, width, height))

    kp0, kp1 = norm_points[0], norm_points[1]
    points3d_hom = cv2.triangulatePoints(
        proj_matrices[0], proj_matrices[1],
        kp0.reshape(2, 1).astype(np.float32),
        kp1.reshape(2, 1).astype(np.float32),
    )
    return (points3d_hom[:3] / points3d_hom[3]).squeeze()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _pose_key(data: dict) -> str:
    """Return whichever known pose-estimation key exists in *data*."""
    for alias in _POSE_KEY_ALIASES:
        if alias in data:
            return alias
    raise KeyError(f"None of {_POSE_KEY_ALIASES} found in data keys: {list(data.keys())}")


def _load_pkl(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def _check_shape(points3d: np.ndarray, pkl_path: str):
    data = _load_pkl(pkl_path)
    gt = data[_pose_key(data)]['ground_truth']
    assert gt.shape == points3d.shape, (
        f"Shape mismatch — points3d {points3d.shape} vs ground_truth {gt.shape}"
    )


def _build_filename_list(dataset_type: str, phase: str, dataroot: str) -> List[str]:
    """Glob all .pkl files for the requested dataset type / phase."""
    folders = _DATASET_FOLDERS[dataset_type][phase]

    if dataset_type == 'amass_p1':
        # Single wildcard covers all subsets
        return glob.glob(os.path.join(dataroot, '*', phase, 'preprocessed', '*.pkl'))

    filenames = []
    for dataset in folders:
        pattern = os.path.join(dataroot, dataset, phase, 'preprocessed', '*.pkl')
        filenames.extend(glob.glob(pattern))
    return filenames


# ---------------------------------------------------------------------------
# 2-D keypoint animation
# ---------------------------------------------------------------------------

def animate_2d_keypoints(
    yolo_keypoints: dict,
    cam_keys: List[str],
    output_dir: str = './',
    fps: int = 60,
    point_size: int = 30,
    point_color: str = 'red',
):
    """
    Save scatter-plot animations of 2-D keypoints as AVI files.

    Args:
        yolo_keypoints: Dict mapping camera key → (T, J, 2) tensor.
        cam_keys:       Camera keys to animate.
        output_dir:     Directory for output AVI files.
        fps:            Frames per second.
        point_size:     Scatter-point size.
        point_color:    Scatter-point colour.
    """
    for cam_key in cam_keys:
        if cam_key not in yolo_keypoints:
            continue

        points2d = yolo_keypoints[cam_key].cpu().numpy()  # (T, J, 2)
        nframes = points2d.shape[0]

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.set_title(f'Camera: {cam_key}', color='white')

        x_min, x_max = points2d[..., 0].min(), points2d[..., 0].max()
        y_min, y_max = points2d[..., 1].min(), points2d[..., 1].max()
        padding = max((x_max - x_min) * 0.1, (y_max - y_min) * 0.1, 50)
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        scatter = ax.scatter([], [], c=point_color, s=point_size, alpha=0.8)

        def update(frame):
            scatter.set_offsets(points2d[frame])
            return [scatter]

        anim = FuncAnimation(fig, update, frames=nframes,
                             interval=1000 / fps, blit=True)

        output_path = os.path.join(output_dir, f'{cam_key}_2d_keypoints.avi')
        anim.save(output_path, writer='ffmpeg', codec='mpeg4',
                  fps=fps, dpi=100, bitrate=2000,
                  extra_args=['-vcodec', 'mpeg4'])
        print(f"Saved animation for {cam_key} → {output_path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Multi-view 3-D pose triangulation.')
    parser.add_argument('config', type=str, help='Path to TOML configuration file.')
    parser.add_argument('--dataset_type', choices=list(_CAMERA_PATHS),
                        help='Dataset split identifier (overrides config).')
    parser.add_argument('--dataroot', help='Path to directory containing pre-processed .pkl files (overrides config).')
    parser.add_argument('--output_dir', help='Sub-directory name for triangulated output files (overrides config).')
    args = parser.parse_args()

    cfg = _load_config(args)
    log_config(cfg, args)

    camera_path   = _CAMERA_PATHS[cfg.dataset_type]
    camera_files  = _sorted_camera_files(camera_path)
    mv            = init_mesh_viewer(camera_files)

    for phase in ('train', 'test'):
        filename_list = sorted(_build_filename_list(cfg.dataset_type, phase, cfg.dataroot))
        logging.info(f'[{phase}] {len(filename_list)} file(s) found.')

        if not filename_list:
            continue

        # Create output directory mirroring input structure
        input_dir  = os.path.dirname(filename_list[0])
        output_dir = os.path.join(input_dir.replace('/preprocessed', '/'), cfg.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        for filename in filename_list:
            output_path = (filename
                           .replace('/preprocessed/', f'/{cfg.output_dir}/')
                           .replace('.pkl', '.npz'))

            if os.path.exists(output_path):
                logging.info(f'Already exists, skipping: {output_path}')
                continue

            logging.info(f'Loading {filename}')
            data = _load_pkl(filename)

            pose_kp      = data[_pose_key(data)]
            confidences  = pose_kp['confidences']           # (T, J, num_cams)
            nframes, njoints, _ = confidences.shape

            top2_conf, cams_max_conf = torch.topk(confidences, k=2, dim=-1)
            points3d = np.zeros((nframes, njoints, 3))

            _check_shape(points3d, filename)

            for f in range(nframes):
                for j in range(njoints):
                    cam_keys = [f'vcam{k}' for k in cams_max_conf[f, j].tolist()]
                    Ks, camera_poses, image_sizes = get_cam_params(cam_keys, camera_files)

                    points3d[f, j] = triangulate(
                        Ks=Ks,
                        camera_poses=camera_poses,
                        image_sizes=image_sizes,
                        yolo_keypoints=pose_kp,
                        cam_keys=cam_keys,
                        frame_id=f,
                        joint_id=j,
                        mv=mv,
                    )

            _check_shape(points3d, filename)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.savez(output_path, points3d=points3d, conf=top2_conf)
            logging.info(f'Saved → {output_path}')


if __name__ == '__main__':
    main()