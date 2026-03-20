import os
import shutil
import glob
from tqdm import tqdm
import argparse
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch

#Internal 
from utils import utils_filters
from anim.data.amass import load_gt, load_kp, preprocess_keypoints
from data.data_config import YoloJoints

# Example for COCO-style keypoints
SKELETON_CONNECTIONS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6)
]

def get_savepath(cfg)->str:
    if getattr(cfg, 'with_kalman_filter', False):
        filter_name = getattr(cfg, 'filter_name', None)
        base_name = f"data/vis/kalman_{filter_name}" if filter_name else "data/vis/kalman"
        savepath = f"{base_name}_0"
        counter = 1
        # Check if the folder exists, and increment if it does
        while os.path.exists(savepath):
            savepath = f"{base_name}_{counter}"
            counter += 1
    else:
        # This path remains static and does not increment
        savepath = "data/vis/no_kalman"

    os.makedirs(savepath, exist_ok=True)
    return savepath

def static_3D_plot(points3d_pred, points3d_gt, cfg, savepath:str, motion_id:int, connections=SKELETON_CONNECTIONS, nframes:int=50, stride:int=1):
    # Slice inputs based on stride and nframes
    indices = np.arange(0, min(len(points3d_pred), len(points3d_gt)), stride)[:nframes]

    points3d_pred = torch.Tensor(points3d_pred[indices])
    points3d_gt = points3d_gt[indices]

    # --- Recenter: Move first frame's joint 0 to (0,0) in X,Y ---
    # We use the ground truth joint 0 of the first frame as the reference

    offset_xy = points3d_gt[0, 0, :2].clone()

    points3d_gt[..., :2] -= offset_xy
    points3d_pred[..., :2] -= offset_xy

    n_actual = len(indices)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Set background color based on Kalman filter
    # Light Blue for no filter, Light Orange for filter
    bg_color = (0.8, 0.9, 1.0, 0.15) if not getattr(cfg, 'with_kalman_filter', False) else (1.0, 0.9, 0.8, 0.15)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor((0.0, 0.0, 0.0, 0.0)) # Keep axes background transparent

    # Create temporal alpha values to show progression
    alphas = np.linspace(0.1, 0.8, n_actual)
    
    # 1. Plot Ground Truth (Joints + Bones + Temporal Trails)
    for f in range(n_actual):
        gt = points3d_gt[f]
        
        # Highlight logic for joints
        body_mask = np.arange(len(gt)) >= 5
        
        # Scatter Body Joints (5+)
        ax.scatter(gt[body_mask, 0], gt[body_mask, 1], gt[body_mask, 2], 
                   c='forestgreen', s=20, alpha=alphas[f])
        # Scatter Head Joints (0-4) - Lighter
        ax.scatter(gt[~body_mask, 0], gt[~body_mask, 1], gt[~body_mask, 2], 
                   c='lightgreen', s=10, alpha=alphas[f] * 0.3)
        
        # Draw skeleton for the last frame
        if f == n_actual - 1:
            for connection in connections:
                start_idx, end_idx = connection
                # A connection is "body" if both joints are body joints
                is_body_connection = start_idx >= 5 and end_idx >= 5
                
                color = 'forestgreen' if is_body_connection else 'lightgray'
                linewidth = 2 if is_body_connection else 1
                alpha = 0.4 if is_body_connection else 0.15
                
                if start_idx < len(gt) and end_idx < len(gt):
                    xs = [gt[start_idx, 0], gt[end_idx, 0]]
                    ys = [gt[start_idx, 1], gt[end_idx, 1]]
                    zs = [gt[start_idx, 2], gt[end_idx, 2]]
                    ax.plot(xs, ys, zs, c=color, alpha=alpha, linewidth=linewidth)

    # 2. Plot Predictions (Joints + Temporal Trails)
    for f in range(n_actual):
        pred = points3d_pred[f]
        
        # Similar logic for predictions
        body_mask_pred = np.arange(len(pred)) >= 5
        
        # Scatter Body Joints (5+)
        ax.scatter(pred[body_mask_pred, 0], pred[body_mask_pred, 1], pred[body_mask_pred, 2], 
                   c='crimson', s=25, marker='x', alpha=alphas[f])
        
        # Scatter Head Joints (0-4) - Lighter
        ax.scatter(pred[~body_mask_pred, 0], pred[~body_mask_pred, 1], pred[~body_mask_pred, 2], 
                   c='pink', s=15, marker='x', alpha=alphas[f] * 0.3)

    # 3. Connect dots of the same joints through the frames (The "Trails")
    num_joints = points3d_gt.shape[1]
    for j in range(num_joints):
        # GT Trails
        ax.plot(points3d_gt[:, j, 0], 
                points3d_gt[:, j, 1], 
                points3d_gt[:, j, 2], 
                c='forestgreen', alpha=0.2, linestyle='--')
        
        # Pred Trails (for body joints only)
        if j >= 5:
            ax.plot(points3d_pred[:, j, 0], 
                    points3d_pred[:, j, 1], 
                    points3d_pred[:, j, 2], 
                    c='crimson', alpha=0.2, linestyle=':')

    # --- Styling ---
    ax.view_init(elev=15, azim=-60)
    ax.set_title(f'3D Pose: Temporal Motion Trails ({n_actual} frames, stride={stride})', fontsize=14)
    
    # Remove axis system for a cleaner look
    ax.set_axis_off()
    
    # Setting limits manually since axis is off (prevents zooming out too much)
    # We center around 0,0 due to the recentering step
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 2)
    
    # Custom Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Ground Truth Joints', markerfacecolor='forestgreen', markersize=12),
        Line2D([0], [0], marker='x', color='w', label='Pose Estimation Joints', markeredgecolor='crimson', markersize=12),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=16)
    
    plt.savefig(f"{savepath}/skeleton_plot_{motion_id}.png", dpi=300)
    print(f"Plot saved: Temporal joint trails for {n_actual} frames at {savepath}/skeleton_plot_{motion_id}.png")
    return savepath

def animate(args, cfg):
    files_gt = sorted(glob.glob(os.path.join(args.model, "**", "test", "preprocessed", "*.pkl")))
    files_yolo = sorted(glob.glob(os.path.join(args.model, "**", "test", "triang", "*.npz")))

    # defining not used joints 
    excluded_joints = {YoloJoints.NOSE, YoloJoints.LEFT_EYE, YoloJoints.RIGHT_EYE, YoloJoints.LEFT_EAR, YoloJoints.RIGHT_EAR}

    kalman_params_path = None
    if args.use_kalman_filter:
        # Load parameters from JSON
        kalman_params_path = './utils/kalman_parameters_animate.json'

    savepath = get_savepath(cfg)

    if len(files_gt) == len(files_yolo):
        if args.motion_id is not None:
            files_gt = [files_gt[args.motion_id]]
            files_yolo = [files_yolo[args.motion_id]]

        for i, (file_gt, file_yolo) in tqdm(enumerate(zip(files_gt, files_yolo))):

            data_gt = load_gt(file_gt)
            data_pred = load_kp(cfg, file_yolo)

            #ground truth 3d data in coco format
            points3d_gt = data_gt.get('yolo_keypoints') or data_gt.get('pose_estimation_keypoints') # Exactly one of these will be present
            
            assert 'ground_truth' in points3d_gt, f"ground_truth is not a key in data for file {file_gt} | {file_yolo}!"
            
            points3d_gt = points3d_gt['ground_truth']
            #3d data computed from args.method
            conf_pred = data_pred.get('conf')

            points3d_pred = preprocess_keypoints(torch.tensor(data_pred['points3d'], dtype=torch.float32), cfg, conf_scores=torch.tensor(conf_pred, dtype=torch.float32), threshold=0.5, kalman_params_path=kalman_params_path) 
            
            motion_id_str_in_savepath = args.motion_id if args.motion_id is not None else i
            static_3D_plot(points3d_pred.numpy(), points3d_gt, cfg, savepath,motion_id=motion_id_str_in_savepath, nframes=args.nframes, stride=args.stride)
            
            #copy kalman parameters to savepath once, if kalman filtering enabled
            if getattr(cfg, 'with_kalman_filter', False) and i == 0:
                shutil.copy('./utils/kalman_parameters_animate.json', os.path.join(savepath, 'kalman_parameters_animate.json'))
            
def __main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', type=str, help='Path to keypoints folder', default='./data/keypoints/yolov8n-pose_protocol_1')
    parse.add_argument('--use_kalman_filter', action='store_true')
    parse.add_argument('--filter_name', type=str, default=None)
    parse.add_argument('--nframes', type=int, default=15, help='Number of frames to visualize')
    parse.add_argument('--stride', type=int, default=1, help='Stride between frames for visualization')
    parse.add_argument('--motion_id', type=int, default=None, help="Motion id to process in animate function. If None, run over all the motion files")
    args = parse.parse_args()

    class Config:
        pass
    cfg = Config()
    cfg.with_kalman_filter = args.use_kalman_filter
    cfg.filter_name = args.filter_name
    cfg.mode = "triang"
    
    animate(args, cfg)
    
if __name__ == '__main__':
    __main()