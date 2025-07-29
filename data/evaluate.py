"""
This code evaluate the reconstruction of the 3D keypoints by the desired 3D pose lifter method
"""
#External
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D

import argparse 
import os
import glob
import pathlib
from tqdm import tqdm
import typing
from pathlib import Path
import numpy as np

#Internal 
from data.data_config import FPS, YOLO_PARENTS
from anim.data.amass import YOLO_LOWER_JOINTS, YOLO_UPPER_JOINTS, YoloJoints

def animate(points3d: np.ndarray, points3d_gt: np.ndarray, filename_keypoints: str = 'anim'):
    nframes, njoints, _ = points3d.shape
    assert points3d.shape == points3d_gt.shape, "Predicted and ground truth shapes must match"

    bones = [(i, p) for i, p in enumerate(YOLO_PARENTS) if p != -1]

    fig = plt.figure(figsize=(6.4, 4.8), dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(0, 3)
    ax.view_init(elev=15, azim=-70)

    # Prediction
    scat_pred = ax.scatter([], [], [], c='r', label='Prediction')
    lines_pred = [ax.plot([], [], [], 'r')[0] for _ in bones]

    # Ground truth
    scat_gt = ax.scatter([], [], [], c='g', label='Ground Truth')
    lines_gt = [ax.plot([], [], [], 'g')[0] for _ in bones]

    ax.legend()

    def update(frame):
        joints_pred = points3d[frame]
        joints_gt = points3d_gt[frame]

        # Scatter update
        scat_pred._offsets3d = (joints_pred[:, 0], joints_pred[:, 1], joints_pred[:, 2])
        scat_gt._offsets3d = (joints_gt[:, 0], joints_gt[:, 1], joints_gt[:, 2])

        # Line update
        for i, (j1, j2) in enumerate(bones):
            # Prediction lines
            line_pred = lines_pred[i]
            x = [joints_pred[j1, 0], joints_pred[j2, 0]]
            y = [joints_pred[j1, 1], joints_pred[j2, 1]]
            z = [joints_pred[j1, 2], joints_pred[j2, 2]]
            line_pred.set_data(x, y)
            line_pred.set_3d_properties(z)

            # Ground truth lines
            line_gt = lines_gt[i]
            x = [joints_gt[j1, 0], joints_gt[j2, 0]]
            y = [joints_gt[j1, 1], joints_gt[j2, 1]]
            z = [joints_gt[j1, 2], joints_gt[j2, 2]]
            line_gt.set_data(x, y)
            line_gt.set_3d_properties(z)

        return [scat_pred, scat_gt] + lines_pred + lines_gt

    ani = FuncAnimation(fig, update, frames=nframes, interval=50, blit=False)

    writer = FFMpegWriter(
        fps=60,
        metadata=dict(artist='Me'),
        codec='mpeg4',
        bitrate=500,
        extra_args=['-pix_fmt', 'yuv420p']
    )

    avi_filename = filename_keypoints + ".avi"
    ani.save(avi_filename, writer=writer)
    print(f'Video successfully saved at {avi_filename}')

#Evaluation metrics
class EvaluationMetrics:
    def __init__(self,
        gt:np.ndarray,
        pred:np.ndarray,
        conf_pred:np.ndarray,
        excluded:dict={},
        files:typing.List[str]=None
    ):
        self.gt = gt.numpy()
        self.pred = pred
        self.conf_pred = conf_pred
        self.excluded = excluded
        self.files = files

        if self.gt.shape != self.pred.shape:
            self._check_filenames(*self.files)
            raise ValueError(f"Ground truth shape {self.gt.shape} and prediction shape {self.pred.shape} do not match!")

        if self.conf_pred is not None:  assert self.gt.shape[:-1] == self.conf_pred.shape[:-1], f"ground truth {self.gt.shape} and confidence score {self.conf_pred.shape} have not the same shape!"
        self.nframes, self.njoints, _ = self.gt.shape

        # Joint indices to include (all except excluded)
        self.included_joints = [i for i in range(self.njoints) if i not in excluded]
        self.upper_body_pose_joints = [jt for jt in YOLO_UPPER_JOINTS if jt not in excluded]
        self.lower_body_pose_joints = [jt for jt in YOLO_LOWER_JOINTS if jt not in excluded]

    @staticmethod
    def set_output(masked_errors:np.ndarray, mult:int=1):
        if not np.isnan(masked_errors).all():
            return np.nanmean(masked_errors) * mult # only compute if there's valid data
        else:
            return np.nan
    @staticmethod
    def set_output_by_joint(masked_errors:np.ndarray, mult:int=1):
        _, njoints = masked_errors.shape
        if not np.isnan(masked_errors).all():
            return np.nanmean(masked_errors, axis=0) * mult # only compute if there's valid data
        else:
            return np.full((njoints,), np.nan)  # return NaN array of shape (njoints,)

    @staticmethod
    def _check_filenames(file_gt:str, file_pred:str):
        print('file_gt : ', file_gt ,' | file_pred : ', file_pred)

    # --- Mean Error accross all joints --- 
    
    def mpjpe(self) -> np.ndarray:
        """Mean Per Joint Position Error [cm]"""
        pred = self.pred[:, self.included_joints]
        gt = self.gt[:, self.included_joints]

        if self.conf_pred is not None:
            conf = self.conf_pred[:, self.included_joints]
        else:
            #always visible
            conf = np.ones((self.gt.shape[0], len(self.included_joints), 2)) 

        errors = np.linalg.norm(pred - gt, axis=-1)
        
        visible = (conf > 0.0).all(axis=-1)  # (nframes, njoints)
        masked_errors = np.where(visible, errors, np.nan)
        return self.set_output(masked_errors, mult=100)
        #return np.mean(errors) * 100
    
    '''
    def mpjpe(self) -> np.ndarray:
        """Mean Per Joint Position Error [cm], excluding occluded joints."""
        pred = self.pred[:, self.included_joints]
        gt = self.gt[:, self.included_joints]
        # Visibility mask: True if all cameras have non-zero confidence
        conf = self.conf_pred[:, self.included_joints, :]  # shape: (nframes, njoints, ncam)

        visible = (conf > 0.0).all(axis=-1)  # shape: (nframes, njoints)
        
        errors = np.linalg.norm(pred - gt, axis=-1)  # shape: (nframes, njoints)
        # Mask occluded joints
        masked_errors = np.where(visible, errors, np.nan)
        return self.set_output(masked_errors, mult=100) #m -> cm
    '''
    
    def mpjve(self) -> np.ndarray:
        """Mean Per Joint Velocity Error [cm/s]"""
        vel_gt = np.diff(self.gt[:, self.included_joints], axis=0)
        vel_pred = np.diff(self.pred[:, self.included_joints], axis=0)
        
        if self.conf_pred is not None:
            conf = self.conf_pred[:, self.included_joints]
        else:
            #always visible
            conf = np.ones((self.gt.shape[0], len(self.included_joints), 2))
        
        visible = (conf > 0.0).all(axis=-1)  # (nframes, njoints)
        visible = visible[1:] & visible[:-1]  # Must be visible in both frames

        errors = np.linalg.norm(vel_pred - vel_gt, axis=-1)
        masked_errors = np.where(visible, errors, np.nan)
        return self.set_output(masked_errors, mult=100) #m -> cm
        #return np.mean(errors) * FPS * 10
    
    '''
    def mpjve(self) -> np.ndarray:
        """Mean Per Joint Velocity Error [cm/s], excluding occluded joints."""
        vel_gt = np.diff(self.gt[:, self.included_joints], axis=0)
        vel_pred = np.diff(self.pred[:, self.included_joints], axis=0)

        # Visibility mask for consecutive frames
        conf = self.conf_pred[:, self.included_joints, :]  # (nframes, njoints, ncam)
        visible = (conf > 0.0).all(axis=-1)  # (nframes, njoints)
        visible = visible[1:] & visible[:-1]  # Must be visible in both frames

        errors = np.linalg.norm(vel_pred - vel_gt, axis=-1)  # shape: (nframes-1, njoints)
        masked_errors = np.where(visible, errors, np.nan)
        return self.set_output(masked_errors, mult=100) #m -> cm
    '''

    def jitter(self) -> float:
        """Compute relative jitter between prediction and ground truth."""
        pred = self.pred[:, self.included_joints]
        gt = self.gt[:, self.included_joints]
        
        if self.conf_pred is not None:
            conf = self.conf_pred[:, self.included_joints, :]
        else:
            #always visible
            conf = np.ones((self.gt.shape[0], len(self.included_joints), 2)) 

        
        visible = (conf > 0.0).all(axis=-1)  # shape: (nframes, njoints)
        visible_mask = (
            visible[:-3] & visible[1:-2] & visible[2:-1] & visible[3:]
        )  # shape: (nframes - 3, njoints)
        true_mask = np.ones_like(visible_mask, dtype=bool)

        # Compute third-order difference
        diff_pred = (
            pred[3:] - 3 * pred[2:-1] + 3 * pred[1:-2] - pred[:-3]
        ) * (FPS ** 3)
        diff_gt = (
            gt[3:] - 3 * gt[2:-1] + 3 * gt[1:-2] - gt[:-3]
        ) * (FPS ** 3)

        # Apply mask
        pred_norm = np.linalg.norm(diff_pred, axis=-1)
        gt_norm = np.linalg.norm(diff_gt, axis=-1)

        # Only consider visible joints
        if np.sum(visible_mask) == 0: #no data
            return np.nan
        else:
            jitter_pred = np.sum(pred_norm * visible_mask) / np.sum(visible_mask)
            jitter_gt = np.sum(gt_norm * true_mask) / np.sum(true_mask)
        
        return jitter_pred / jitter_gt
    '''
    def jitter(self) -> float:
        """Compute relative jitter between prediction and ground truth, excluding occluded joints."""
        pred = self.pred[:, self.included_joints]
        gt = self.gt[:, self.included_joints]

        diff_pred = (
            pred[3:] - 3 * pred[2:-1] + 3 * pred[1:-2] - pred[:-3]
        )
        diff_gt = (
            gt[3:] - 3 * gt[2:-1] + 3 * gt[1:-2] - gt[:-3]
        )

        # Visibility: must be visible in all four consecutive frames
        conf = self.conf_pred[:, self.included_joints, :]  # (nframes, njoints, ncam)
        visible = (conf > 0.0).all(axis=-1)  # (nframes, njoints)
        visible = (
            visible[3:] & visible[2:-1] & visible[1:-2] & visible[:-3]
        )  # shape: (nframes-3, njoints)

        diff_pred *= FPS ** 3
        diff_gt *= FPS ** 3

        jitter_pred = np.linalg.norm(diff_pred, axis=-1)  # (nframes-3, njoints)
        jitter_gt = np.linalg.norm(diff_gt, axis=-1)

        # Mask occluded joints
        jitter_pred = np.where(visible, jitter_pred, np.nan)

        # Compute mean jitter, ignoring NaNs
        mean_pred = np.nanmean(jitter_pred)
        mean_gt = np.mean(jitter_gt)

        return mean_pred / mean_gt if mean_gt > 0 else np.nan
    '''
    def occlusion_rate(self):
        """
        Computes the fraction of occluded joints in prediction.
        A joint is considered occluded if at least one camera reports a confidence score of 0.0.
        """
        # Filter to included joints
        conf = self.conf_pred[:, self.included_joints, :]  # shape: (nframes, nincluded_joints, ncam)
        # A joint is occluded if ANY camera reports 0.0 confidence
        occluded = (conf == 0.0).any(axis=-1)  # shape: (nframes, nincluded_joints)
        # Total number of joints across all frames
        total_points = np.prod(occluded.shape)
        # Number of occluded joints
        missing = occluded.sum()
        return missing / total_points
    
    # --- Errors by joint for further analyzes --- 

    '''
    def pjpe_by_joint(self) -> np.ndarray:
        """Per Joint Position Error [cm], excluding occluded joints."""
        errors = np.linalg.norm(self.pred - self.gt, axis=-1)  # shape: (nframes, njoints)
        
        # Visibility mask: joint is visible if all cameras have non-zero confidence
        visible = (self.conf_pred > 0.0).all(axis=-1)  # shape: (nframes, njoints)
        
        # Mask occluded values with NaN
        masked_errors = np.where(visible, errors, np.nan)
        
        # Mean over time, ignoring occluded frames
        return np.nanmean(masked_errors, axis=0) * 100  # cm

    def pjve_by_joint(self) -> np.ndarray:
        """Per Joint Velocity Error [cm/s], excluding occluded joints."""
        vel_gt = np.diff(self.gt, axis=0)      # shape: (nframes-1, njoints, 3)
        vel_pred = np.diff(self.pred, axis=0)  # shape: (nframes-1, njoints, 3)
        errors = np.linalg.norm(vel_pred - vel_gt, axis=-1)  # shape: (nframes-1, njoints)
        
        # Visibility mask: must be visible in both consecutive frames
        visible = (self.conf_pred > 0.0).all(axis=-1)  # shape: (nframes, njoints)
        visible = visible[1:] & visible[:-1]           # shape: (nframes-1, njoints)
        
        # Mask occluded values
        masked_errors = np.where(visible, errors, np.nan)

        return np.nanmean(masked_errors, axis=0) * FPS * 100  # cm/s
    
    def occlusion_rate_by_joint(self) -> np.ndarray:
        conf = self.conf_pred[:, self.included_joints, :]  # shape: (nframes, nincluded_joints, ncam)
        occluded = (conf == 0.0).any(axis=-1)              # shape: (nframes, nincluded_joints)
        # Flatten boolean to int for uniqueness check
        unique_patterns = np.unique(occluded.T, axis=0)
        #print(f"Number of unique joint occlusion patterns: {len(unique_patterns)}")

        #print("occluded shape:", occluded.shape)
        #print("occluded (first 5 frames):", occluded[:5].astype(int))  # To see variation

        occlusion_counts = occluded.sum(axis=0)
        total_frames = conf.shape[0]
        rates = occlusion_counts / total_frames

        #print("occlusion rates:", rates)

        #assert False

        return rates
    '''
    
    def occlusion_rate_by_joint(self) -> np.ndarray:
        """
        Computes the occlusion rate per joint.
        A joint is considered occluded if at least one camera reports a confidence score of 0.0.
        Returns:
            np.ndarray of shape (nincluded_joints,) with occlusion rate per joint.
        """
        # A joint is occluded if ANY camera reports 0.0 confidence
        occluded = (self.conf_pred == 0.0).any(axis=-1)  # shape: (nframes, nincluded_joints)
        # Sum occlusions per joint (axis=0: over frames)
        occlusion_counts = occluded.sum(axis=0)  # shape: (nincluded_joints,)
        # Total number of frames
        total_frames = self.conf_pred.shape[0]
        # Occlusion rate = occluded frames / total frames per joint
        return occlusion_counts / total_frames  # shape: (nincluded_joints,)
    
    def pjpe_by_joint(self)->np.ndarray:
        """Mean Per Joint Position Error [cm]"""
        errors = np.linalg.norm(self.pred - self.gt, axis=-1)

        conf = self.conf_pred if self.conf_pred is not None else np.ones((self.gt.shape[0], self.gt.shape[1], 2))
        
        visible = (conf > 0.0).all(axis=-1)  # (nframes, njoints)
        masked_errors = np.where(visible, errors, np.nan)
        return self.set_output_by_joint(masked_errors, mult=100)
        #return np.mean(errors, axis=0) * 100  # convert from meters to centimeters


    def pjve_by_joint(self) -> np.ndarray:
        """Mean Per Joint Velocity Error [cm/s]"""
        vel_gt = np.diff(self.gt, axis=0)
        vel_pred = np.diff(self.pred, axis=0)
        errors = np.linalg.norm(vel_pred - vel_gt, axis=-1)

        conf = self.conf_pred if self.conf_pred is not None else np.ones((self.gt.shape[0], self.gt.shape[1], 2))

        visible = (conf > 0.0).all(axis=-1)  # (nframes, njoints)
        visible = visible[1:] & visible[:-1]  # Must be visible in both frames
        masked_errors = np.where(visible, errors, np.nan)

        return self.set_output_by_joint(masked_errors, mult=100) #m -> cm
        #return np.mean(errors, axis=0) * FPS * 100  # convert m/frame -> cm/s
    

# --- Pretty print ---
def print_summary(mpjpes:typing.List, mpjves:typing.List, jitters:typing.List, occlusion_rates:typing.List=None, **kwargs):

    method = kwargs["method"]
    model = kwargs["model"]

    def format_metric(name, values):
        mean = np.nanmean(values)
        std = np.nanstd(values)
        return f"{name:<20}: {mean:.2f} ± {std:.2f}"

    print(f"\n📊 Evaluation Summary for {method} with 2D data from {model}")
    print("-" * 40)
    print(format_metric("MPJPE [cm]", mpjpes))
    print(format_metric("MPJVE [cm/s]", mpjves))
    print(format_metric("Jitter ratio", jitters))
    if occlusion_rates is not None: print(format_metric("Occlusion Rate [%]", [v * 100 for v in occlusion_rates]))
    print("-" * 40)

def print_summary_by_joint(pjpes_by_joint: typing.List, pjves_by_joint: typing.List, occlusion_rates_by_joint:typing.List=None,  excluded:dict={}, **kwargs):    
    method = kwargs["method"]
    model = kwargs["model"]

    pjpes_by_joint           = np.array(pjpes_by_joint)
    pjves_by_joint           = np.array(pjves_by_joint)
    if occlusion_rates_by_joint is not None:    occlusion_rates_by_joint = np.array(occlusion_rates_by_joint)

    # Joint names and indices to keep (exclude eyes and ears)
    joint_indices = [j for j in range(len(YoloJoints)) if j not in excluded and j != YoloJoints.NUM_JTS]
    joint_names = [YoloJoints(j).name.replace("_", " ").title() for j in joint_indices]

    print(f"\n📊 Evaluation Summary for {method} with 2D data from {model}")
    print("-" * 40)
    print(f"{'Joint':<20} {'PJPE [cm]':>15} {'PJVE [cm/s]':>15} {'Occlusion rate [%]':>15}")
    print("-" * 40)

    for j,jname in zip(joint_indices, joint_names):
        pjpe_mean, pjpe_std = np.nanmean(pjpes_by_joint[:, j]), np.nanstd(pjpes_by_joint[:, j])
        pjve_mean, pjve_std = np.nanmean(pjves_by_joint[:, j]), np.nanstd(pjves_by_joint[:, j])
        if occlusion_rates_by_joint is not None:
            occl_rate_mean, occl_rate_std = np.mean(occlusion_rates_by_joint[:, j] * 100), np.std(occlusion_rates_by_joint[:, j] * 100)
            print(f"{jname:<20} {pjpe_mean:>7.2f} ± {pjpe_std:<5.2f} {pjve_mean:>7.2f} ± {pjve_std:<5.2f} {occl_rate_mean:>7.2f} ± {occl_rate_std:<5.2f}")
        else:
            print(f"{jname:<20} {pjpe_mean:>7.2f} ± {pjpe_std:<5.2f} {pjve_mean:>7.2f} ± {pjve_std:<5.2f}")

    print("-" * 40)

#Loading data
def load_gt(relative_rec_path:str):
    data = np.load(relative_rec_path, allow_pickle=True)
    return data

def load_kp(relative_rec_path: str):
    """
    Given a .pkl path, loads the corresponding .npz file.
    """
    npz_path = pathlib.Path(relative_rec_path).with_suffix('.npz')
    data = np.load(npz_path, allow_pickle=True)
    return data

#Check data lists 
def normalize_path(path: str) -> str:
    # Split path into parts
    parts = path.split('/')

    # Remove 'triang' or 'preprocessed' components
    parts = [p for p in parts if p not in ('triang', 'preprocessed', 'openmpl')]

    if not parts:
        return ''

    # Remove extension from last part (filename)
    parts[-1] = os.path.splitext(parts[-1])[0]

    # Rebuild path
    return '/'.join(parts)

def compare_lists(files_gt: typing.List[str], files_pred: typing.List[str]):
    # Normalize
    gt_norm = sorted([normalize_path(f) for f in files_gt])
    pred_norm = sorted([normalize_path(f) for f in files_pred])

    for gt, pred in zip(gt_norm, pred_norm):
        if gt != pred:
            print(gt, pred)
            assert False
            
    else:
        # If all items matched in the zipped loop, the missing item is the last one in pred_norm
        print("Missing item from gt_norm:", pred_norm[-1])
    
    

def one_method_vs_gt(args)->None:

    #comparison
    mpjpes, mpjves, jitters, occlusion_rates                           = [], [], [], []
    pjpes_by_joint, pjves_by_joint, occlusion_rates_by_joint           = [], [], []


    # Ensure patterns match correct file types
    files_pred = sorted(glob.glob(os.path.join(args.model, "**", "**", args.method, "*.npz")))
    files_gt = sorted(glob.glob(os.path.join(args.model, "**", "**", "preprocessed", "*.pkl")))
    # defining not used joints 
    excluded_joints = {YoloJoints.NOSE, YoloJoints.LEFT_EYE, YoloJoints.RIGHT_EYE, YoloJoints.LEFT_EAR, YoloJoints.RIGHT_EAR}

    if len(files_gt) == len(files_pred):

        for _, (file_gt, file_pred) in tqdm(enumerate(zip(files_gt, files_pred))):

            #print(file_gt, file_pred)

            data_gt, data_pred = load_gt(file_gt), load_kp(file_pred)

            #ground truth 3d data in coco format
            points3d_gt = data_gt.get('yolo_keypoints') or data_gt.get('pose_estimation_keypoints') # Exactly one of these will be present
            
            assert 'ground_truth' in points3d_gt, f"ground_truth is not a key in data for file {file_gt} | {file_pred}!"
            
            points3d_gt = points3d_gt['ground_truth']
            #3d data computed from args.method
            points3d_pred = data_pred['points3d']
            conf_pred = data_pred.get('conf')

            #animate(points3d=points3d_pred, points3d_gt=points3d_gt)
            #print(points3d_gt.shape, points3d_pred.shape)            

            #compute metrics
            metrics = EvaluationMetrics(points3d_gt, points3d_pred, conf_pred, excluded=excluded_joints, files=[file_gt, file_pred])
            mpjpes.append(metrics.mpjpe())
            mpjves.append(metrics.mpjve())
            jitters.append(metrics.jitter())
            list_of_metrics = [mpjpes, mpjves, jitters]
            if conf_pred is not None:   
                occlusion_rates.append(metrics.occlusion_rate())
                list_of_metrics.append(occlusion_rates)
        
            pjpes_by_joint.append(metrics.pjpe_by_joint())
            pjves_by_joint.append(metrics.pjve_by_joint())
            list_of_metrics_by_joints = [pjpes_by_joint, pjves_by_joint]
            if conf_pred is not None:   
                list_of_metrics_by_joints.append(occlusion_rates_by_joint)
                occlusion_rates_by_joint.append(metrics.occlusion_rate_by_joint())


        #Print metrics
        kwargs = {'method':args.method,'model':args.model, 'excluded':excluded_joints}
        print_summary(*list_of_metrics, **kwargs)
        print_summary_by_joint(*list_of_metrics_by_joints, **kwargs)
    else:
        compare_lists(files_gt, files_pred)
        raise ValueError(f"Filename list of different length for files_gt {len(files_gt)} and files_pred {len(files_pred)}!")
        

def methods_vs_gt(args):
    pass

def __main():
    
    parse = argparse.ArgumentParser()
    parse.add_argument('--method', type=str, required=True, choices=('triang','openmpl'), help="Use keypoints from method [triang|openmpl]")
    parse.add_argument('--model', type=str, help='Path to keypoints folder', default='./data/keypoints/yolov8x-pose_protocol_1')
    args = parse.parse_args()
    
    assert os.path.exists(args.model), f"args.model {args.model} does not exists" 

    one_method_vs_gt(args)

if __name__ == '__main__':
    __main()