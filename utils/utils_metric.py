import torch
import numpy as np
import typing
from data.data_config import FPS, YOLO_PARENTS
from anim.data.amass import YOLO_LOWER_JOINTS, YOLO_UPPER_JOINTS, YoloJoints
import warnings

# Suppress "Mean of empty slice" warnings from nanmean on all-nan joints/frames
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")


def penetration_error(pred_mesh, floor_height):
    '''
    pred_mesh: (batch, v_num, 3)
    '''
    lowest_z, lowest_z_index = pred_mesh.min(1)
    lowest_z = lowest_z[:, 2]
    lowest_z_index = lowest_z_index[:, 2]
    floor_height = floor_height.float()
    lowest_z_filtered = torch.where(lowest_z>=floor_height, torch.FloatTensor([0]).to(pred_mesh.device), lowest_z)
    floor_height_filtered = torch.where(lowest_z>=floor_height, torch.FloatTensor([0]).to(pred_mesh.device), floor_height)
    seq_len = pred_mesh.shape[0] // floor_height.shape[0]
    floor_height = floor_height[:, None].repeat(1, seq_len).view(-1)
    return torch.abs(floor_height_filtered - lowest_z_filtered).mean()

def floating_error(pred_mesh, floor_height):
    '''
    pred_mesh: (batch, v_num, 3)
    '''
    lowest_z, lowest_z_index = pred_mesh.min(1)
    lowest_z = lowest_z[:, 2]
    lowest_z_index = lowest_z_index[:, 2]

    floor_height = floor_height.float()
    lowest_z_filtered = torch.where(lowest_z<=floor_height, torch.FloatTensor([0]).to(pred_mesh.device), lowest_z)
    floor_height_filtered = torch.where(lowest_z<=floor_height, torch.FloatTensor([0]).to(pred_mesh.device), floor_height)
    seq_len = pred_mesh.shape[0] // floor_height.shape[0]
    floor_height = floor_height[:, None].repeat(1, seq_len).view(-1)
    return torch.abs(floor_height_filtered - lowest_z_filtered).mean()

def skating_error(pred, gt):
    '''
    pred_mesh: (batch, v_num, 3)
    '''
    seq_len = pred.shape[0]
    batch = pred.shape[0] // seq_len

    # batch, seq_len = pred.shape[0], pred.shape[1]
    pred = pred.reshape(batch, seq_len, -1)
    gt = gt.reshape(batch, seq_len, -1)
    pred = pred[:, :, :22*3].reshape(batch, seq_len, 22, 3)
    gt = gt[:, :, :22*3].reshape(batch, seq_len, 22, 3)

    # 'L_Ankle',  # 7, 'R_Ankle',  # 8 , 'L_Foot',  # 10, 'R_Foot',  # 11
    l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11
    relevant_joints = [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx]
    gt_joint_xyz = gt[:, :, relevant_joints, :]  # [BatchSize, 4, 3, Frames]
    gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, 1:, :, :] - gt_joint_xyz[:, :-1, :, :], dim=-1)  # [BatchSize, 4, Frames]
    fc_mask = torch.unsqueeze((gt_joint_vel <= 0.01), dim=-1).repeat(1, 1, 1, 3)
    pred_joint_xyz = pred[:, :, relevant_joints, :]  # [BatchSize, 4, 3, Frames]
    pred_vel = pred_joint_xyz[:, 1:, :, :] - pred_joint_xyz[:, :-1, :, :]
    pred_vel[~fc_mask] = 0
    foot_concat_loss = torch.abs(torch.zeros(pred_vel.shape, device=pred_vel.device) - pred_vel).mean()
    return foot_concat_loss

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
        self.included_joints = [i for i in range(self.njoints) if i not in excluded] #ok
        self.upper_body_pose_joints = [jt for jt in YOLO_UPPER_JOINTS if jt not in excluded] #ok
        self.lower_body_pose_joints = [jt for jt in YOLO_LOWER_JOINTS if jt not in excluded] #ok

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
        
        visible = (conf >= 0.5).all(axis=-1)  # shape: (nframes, njoints)
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
        vel_gt = np.diff(self.gt[:, self.included_joints], axis=0) * FPS
        vel_pred = np.diff(self.pred[:, self.included_joints], axis=0) * FPS
        
        if self.conf_pred is not None:
            conf = self.conf_pred[:, self.included_joints]
        else:
            #always visible
            conf = np.ones((self.gt.shape[0], len(self.included_joints), 2))
        
        visible = (conf >= 0.5).all(axis=-1)  # (nframes, njoints)
        visible = visible[1:] & visible[:-1]  # Must be visible in both frames

        errors = np.linalg.norm(vel_pred - vel_gt, axis=-1)
        masked_errors = np.where(visible, errors, np.nan)
        return self.set_output(masked_errors, mult=100) #m/frame -> cm/s
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

        
        visible = (conf >= 0.5).all(axis=-1)  # shape: (nframes, njoints)
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
        # A joint is occluded if ANY camera reports < 0.5 confidence
        occluded = (conf < 0.5).any(axis=-1)  # shape: (nframes, nincluded_joints)
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
        A joint is considered occluded if at least one camera reports a confidence score < 0.5.
        Returns:
            np.ndarray of shape (nincluded_joints,) with occlusion rate per joint.
        """
        # A joint is occluded if ANY camera reports 0.0 confidence
        occluded = (self.conf_pred < 0.5).any(axis=-1)  # shape: (nframes, nincluded_joints)
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
        
        visible = (conf >= 0.5).all(axis=-1)  # (nframes, njoints)
        masked_errors = np.where(visible, errors, np.nan)
        return self.set_output_by_joint(masked_errors, mult=100)

    def pjpe_by_joint_m(self)->np.ndarray:
        """Mean Per Joint Position Error [cm] for occluded frames ONLY"""
        errors = np.linalg.norm(self.pred - self.gt, axis=-1)
        conf = self.conf_pred if self.conf_pred is not None else np.ones((self.gt.shape[0], self.gt.shape[1], 2))
        missing = ~(conf >= 0.5).all(axis=-1)
        masked_errors = np.where(missing, errors, np.nan)
        return self.set_output_by_joint(masked_errors, mult=100)
        #return np.mean(errors, axis=0) * 100  # convert from meters to centimeters


    def pjve_by_joint(self) -> np.ndarray:
        """Mean Per Joint Velocity Error [cm/s]"""
        vel_gt = np.diff(self.gt, axis=0) * FPS
        vel_pred = np.diff(self.pred, axis=0) * FPS
        errors = np.linalg.norm(vel_pred - vel_gt, axis=-1)

        conf = self.conf_pred if self.conf_pred is not None else np.ones((self.gt.shape[0], self.gt.shape[1], 2))

        visible = (conf >= 0.5).all(axis=-1)  # (nframes, njoints)
        visible = visible[1:] & visible[:-1]  # Must be visible in both frames
        masked_errors = np.where(visible, errors, np.nan)

        return self.set_output_by_joint(masked_errors, mult=100) #m -> cm

    def pjve_by_joint_m(self) -> np.ndarray:
        """Mean Per Joint Velocity Error [cm/s] for occluded frames ONLY"""
        vel_gt = np.diff(self.gt, axis=0) * FPS
        vel_pred = np.diff(self.pred, axis=0) * FPS
        errors = np.linalg.norm(vel_pred - vel_gt, axis=-1)

        conf = self.conf_pred if self.conf_pred is not None else np.ones((self.gt.shape[0], self.gt.shape[1], 2))

        # Occluded in at least one of the two frames
        missing = ~((conf >= 0.5).all(axis=-1)[1:] & (conf >= 0.5).all(axis=-1)[:-1])
        masked_errors = np.where(missing, errors, np.nan)

        return self.set_output_by_joint(masked_errors, mult=100)
        #return np.mean(errors, axis=0) * FPS * 100  # convert m/frame -> cm/s
    

    # --- Mean errors regardless occlusion ---    
    def mpjpe_o(self) -> np.ndarray:
        """Mean Per Joint Position Error [cm]"""
        pred = self.pred[:, self.included_joints]
        gt = self.gt[:, self.included_joints]
        errors = np.linalg.norm(pred - gt, axis=-1) #m-> cm
        return np.mean(errors) * 100
        #return np.mean(errors) * 100

    def mpjve_o(self) -> np.ndarray:
        """Mean Per Joint Velocity Error [cm/s]"""
        vel_gt = np.diff(self.gt[:, self.included_joints], axis=0) * FPS
        vel_pred = np.diff(self.pred[:, self.included_joints], axis=0) * FPS
        errors = np.linalg.norm(vel_pred - vel_gt, axis=-1)
        return np.mean(errors) * 100
    
    # --- Mean errors by joint regardless occlusion ---
    def pjpe_by_joint_o(self)->np.ndarray:
        """Mean Per Joint Position Error [cm]"""
        errors = np.linalg.norm(self.pred- self.gt, axis=-1)
        return np.mean(errors, axis=0) * 100
        #return np.mean(errors, axis=0) * 100  # convert from meters to centimeters

    def pjve_by_joint_o(self) -> np.ndarray:
        """Mean Per Joint Velocity Error [cm/s]"""
        vel_gt = np.diff(self.gt, axis=0) * FPS
        vel_pred = np.diff(self.pred, axis=0) * FPS
        errors = np.linalg.norm(vel_pred - vel_gt, axis=-1)
        return np.mean(errors, axis=0) * 100  # convert m/s -> cm/s

    def jitter_by_joint_o(self) -> float:
        """Compute relative jitter between prediction and ground truth."""
        pred = self.pred
        gt = self.gt
        
        # Compute third-order difference
        diff_pred = (
            pred[3:] - 3 * pred[2:-1] + 3 * pred[1:-2] - pred[:-3]
        ) * (FPS ** 3)
        diff_gt = (
            gt[3:] - 3 * gt[2:-1] + 3 * gt[1:-2] - gt[:-3]
        ) * (FPS ** 3)

        jitter_pred_norm = np.linalg.norm(diff_pred, axis=-1)
        jitter_gt_norm = np.linalg.norm(diff_gt, axis=-1)

        jitter_gt_norm = np.mean(jitter_gt_norm, axis=0)
        jitter_pred_norm = np.mean(jitter_pred_norm, axis=0)

        return jitter_pred_norm/jitter_gt_norm

    def jitter_by_joint_v(self) -> np.ndarray:
        """Compute relative jitter between prediction and ground truth for visible frames ONLY."""
        pred = self.pred
        gt = self.gt
        conf = self.conf_pred if self.conf_pred is not None else np.ones((self.gt.shape[0], self.gt.shape[1], 2))

        # Compute third-order difference
        diff_pred = (pred[3:] - 3 * pred[2:-1] + 3 * pred[1:-2] - pred[:-3]) * (FPS ** 3)
        diff_gt = (gt[3:] - 3 * gt[2:-1] + 3 * gt[1:-2] - gt[:-3]) * (FPS ** 3)

        jitter_pred_norm = np.linalg.norm(diff_pred, axis=-1)
        jitter_gt_norm = np.linalg.norm(diff_gt, axis=-1)

        visible = (conf[3:] >= 0.5).all(axis=-1) & (conf[2:-1] >= 0.5).all(axis=-1) & (conf[1:-2] >= 0.5).all(axis=-1) & (conf[:-3] >= 0.5).all(axis=-1)
        
        mean_jitter_pred = np.nanmean(np.where(visible, jitter_pred_norm, np.nan), axis=0)
        mean_jitter_gt = np.nanmean(np.where(visible, jitter_gt_norm, np.nan), axis=0)

        return mean_jitter_pred / mean_jitter_gt

    def jitter_by_joint_m(self) -> np.ndarray:
        """Compute relative jitter between prediction and ground truth for occluded frames ONLY."""
        pred = self.pred
        gt = self.gt
        conf = self.conf_pred if self.conf_pred is not None else np.ones((self.gt.shape[0], self.gt.shape[1], 2))

        # Compute third-order difference
        diff_pred = (pred[3:] - 3 * pred[2:-1] + 3 * pred[1:-2] - pred[:-3]) * (FPS ** 3)
        diff_gt = (gt[3:] - 3 * gt[2:-1] + 3 * gt[1:-2] - gt[:-3]) * (FPS ** 3)

        jitter_pred_norm = np.linalg.norm(diff_pred, axis=-1)
        jitter_gt_norm = np.linalg.norm(diff_gt, axis=-1)

        # Occluded in at least one of the four frames
        missing = ~((conf[3:] >= 0.5).all(axis=-1) & (conf[2:-1] >= 0.5).all(axis=-1) & (conf[1:-2] >= 0.5).all(axis=-1) & (conf[:-3] >= 0.5).all(axis=-1))
        
        mean_jitter_pred = np.nanmean(np.where(missing, jitter_pred_norm, np.nan), axis=0)
        mean_jitter_gt = np.nanmean(np.where(missing, jitter_gt_norm, np.nan), axis=0)

        return mean_jitter_pred / mean_jitter_gt

