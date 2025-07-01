
from torch.utils.data import Dataset
from typing import List, Tuple, Iterable
import numpy as np
import os
import pathlib 
import torch
import enum

# Internal
import config

FPS = 60.0

class Gender(enum.IntEnum):
    MALE = 0
    FEMALE = 1

class SmplxJoints(enum.IntEnum):
    PELVIS = 0  # The root joint for which the model moves about via 'global_orient'.
    LEFT_HIP = 1  # The remaining joints are governed by 'body_pose' (and indirectly 'betas')
    RIGHT_HIP = 2
    SPINE_1 = 3
    LEFT_KNEE = 4
    RIGHT_KNEE = 5
    SPINE_2 = 6
    LEFT_ANKLE = 7
    RIGHT_ANKLE = 8
    SPINE_3 = 9
    LEFT_FOOT = 10
    RIGHT_FOOT = 11
    NECK = 12
    LEFT_COLLAR = 13
    RIGHT_COLLAR = 14
    HEAD = 15
    LEFT_SHOULDER = 16
    RIGHT_SHOULDER = 17
    LEFT_ELBOW = 18
    RIGHT_ELBOW = 19
    LEFT_WRIST = 20
    RIGHT_WRIST = 21
    NUM_JTS = 22  # include pelvis

SMPLX_BODY_HIERARCHY = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)

SMPLX_UPPER_JOINTS = [
    SmplxJoints.SPINE_1,
    SmplxJoints.SPINE_2,
    SmplxJoints.SPINE_3,
    SmplxJoints.NECK,
    SmplxJoints.LEFT_COLLAR,
    SmplxJoints.RIGHT_COLLAR,
    SmplxJoints.HEAD,
    SmplxJoints.LEFT_SHOULDER,
    SmplxJoints.RIGHT_SHOULDER,
    SmplxJoints.LEFT_ELBOW,
    SmplxJoints.RIGHT_ELBOW,
    SmplxJoints.LEFT_WRIST,
    SmplxJoints.RIGHT_WRIST
]

SMPLX_LOWER_JOINTS = [
    SmplxJoints.PELVIS,
    SmplxJoints.LEFT_HIP,
    SmplxJoints.RIGHT_HIP,
    SmplxJoints.LEFT_KNEE,
    SmplxJoints.RIGHT_KNEE,
    SmplxJoints.LEFT_ANKLE,
    SmplxJoints.RIGHT_ANKLE,
    SmplxJoints.LEFT_FOOT,
    SmplxJoints.RIGHT_FOOT,
]
#YOLO
class YoloJoints(enum.IntEnum):
    """Enum mapping YOLO pose estimation joint names to their indices"""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    NUM_JTS = 17

assert os.path.isdir(config.DATA_DIR), f"{config.DATA_DIR} is not a directory"

def get_protocol1_relative_recording_paths(use_cmu: bool = True, use_hdm05: bool = True, use_bmlrub: bool = True)-> List[str]:
    data_dir = pathlib.Path(config.DATA_DIR)
    return  list(data_dir.glob('*/*/**/*.pkl'))

def get_protocol1_split_relative_paths(split:str, verbose:bool=True)-> Tuple:
    assert split in ('train', 'test')
    data_dir = pathlib.Path(config.DATA_DIR)
    return list(data_dir.glob(f"*/{split}/**/*.pkl"))

def load_gt(relative_rec_path:str, fps:int=60):
    data = np.load(relative_rec_path, allow_pickle=True)
    return data

def load_kp(relative_rec_path: str):
    """
    Given a .pkl path, loads the corresponding .npz file.
    """
    npz_path = pathlib.Path(relative_rec_path).with_suffix('.npz')
    data = np.load(npz_path, allow_pickle=True)
    return data

def get_protocol2_split_relative_paths(relative_rec_path:str):
    raise NotImplementedError

def get_dataset_recording_names_for_split(dataset_str: str, split: str):
    # TODO Consider returning to random splits?
    if dataset_str == 'amass-p1':
        # For amass, the recording names are relative file paths of recordings
        if split != 'full':
            rec_names = get_protocol1_split_relative_paths(split)
        else:
            rec_names = get_protocol1_relative_recording_paths()
    elif dataset_str == 'amass-p2':
        rec_names = get_protocol2_split_relative_paths(split)
    elif dataset_str == 'egobody':
        import egobody
        rec_names = egobody.get_recording_names()
        if split != 'full':
            rec_names = list(filter(lambda rn: egobody.get_recording_by_name(rn).split == split, rec_names))
    else:
        raise NotImplementedError(f"Dataset {dataset_str} not supported.")
    
    return rec_names

def get_dataset(dataset_str: str, split: str, ratio: float = None, **dataset_args) -> Dataset:
    rec_names = get_dataset_recording_names_for_split(dataset_str, split)

    if ratio is not None:
        assert 0.0 < ratio < 1.0
        num_recs = len(rec_names)
        new_num_recs = int(num_recs * ratio)
        assert new_num_recs > 0, f"provided 'ratio' too small for number of recordings ({num_recs})"
        rec_names = rec_names[:new_num_recs]

    if dataset_str in ('amass-p1', 'amass-p2'):
        
        dataset = AMASSDataset(rec_names, **dataset_args)
    elif dataset_str == 'egobody':
        import egobody
        dataset = egobody.EgoBodyDataset(rec_names, **dataset_args)
    else:
        raise NotImplementedError(f"Dataset '{dataset_str}' not supported.")
    
    return dataset

class AMASSDataset(Dataset):
    def __init__(self, 
        relative_recording_paths: Iterable[str],
        win_len:int         = 40,
        win_overlap:int     = 5,
        zero_betas:bool     = True, #zero_betas means average morpholgy
        dtype:torch.dtype   = torch.float32,
        phase:str           = 'train'
    ):
        assert win_len > 0
        assert 0 <= win_overlap < win_len
        assert phase in ('train', 'valid', 'test'), f"Provide a valid phase, got {phase}"

        self.phase = phase 

        self._rotations_local_full_gt_list = []
        self._hmd_position_global_full_gt_list = []
        self._head_global_trans_list = []
        self._betas_wins = []
        self._gender = []
        #self._framerate = []
        #self._filepath = []
        self._body_parms_list = []
        self._external_3d_kp = []
        self._external_conf = []
        # 
        # --- Train ---
        # 
        if self.phase in ('train', 'valid'):

            win_step = win_len - win_overlap

            for relative_rec_path in relative_recording_paths:
                data_gt = load_gt(relative_rec_path) #get sparse signals from VR
                data_kp = load_kp(relative_rec_path) #get 3D keypoints

                assert len(data_gt['hmd_position_global_full_gt_list']) == len(data_kp['points3d']), "Length mismatch between keypoints and ground truth"

                num_frames = data_gt['hmd_position_global_full_gt_list'].shape[0]
                
                # gt
                rotation_local_full_gt_list = data_gt['rotation_local_full_gt_list'].cpu().clone()
                hmd_position_global_full_gt_list = data_gt['hmd_position_global_full_gt_list'].cpu().clone()
                head_global_trans_list = data_gt['head_global_trans_list'].cpu().clone()
                betas = torch.tensor(data_gt["shape"][None,:].repeat(num_frames, axis=0), dtype=dtype)
                body_parms_list = data_gt['body_parms_list']

                #3D keypoints from pose estimation
                keypoints = torch.tensor(data_kp['points3d'], dtype=dtype) 
                conf_scores = torch.tensor(data_kp['conf'], dtype=dtype)

                if num_frames < win_len:
                    continue
                for start_frame in range(0, num_frames, win_step):
                    end_frame = start_frame + win_len
                    if end_frame > num_frames:
                        offset = end_frame - num_frames
                        start_frame -= offset
                        end_frame -= offset

                    self._rotations_local_full_gt_list.append(rotation_local_full_gt_list[start_frame:end_frame])
                    self._hmd_position_global_full_gt_list.append(hmd_position_global_full_gt_list[start_frame:end_frame])
                    self._head_global_trans_list.append(head_global_trans_list[start_frame:end_frame])
                    self._betas_wins.append(betas[start_frame:end_frame])  # expand to make consistent with EgoBody
                    self._external_3d_kp.append(keypoints[start_frame:end_frame])
                    self._external_conf.append(conf_scores[start_frame:end_frame])

                    #self._gender.append(data_gt['gender'])
                    #self._framerate.append(data_gt['framerate'])
                    #self._filepath.append(data_gt['filepath'])
                    self._body_parms_list.append(body_parms_list)

                    if data_gt['gender'] == 'male':
                        self._gender.append(torch.tensor(Gender.MALE, dtype=dtype))
                    elif data_gt['gender'] == 'female':
                        self._gender.append(torch.tensor(Gender.FEMALE, dtype=dtype))

            self._num_wins = len(self._betas_wins)
            self._win_len = win_len
            self._zero_betas = zero_betas

        # 
        # --- Test ---
        # 
        elif self.phase == 'test':

            for relative_rec_path in relative_recording_paths:
                data_gt = load_gt(relative_rec_path) #get sparse signals from VR
                data_kp = load_kp(relative_rec_path) #get 3D keypoints

                assert len(data_gt['hmd_position_global_full_gt_list'].shape[0]) == len(data_kp['points3d']), "Length mismatch between keypoints and ground truth"

                num_frames = data_gt['hmd_position_global_full_gt_list'].shape[0]
                
                if data_gt['gender'] == 'male':
                    self._gender.append(torch.tensor(Gender.MALE, dtype=dtype))
                elif data_gt['gender'] == 'female':
                    self._gender.append(torch.tensor(Gender.FEMALE, dtype=dtype))

                rotation_local_full_gt_list = data_gt['rotation_local_full_gt_list'].cpu().clone()
                hmd_position_global_full_gt_list = data_gt['hmd_position_global_full_gt_list'].cpu().clone()
                head_global_trans_list = data_gt['head_global_trans_list'].cpu().clone()
                betas = torch.tensor(data_gt["shape"][None,:].repeat(num_frames, axis=0), dtype=dtype)
                body_parms_list = data_gt['body_parms_list']

                #3D keypoints from pose estimation
                keypoints = data_kp['points3d']
                conf_scores = data_kp['conf']
                self._external_3d_kp.append(keypoints)
                self._external_conf.append(conf_scores)

                self._rotations_local_full_gt_list.append(rotation_local_full_gt_list)
                self._hmd_position_global_full_gt_list.append(hmd_position_global_full_gt_list)
                self._head_global_trans_list.append(head_global_trans_list)
                self._betas_wins.append(betas)  # expand to make consistent with EgoBody
                self._body_parms_list.append(body_parms_list)

            self._num_wins = len(self._betas_wins)
            self._win_len = win_len
            self._zero_betas = zero_betas

    def __len__(self):
        return self._num_wins

    def __getitem__(self, idx):
        betas = self._betas_wins[idx].clone()
        body_parms_list = self._body_parms_list[idx] if self.phase == 'test' else -1
        if self._zero_betas:
            betas = torch.zeros_like(betas)
        return {
            'rotations_local_full_gt_list' : self._rotations_local_full_gt_list[idx].clone(),
            'hmd_position_global_full_gt_list': self._hmd_position_global_full_gt_list[idx].clone(),
            'head_global_trans_list': self._head_global_trans_list[idx].clone(),
            'betas': betas, 
            'gender': self._gender[idx].clone(),
            'keypoints': self._external_3d_kp[idx].clone(),
            'conf':self._external_conf[idx].clone(),
            #'framerate' : self._framerate[idx],
            #'filepath': self._filepath[idx],
            'body_parms_list': body_parms_list
        }
    
