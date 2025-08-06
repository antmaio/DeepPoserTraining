
from torch.utils.data import Dataset
from typing import List, Tuple, Iterable, Union
import numpy as np
import os
import pathlib 
import torch
import enum
import logging
# Internal
import config

__CACHE_DIR = os.path.join(config.CACHE_DIR, 'hmd-poser-ext-amass')

__CMU_RELATIVE_DIR = "CMU"
__HDM05_RELATIVE_DIR = "MPI_HDM05"
__BMLRUB_RELATIVE_DIR = "BioMotionLab_NTroje"

__SUBSETS = (__CMU_RELATIVE_DIR, __HDM05_RELATIVE_DIR, __BMLRUB_RELATIVE_DIR)

__DATASET_DIR_MAP = {
    'cmu': __CMU_RELATIVE_DIR,
    'hdm05': __HDM05_RELATIVE_DIR,
    'bml_rub': __BMLRUB_RELATIVE_DIR
}

__3D_MODE__ = ('triang', 'openmpl') 


FPS = 60.0




# ------
# --- SMPL ---
# ------

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

# ------
# --- YOLO ---
# ------

class YoloJoints(enum.IntEnum):
    """Enum mapping YOLO pose estimation joint names to their indices"""
    NOSE = 0
    LEFT_EYE = 1 #avoid feeding model with that
    RIGHT_EYE = 2 #avoid feeding model with that
    LEFT_EAR = 3 #avoid feeding model with that
    RIGHT_EAR = 4 #avoid feeding model with that
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

YOLO_UPPER_JOINTS = [
    YoloJoints.NOSE,
    YoloJoints.LEFT_SHOULDER,
    YoloJoints.RIGHT_SHOULDER,
    YoloJoints.LEFT_ELBOW,
    YoloJoints.RIGHT_ELBOW,
    YoloJoints.LEFT_WRIST,
    YoloJoints.RIGHT_WRIST
]

YOLO_LOWER_JOINTS = [
    YoloJoints.LEFT_HIP,
    YoloJoints.RIGHT_HIP,
    YoloJoints.LEFT_KNEE,
    YoloJoints.RIGHT_KNEE,
    YoloJoints.LEFT_ANKLE,
    YoloJoints.RIGHT_ANKLE
]

def get_protocol1_relative_recording_paths(config, use_cmu: bool = True, use_hdm05: bool = True, use_bmlrub: bool = True)-> List[str]:
    data_dir = pathlib.Path(config.DATA_DIR)
    return  list(data_dir.glob('*/*/**/*.pkl'))

def get_protocol1_split_relative_paths(config, split:str, verbose:bool=True)-> Tuple:
    assert split in ('train', 'valid', 'test')
    data_dir = pathlib.Path(config.DATA_DIR)
    split = 'test' if split == 'valid' else split #take test dataset as validation, but segmented of by chunks of size win_len
    return list(data_dir.glob(f"*/{split}/**/*.pkl"))

def get_protocol2_split_relative_paths(config, split: str):
    assert split in ('train', 'valid', 'test'), f"Invalid split: {split}"
    assert config.AS_TESTSET in ('cmu', 'bml_rub', 'hdm05'), f"Dataset {config.AS_TESTSET} not implemented"

    data_dir = pathlib.Path(config.DATA_DIR)

    # Use test set as validation set (common in cross-dataset protocols)
    actual_split = 'test' if split in ('test', 'valid') else 'train'

    if actual_split == 'test':
        logging.info(f'test set: {__DATASET_DIR_MAP[config.AS_TESTSET]}')
        return list(data_dir.glob(f"{__DATASET_DIR_MAP[config.AS_TESTSET]}/**/*.pkl"))

    
    # For 'train': return data from the other two datasets
    train_set = []
    for name, path in __DATASET_DIR_MAP.items():
        if name != config.AS_TESTSET:
            logging.info(f'train set: {path}')
            train_set.extend(data_dir.glob(f"{path}/**/*.pkl"))
    return train_set

def load_gt(relative_rec_path:str, fps:int=60):
    data = np.load(relative_rec_path, allow_pickle=True)
    return data

def load_kp(config, relative_rec_path: str):
    """
    Given a .pkl path, loads the corresponding .npz file.
    
    """
    assert config.MODE in __3D_MODE__, f"mode {config.MODE} is not a valid 3D pose lifter, please provide mode in {__3D_MODE__}"
    npz_path = pathlib.Path(relative_rec_path)
    # data/keypoints/preprocessed/1.pkl -> data/keypoints/{mode}/1.npz
    base_name = relative_rec_path.stem + ".npz"
    base_dir = relative_rec_path.parents[1]
    npz_path = base_dir / config.MODE / base_name
    data = np.load(npz_path, allow_pickle=True)
    return data



def get_dataset_recording_names_for_split(config, dataset_str: str, split: str):
    # TODO Consider returning to random splits?
    if dataset_str == 'amass-p1':
        # For amass, the recording names are relative file paths of recordings
        if split != 'full':
            rec_names = get_protocol1_split_relative_paths(config, split)
        else:
            rec_names = get_protocol1_relative_recording_paths(config)
    elif dataset_str == 'amass-p2':
        assert config.AS_TESTSET is not None, f"Please provide valid testset, not {config.AS_TESTSET}"
        rec_names = get_protocol2_split_relative_paths(config, split)
    elif dataset_str == 'egobody':
        import egobody
        rec_names = egobody.get_recording_names()
        if split != 'full':
            rec_names = list(filter(lambda rn: egobody.get_recording_by_name(rn).split == split, rec_names))
    else:
        raise NotImplementedError(f"Dataset {dataset_str} not supported.")
    
    return rec_names

def get_dataset(config, dataset_str: str, split: str, ratio: float = None, **dataset_args) -> Dataset:
    rec_names = get_dataset_recording_names_for_split(config, dataset_str, split)

    if ratio is not None:
        assert 0.0 < ratio < 1.0
        num_recs = len(rec_names)
        new_num_recs = int(num_recs * ratio)
        assert new_num_recs > 0, f"provided 'ratio' too small for number of recordings ({num_recs})"
        rec_names = rec_names[:new_num_recs]

    if dataset_str in ('amass-p1', 'amass-p2'):
        dataset = AMASSDataset(config, rec_names, phase=split, **dataset_args)
    elif dataset_str == 'egobody':
        import egobody
        dataset = egobody.EgoBodyDataset(rec_names, **dataset_args)
    else:
        raise NotImplementedError(f"Dataset '{dataset_str}' not supported.")
    
    return dataset

def load_smpl(config, relative_rec_path: Union[str, pathlib.Path]) -> dict:

    #only used to load test data 

    os.makedirs(__CACHE_DIR, exist_ok=True)

    # Determine cached version name
    if type(relative_rec_path) is not str:
        relative_rec_path = str(relative_rec_path) 
    cached_name = relative_rec_path.replace(os.sep, '-').replace('.npz', '') + '.pt'
    cached_path = os.path.join(__CACHE_DIR, cached_name)

    # Load from cache if entry exists
    if os.path.exists(cached_path):
        cached = torch.load(cached_path, weights_only=True)
        rotations_local_full_gt_list = cached['rotations_local_full_gt_list']
        hmd_position_global_full_gt_list = cached['hmd_position_global_full_gt_list']
        head_global_trans_list = cached['head_global_trans_list']
        betas = cached['betas']
        gender = cached['gender']
        keypoints = cached['keypoints']
        body_parms_list = cached['body_parms_list']
        conf_scores = cached.get('conf')
        if isinstance(conf_scores, torch.Tensor): 
            out_conf = conf_scores.clone()

    else:  # Otherwise load, compute and update cache
        # Load from dataset
        rec_path = os.path.join(config.AMASS_DIR, relative_rec_path)
        try:
            data_gt = load_gt(relative_rec_path) #get sparse signals from VR
            data_kp = load_kp(config, pathlib.Path(relative_rec_path)) #get 3D keypoints
        except Exception as e: # TODO remove
            print(e)
            breakpoint()

        # gt
        rotations_local_full_gt_list = data_gt['rotation_local_full_gt_list'].cpu().clone()
        hmd_position_global_full_gt_list = data_gt['hmd_position_global_full_gt_list'].cpu().clone()
        head_global_trans_list = data_gt['head_global_trans_list'].cpu().clone()
        num_frames = len(rotations_local_full_gt_list)
        betas = torch.tensor(data_gt["shape"][None,:].repeat(num_frames, axis=0), dtype=torch.float32)
        body_parms_list = data_gt['body_parms_list']

        #3D keypoints from pose estimation
        keypoints = torch.tensor(data_kp['points3d'], dtype=torch.float32)
        conf_scores = data_kp.get('conf')
        if isinstance(conf_scores, np.ndarray): 
            conf_scores = torch.tensor(conf_scores, dtype=torch.float32)
            out_conf = conf_scores.clone()
        else: 
            out_conf = None
        if data_gt['gender'] == 'male':
            gender = torch.tensor(Gender.MALE, dtype=torch.float32)
        elif data_gt['gender'] == 'female':
            gender = torch.tensor(Gender.FEMALE, dtype=torch.float32)

        # Update cache; the cache is a bit wasteful, but we're prioritising speed
        cached = {
            'rotations_local_full_gt_list' : rotations_local_full_gt_list.clone(),
            'hmd_position_global_full_gt_list': hmd_position_global_full_gt_list.clone(),
            'head_global_trans_list': head_global_trans_list.clone(),
            'betas': betas.clone(), 
            'gender': gender.clone(),
            'keypoints': keypoints.clone(),
            #'conf': out_conf,
            #'framerate' : self._framerate[idx],
            #'filepath': self._filepath[idx],
            'body_parms_list': body_parms_list
        }
        if out_conf is not None:
            cached['conf'] = out_conf

        os.makedirs(__CACHE_DIR, exist_ok=True)
        torch.save(cached, cached_path)

    out_dict = {
        'rotations_local_full_gt_list' : rotations_local_full_gt_list.clone(),
        'hmd_position_global_full_gt_list': hmd_position_global_full_gt_list.clone(),
        'head_global_trans_list': head_global_trans_list.clone(),
        'betas': betas.clone(), 
        'gender': gender.clone(),
        'keypoints': keypoints.clone(),
        #'conf': out_conf,
        #'framerate' : self._framerate[idx],
        #'filepath': self._filepath[idx],
        'body_parms_list': body_parms_list
    }
    
    if out_conf is not None:
        out_dict['conf'] = out_conf

    return out_dict
    
    '''
        betas = torch.tensor(rec['betas'], dtype=torch.float32)
        transl = torch.tensor(rec['trans'], dtype=torch.float32)
        global_orient_aa = torch.tensor(rec['root_orient'], dtype=torch.float32)
        body_pose_aa = torch.tensor(rec['pose_body'], dtype=torch.float32)
        frame_rate = rec['mocap_frame_rate']

        # Downsample to 60 fps
        source_fps_is_120 = np.isclose(frame_rate, 120)
        assert source_fps_is_120 or np.isclose(frame_rate, 60), \
            f"mocap_frame_rate must be 60 or 120 but is {frame_rate} ({rec_path})"
        if source_fps_is_120:
            transl = transl[::2]
            global_orient_aa = global_orient_aa[::2]
            body_pose_aa = body_pose_aa[::2]

        # Convert from angle axis to rotation matrix
        global_orient = utils.angle_axis_to_matrix(global_orient_aa)
        body_pose = utils.angle_axis_to_matrix(body_pose_aa.reshape(transl.shape[0], -1, 3))

        # Modify transl and global_orient such that coordinate system is y-up
        to_world = torch.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32)
        with torch.no_grad():
            rest_joints = data.get_smpl_or_smplx_rest_joints(SMPLX_LAYER_NEUTRAL, betas[None])
        hip_jt = rest_joints[:, 0] + transl
        root = torch.zeros((transl.shape[0], 4, 4), dtype=torch.float32)
        root[:, :3, :3] = global_orient
        root[:, :3, 3] = hip_jt
        root[:, 3, 3] = 1.0
        root_world = to_world @ root
        global_orient = root_world[:, :3, :3]
        transl = root_world[:, :3, 3] + transl - hip_jt

        # Update cache; the cache is a bit wasteful, but we're prioritising speed
        cached = {
            'betas': betas,
            'transl': transl,
            'global_orient': global_orient,
            'body_pose': body_pose
        }
        os.makedirs(__CACHE_DIR, exist_ok=True)
        torch.save(cached, cached_path)

    if fps == 30:
        transl = transl[::2]
        global_orient = global_orient[::2]
        body_pose = body_pose[::2]

    return {
        'betas': betas[None].expand(transl.shape[0], -1),  # add temporal dimension
        'transl': transl,
        'global_orient': global_orient,
        'body_pose': body_pose
    }
    '''

class AMASSDataset(Dataset):
    def __init__(self, 
        config, 
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
                data_kp = load_kp(config, relative_rec_path) #get 3D keypoints

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
                conf_scores = data_kp.get('conf')
                if isinstance(conf_scores, np.ndarray):  conf_scores = torch.tensor(data_kp['conf'], dtype=dtype)
                
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
                    if isinstance(conf_scores, torch.Tensor): 
                        self._external_conf.append(conf_scores[start_frame:end_frame])
                    else:
                        self._external_conf.append(None)


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
                data_kp = load_kp(config, relative_rec_path) #get 3D keypoints

                assert len(data_gt['hmd_position_global_full_gt_list']) == len(data_kp['points3d']), "Length mismatch between keypoints and ground truth"

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
                keypoints = torch.tensor(data_kp['points3d'], dtype=dtype) 
                conf_scores = data_kp.get('conf')
                if isinstance(conf_scores, np.ndarray):  conf_scores = torch.tensor(data_kp['conf'], dtype=dtype)
                
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
        out_conf = self._external_conf[idx].clone() if isinstance(self._external_conf[idx], torch.Tensor) else None

        out_dict = {
            'rotations_local_full_gt_list' : self._rotations_local_full_gt_list[idx].clone(),
            'hmd_position_global_full_gt_list': self._hmd_position_global_full_gt_list[idx].clone(),
            'head_global_trans_list': self._head_global_trans_list[idx].clone(),
            'betas': betas, 
            'gender': self._gender[idx].clone(),
            'keypoints': self._external_3d_kp[idx].clone(),
            #'conf': out_conf,
            #'framerate' : self._framerate[idx],
            #'filepath': self._filepath[idx],
            'body_parms_list': body_parms_list
        }
        
        if out_conf is not None:
            out_dict['conf'] = out_conf 
    
        return out_dict
    
