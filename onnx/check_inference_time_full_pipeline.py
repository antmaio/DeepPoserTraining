#External
import ultralytics
import logging  
import typing
import argparse
import cv2
import numpy as np 
import os
from pathlib import Path
import time
import glob
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import tomli
import abc

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

#Internal
from anim.models.hmd_poser_ext import HMDPoserExt
from anim.data.amass import SmplxJoints, YoloJoints

from data.utils_yolo import init_yolo
from data.camera_config import CAMERA_PATH_PROTOCOL_1, CAMERA_PATH_PROTOCOL_2, CAMERA_PATH_PROTOCOL_3

from utils.utils_transform import two_axis_to_matrix_onnx_friendly, matrix_to_two_axis, matrix_to_angle_axis_onnx_friendly

from onnx.__config_mpl import config, update_config, update_dir, get_model_name
from onnx._multiview_inference_openmplposer_rumpl import MultiViewInference_OpenMPLPoser_RUMPL 
import onnx.onnx as ConversionONNX

# Enable anomaly detection to see where the operation occurs
# torch.autograd.set_detect_anomaly(True)



# fixed
__NCAM__            = 3
__NJOINTS__         = 17
__COMPILE_MODE__    = None
__NUM_BETAS__       = 16 # number of body parameters
__NUM_DMPLS__       = 8 # number of DMPL parameters
__SMPLX_JOINTS__    = 22
# modify 
CONFIG_FILE     = '../OpenMPL_private/RUMPL/configs/openmplposer/rumpl_amass_poser_3_openmplposer_aligned_yolo8/rumpl_301_amass_yolo_ConfConcat_3viewsV1V2V3_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV0.yaml'
TRAINED_MODEL   = './pretrained/mpl/yolov8n-pose/model_best_yolov8_p1.pth.tar' 
YOLO_MODEL      = 'yolov8n-pose'
ONNX_PATH       = "onnx"
USE_ONNX_INF    = True
ONNX_CONVERSION = False
CAMERA_PATH     = CAMERA_PATH_PROTOCOL_1
MODEL_DIR       = './saves/hmd-poser-ext_25-9-1-10-39-54'
CHECKPOINT      = 400


# Overwrite log file every time the script runs
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --- from openmpl ---
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument(
        '--cfg', help='experiment configure file name', type=str)
    args, rest = parser.parse_known_args()
    args.cfg = CONFIG_FILE
    
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument(
        '--frequent',
        help='frequency of logging',
        default=config.PRINT_FREQ,
        type=int)
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument(
        '--state',
        help='the state of model which is used to test (best or final)',
        default='best',
        type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--model-file', help='model state file', type=str)
    parser.add_argument(
        '--flip-test', help='use flip test', action='store_true')
    parser.add_argument(
        '--post-process', help='use post process', action='store_true')
    parser.add_argument(
        '--shift-heatmap', help='shift heatmap', action='store_true')

    # philly
    parser.add_argument(
        '--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument(
        '--dataDir', help='data directory', type=str, default='')
    parser.add_argument(
        '--data-format', help='data format', type=str, default='')
    parser.add_argument(
        '--enable-wandb', help='enable wandb', action='store_true', default=False
    )
    parser.add_argument(
        '--test-dataset', help='test dataset', type=str, default=None
    )
    parser.add_argument(
        '--use-mmpose-val', help='use mmpose val', action='store_true', default=None
    )
    parser.add_argument(
        '--not-use-mmpose-val', help='not use mmpose val', action='store_false', dest='use_mmpose_val',default=None
    )
    parser.add_argument(
        '--batch-size', help='batch-size', type=int, default=None
    )
    parser.add_argument(
        '--filter-cmu-wrong-cases', help='filter cmu wrong cases', action='store_true', default=False
    )
    parser.add_argument(
        '--test-cmu-dataset-name', help='test cmu dataset name', type=str, default=None
    )
    parser.add_argument(
        '--test-mmpose-type', help='test mmpose type', type=str, default=None
    )
    parser.add_argument(
        '--n-samples', help='n samples', type=int, default=None
    )
    parser.add_argument(
        '--cameras-path', help='path to camera xml files', type=str, default=''
    )
    parser.add_argument(
        '--data-dir', help='inference data directory', type=str, default=''
    )
    parser.add_argument(
        '--data-out-dir', help='inference data output directory', type=str, default=''
    )
    parser.add_argument(
        '--trained-model', help='path to trained model', type=str, default=None
    )
    

    args = parser.parse_args()

    args.cfg = CONFIG_FILE
    args.trained_model = TRAINED_MODEL
    args.cameras_path = CAMERA_PATH

    update_dir(args.modelDir, args.logDir, args.dataDir)
    return args

def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.data_format:
        config.DATASET.DATA_FORMAT = args.data_format
    if args.workers:
        config.WORKERS = args.workers
    if args.flip_test:
        config.TEST.FLIP_TEST = args.flip_test
    if args.post_process:
        config.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.state:
        config.TEST.STATE = args.state
    if not args.enable_wandb:
        config.WANDB = False
    if args.use_mmpose_val is not None:
        config.DATASET.USE_MMPOSE_VAL = args.use_mmpose_val
    if args.test_dataset is not None:
        config.DATASET.TEST_DATASET = args.test_dataset
    if args.batch_size is not None:
        config.TEST.BATCH_SIZE = args.batch_size
    if args.filter_cmu_wrong_cases:
        config.DATASET.TEST_FILTER_CMU_WRONG_CASES = args.filter_cmu_wrong_cases
    if args.test_cmu_dataset_name is not None:
        config.DATASET.TEST_CMU_DATASET_NAME = args.test_cmu_dataset_name
    if args.test_mmpose_type is not None:
        config.DATASET.TEST_MMPOSE_TYPE = args.test_mmpose_type
    if args.n_samples is not None:
        config.DATASET.TEST_N_SAMPLES = args.n_samples
    if args.cameras_path is not None:
        config.DATASET.INFERENCE_CAMERAS_PATH = args.cameras_path
    if args.data_dir is not None:
        config.DATASET.INFERENCE_DATA_DIR = args.data_dir

    #HACK Added this but not used to print log
    if args.data_out_dir is not None:
        config.OUTPUT_DIR = '/home/antoine/These/OpenMPLPoser/temp'

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        os.makedirs(root_output_dir, exist_ok=True)
        # root_output_dir.mkdir()

    dataset = cfg.DATASET.TRAIN_DATASET
    model, _ = get_model_name(cfg)
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / model / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
        (cfg_name + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def inference_mpl(config, model, input_mpl, onnx_inferencer):

    n_view = 6 if config.DATASET.TEST_DATASET == 'multiview_skipose' else 4
    n_view = 5 if config.DATASET.TEST_DATASET.startswith('multiview_cmu_panoptic') else n_view
    n_view = len(config.DATASET.TEST_VIEWS) if config.DATASET.TEST_VIEWS is not None else n_view
    
    #__check_shape(input_mpl)

    middle_points, closest_points_all, target, rays, meta, joints_2ds = input_mpl

    if config.NETWORK.APPLY_VIEW_FUSION:
        if onnx_inferencer is not None:
            output = onnx_inferencer(rays.unsqueeze(0))
        else:
            output = model(rays.unsqueeze(0), is_training=False) #unsqueeze to create batch dimension
    else:
        raise NotImplementedError

    return output

# --- from hmd-poser ---
@dataclass
class BaseModelInput:
    batch_size: int
    win_len: int
    head_pos_global: torch.Tensor  # (batch_size, win_len, 3)
    lh_pos_global: torch.Tensor  # ditto
    rh_pos_global: torch.Tensor  # ditto
    head_rot_global: torch.Tensor  # (batch_size, win_len, 3, 3)
    lh_rot_global: torch.Tensor  # ditto
    rh_rot_global: torch.Tensor  # ditto
    # TODO Weaken the below assumption that we only use HMR to allow for MediaPipe compatibility
    # hmr_presence: torch.BoolTensor  # TODO Utilise for sparsity
    hmr_joints: torch.Tensor  # (batch_size, win_len, SmplxJoints.NUM_JTS, 3) or (batch_size, win_len, YoloJoints.NUM_JTS, 3) 
    hmr_body_pose: torch.Tensor  # (batch_size, win_len, SmplxJoints.NUM_JTS - 1, 3, 3)
    hmr_global_orient: torch.Tensor  # (batch_size, win_len, 3, 3)
    #Optional
    betas: typing.Optional[torch.Tensor]#(batch_size, win_len, num_betas)
    gender: typing.Optional[int] #(batch_size,)
    conf: typing.Optional[torch.Tensor] #(batch_size, win_len, YoloJoints.NUM_JTS, 2)
@dataclass
class BaseModelOutput:
    # betas for SMPLX model, can be omitted to signal different model (e.g. aligner)
    betas: typing.Optional[torch.Tensor]
    # TODO Do we make other params optional?
    #transl: torch.Tensor  # (batch_size, win_len, 3)
    global_orient: torch.Tensor  # (batch_size, win_len, 3, 3)
    body_pose: torch.Tensor  # (batch_size, win_len, SmplxJoints.NUM_JTS-1, 3, 3)
    # Redundant but packing in here anyway for convenience:
    joints: torch.Tensor   # (batch_size, win_len, SmplxJoints.NUM_JTS, 3)
    #vertices: torch.Tensor
    #faces: list[int]  # for rendering; same set of indices for all vertices
    gender: typing.Optional[int] #(batch_size,)
class BaseModel(torch.nn.Module):
#class BaseModel(torch.jit.ScriptModule):
    """ Base class for synthetic_models """

    @staticmethod
    @abc.abstractmethod
    def model_str() -> str:
        """ Return unique string representing model class; cannot contain underscores """
        pass

    @abc.abstractmethod
    def dataset_pass(self, dataset: Dataset, device, dtype):
        """
        Compute dataset dependent parameters prior to training (e.g. normalisation).
        Only called for fresh models, so copy into torch parameters made in __init__ for them to save!
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """
        Reset any temporal state held by the model.

        Specifically, 'forward' and 'forward_pass' must assume consecutive calls provide consecutive frames.
        Thus, a call to reset signals to the model that new, unrelated, windows of data are inbound.

        For example, we might reset accumulated RNN hidden state.

        To clarify, consider a batch of windows B, containing data with shapes (batch_size, win_len, ...), and that
          we split these windows in consecutive halves B1 and B2, each with data shapes (batch_size, win_len/2, ...).
        Then, calling 'forward(B1)' and then 'forward(B2)', with no intermediate 'reset', is equivalent to 'forward(B)'.
        On the other hand, an intermediate 'reset' here would signal that B1 and B2 are temporally unrelated.
        """
        pass

    @abc.abstractmethod
    def forward(self, model_input: BaseModelInput) -> BaseModelOutput:
        """
        Predict for one batch of input data (BaseModelInput).

        As described by the 'reset' docstring, consecutive calls of 'forward' should be considered
          temporally consecutive, so you should maintain model temporal state (e.g. RNN history).

        Returns prediction (BaseModelInput).
        """
        pass

    # TODO This might be doing too many things? what about a train_pass that accepts model_output, model_target?
    @abc.abstractmethod
    def forward_pass(self, model_input: BaseModelInput, model_target: BaseModelOutput, optimise: bool = False) -> dict:
        """
        Predict on one batch of input data (i.e. forward), compute losses, and update parameters if optimise true.

        This method would typically call 'forward' internally is expected to have the same 'reset' semantics.

        Returns dict with model losses on that batch.
        """
        pass

    @abc.abstractmethod
    def epoch_end(self, epoch: int, train_losses: dict, val_losses: dict):
        """
        Called by training loop after an epoch of training with 'epoch' index passed.
        Train losses and val losses are averaged numpy losses with same keys as returned 'forward_pass'.
        Useful for updating per-epoch state, such as LR schedulers.
        """
        pass

def create_model(model_cfg_path: str) -> BaseModel:

    assert os.path.isfile(model_cfg_path)
    with open(model_cfg_path, 'rb') as fp:
        model_cfg = tomli.load(fp)
    assert 'model_str' in model_cfg, "Model TOML must have 'model_str' entry"
    assert 'model_args' in model_cfg, "Model TOML must have '[model_args]'"
    model_str = model_cfg['model_str']
    model_args = model_cfg['model_args']
    assert '_' not in model_str, 'model_str containing underscore is forbidden'
    return ONNXCompatibleAnimModel(**model_args)

def adapt_ckpt_to_model_key(model, checkpoint):
    #pytorch 2.3.1 -> 2.0.1
    model_keys = sorted(model.state_dict().keys())
    ckpt_keys_raw = sorted(checkpoint.keys())
    suffix = '.parametrizations.weight_norm.original'

    new_state_dict = {}

    for k, v in checkpoint.items():
        if '.parametrizations.' in k and (
            k.endswith('.original0') or k.endswith('.original1')
        ):
            # Figure out if it's original0 or original1
            if k.endswith('.original0'):
                suffix_new = '_g'
                k_base = k[:-len('.parametrizations.weight_hh_l0.original0')]
                if 'weight_hh_l0' in k:
                    k_base = k.replace('.parametrizations.weight_hh_l0.original0', '')
                    new_k = f"{k_base}.weight_hh_l0_g"
                elif 'weight_ih_l0' in k:
                    k_base = k.replace('.parametrizations.weight_ih_l0.original0', '')
                    new_k = f"{k_base}.weight_ih_l0_g"
                else:
                    new_k = k  # fallback, keep as-is
            elif k.endswith('.original1'):
                suffix_new = '_v'
                if 'weight_hh_l0' in k:
                    k_base = k.replace('.parametrizations.weight_hh_l0.original1', '')
                    new_k = f"{k_base}.weight_hh_l0_v"
                elif 'weight_ih_l0' in k:
                    k_base = k.replace('.parametrizations.weight_ih_l0.original1', '')
                    new_k = f"{k_base}.weight_ih_l0_v"
                else:
                    new_k = k
            else:
                new_k = k

            #logging.info(f"Adapting: {k} -> {new_k}")
            new_state_dict[new_k] = v
        else:
            #copy value
            new_state_dict[k] = v
        
    return new_state_dict

def load_model(model_dir: str, epoch: int):

    __MODEL_CONFIG_NAME = 'model_config.toml'
    __MODEL_STATE_DICTS_DIR_NAME = 'model_state_dicts'

    model_state_dict_dir = os.path.join(model_dir, __MODEL_STATE_DICTS_DIR_NAME)
    checkpoint_path = os.path.join(model_state_dict_dir, str(epoch) + '.pt')
    assert os.path.isfile(checkpoint_path)
    model_cfg_path = os.path.join(model_dir, __MODEL_CONFIG_NAME)
    model = create_model(model_cfg_path)
    try:
        logging.info('Loading weights as is...')
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    except:
        logging.info('Adapting keys name while loading ckpt weights...')
        state_dict = adapt_ckpt_to_model_key(model, torch.load(checkpoint_path, weights_only=True))
        #state_dict = _fix_key_names(torch.load(checkpoint_path, weights_only=True))
        model.load_state_dict(state_dict)
    return model


def __create_dummy_BaseModeInput(device, output_mpl=None):

    batch_size, win_len = 1, 1
    """Return as tuple of tensors"""
    if output_mpl is not None:
        return (
            torch.ones((batch_size, win_len, 3), dtype=torch.float32, device=device),  # head_pos_global
            torch.eye(3, dtype=torch.float32, device=device).repeat(batch_size, win_len, 1, 1),  # head_rot_global
            torch.ones((batch_size, win_len, 3), dtype=torch.float32, device=device),  # lh_pos_global
            torch.eye(3, dtype=torch.float32, device=device).repeat(batch_size, win_len, 1, 1),  # lh_rot_global
            torch.ones((batch_size, win_len, 3), dtype=torch.float32, device=device),  # rh_pos_global
            torch.eye(3, dtype=torch.float32, device=device).repeat(batch_size, win_len, 1, 1),  # rh_rot_global
            torch.ones((batch_size, win_len, __NJOINTS__, 3), dtype=torch.float32, device=device),  # hmr_joints
            torch.eye(3, dtype=torch.float32, device=device).repeat(batch_size, win_len, __SMPLX_JOINTS__ - 1, 1, 1),  # hmr_body_pose
            torch.eye(3, dtype=torch.float32, device=device).repeat(batch_size, win_len, 1, 1, 1),  # hmr_global_orient
            torch.ones((batch_size, win_len, __NUM_BETAS__), dtype=torch.float32, device=device),  # betas
            torch.tensor([1.0]*batch_size, dtype=torch.float32, device=device),  # gender - converted to tensor
            torch.ones((batch_size, win_len, __NJOINTS__, 2), dtype=torch.float32, device=device),  # conf
            torch.ones((batch_size, win_len, 3), dtype=torch.float32, device=device),  # head_vel_global
            torch.ones((batch_size, win_len, 3), dtype=torch.float32, device=device),  # lh_vel_global
            torch.ones((batch_size, win_len, 3), dtype=torch.float32, device=device),  # lh_vel_local
            torch.ones((batch_size, win_len, 3), dtype=torch.float32, device=device),  # rh_vel_global
            torch.ones((batch_size, win_len, 3), dtype=torch.float32, device=device),  # rh_vel_local
            torch.eye(3, dtype=torch.float32, device=device).repeat(batch_size, win_len, 1, 1),  # head_rvel_global
            torch.eye(3, dtype=torch.float32, device=device).repeat(batch_size, win_len, 1, 1),  # lh_rvel_global
            torch.eye(3, dtype=torch.float32, device=device).repeat(batch_size, win_len, 1, 1),  # lh_rvel_local
            torch.eye(3, dtype=torch.float32, device=device).repeat(batch_size, win_len, 1, 1),  # rh_rvel_global
            torch.eye(3, dtype=torch.float32, device=device).repeat(batch_size, win_len, 1, 1)  # rh_rvel_local
        )
    else:
        return (
            torch.ones((batch_size, win_len, 3), dtype=torch.float32, device=device),  # head_pos_global
            torch.eye(3, dtype=torch.float32, device=device).repeat(batch_size, win_len, 1, 1),  # head_rot_global
            torch.ones((batch_size, win_len, 3), dtype=torch.float32, device=device),  # lh_pos_global
            torch.eye(3, dtype=torch.float32, device=device).repeat(batch_size, win_len, 1, 1),  # lh_rot_global
            torch.ones((batch_size, win_len, 3), dtype=torch.float32, device=device),  # rh_pos_global
            torch.eye(3, dtype=torch.float32, device=device).repeat(batch_size, win_len, 1, 1),  # rh_rot_global
            torch.ones((batch_size, win_len, __NJOINTS__, 3), dtype=torch.float32, device=device),  # hmr_joints
            torch.eye(3, dtype=torch.float32, device=device).repeat(batch_size, win_len, __SMPLX_JOINTS__ - 1, 1, 1),  # hmr_body_pose
            torch.eye(3, dtype=torch.float32, device=device).repeat(batch_size, win_len, 1, 1, 1),  # hmr_global_orient
            torch.ones((batch_size, win_len, __NUM_BETAS__), dtype=torch.float32, device=device),  # betas
            torch.tensor([1]*batch_size, dtype=torch.int64, device=device),  # gender - converted to tensor
            torch.ones((batch_size, win_len, __NJOINTS__, 2), dtype=torch.float32, device=device),  # conf
            torch.ones((batch_size, win_len, 3), dtype=torch.float32, device=device),  # head_vel_global
            torch.ones((batch_size, win_len, 3), dtype=torch.float32, device=device),  # lh_vel_global
            torch.ones((batch_size, win_len, 3), dtype=torch.float32, device=device),  # lh_vel_local
            torch.ones((batch_size, win_len, 3), dtype=torch.float32, device=device),  # rh_vel_global
            torch.ones((batch_size, win_len, 3), dtype=torch.float32, device=device),  # rh_vel_local
            torch.eye(3, dtype=torch.float32, device=device).repeat(batch_size, win_len, 1, 1),  # head_rvel_global
            torch.eye(3, dtype=torch.float32, device=device).repeat(batch_size, win_len, 1, 1),  # lh_rvel_global
            torch.eye(3, dtype=torch.float32, device=device).repeat(batch_size, win_len, 1, 1),  # lh_rvel_local
            torch.eye(3, dtype=torch.float32, device=device).repeat(batch_size, win_len, 1, 1),  # rh_rvel_global
            torch.eye(3, dtype=torch.float32, device=device).repeat(batch_size, win_len, 1, 1)  # rh_rvel_local
        )

class ONNXCompatibleAnimModel(HMDPoserExt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = 1
        self.win_len = 1
    
    def forward(self, *tensors):        
        # Unpack all 22 tensors in the correct order

        head_pos_global, head_rot_global, lh_pos_global, lh_rot_global, rh_pos_global, rh_rot_global, \
        hmr_joints, hmr_body_pose, hmr_global_orient, betas, gender, conf, \
        head_vel_global, lh_vel_global, lh_vel_local, rh_vel_global, rh_rvel_local, \
        head_rvel_global, lh_rvel_global, lh_rvel_local, rh_rvel_global, rh_rvel_local \
         = tensors
        # Convert gender tensor to string using conditional logic
        # gender_str = "male" if gender.item() == 1 else "female"
        
        #head_rot_3x3_global_inv = torch.linalg.inv(head_rot_global)
        head_rot_3x3_global_inv = torch.transpose(head_rot_global, -2, -1)
        hmd_posis = [
            # Head + hands
            head_pos_global,
            lh_pos_global,
            rh_pos_global,
            # Hands in head-local space
            (head_rot_3x3_global_inv @ (lh_pos_global - head_pos_global)[..., None])[..., 0],
            (head_rot_3x3_global_inv @ (rh_pos_global - head_pos_global)[..., None])[..., 0],
        ]
        hmd_vel = [
            head_vel_global,
            lh_vel_global,
            rh_vel_global,
            #vel hands in head-local space
            (head_rot_3x3_global_inv @ (lh_vel_global - head_vel_global)[..., None])[..., 0],
            (head_rot_3x3_global_inv @ (rh_vel_global - head_vel_global)[..., None])[..., 0],
        ] 
        hmd_rots = [
            # Head + hands
            head_rot_global,
            lh_rot_global,
            rh_rot_global,
            # Hands in head-local space
            head_rot_3x3_global_inv @ lh_rot_global,
            head_rot_3x3_global_inv @ rh_rot_global,
        ]
        hmd_vel_rots = [
            # Head + hands
            head_rvel_global,
            lh_rvel_global,
            rh_rvel_global,
            # Hands in head-local space
            head_rot_3x3_global_inv @ (lh_rvel_global - head_rvel_global),
            head_rot_3x3_global_inv @ (rh_rvel_global - head_rvel_global),
        ]

        # Model variation that only uses body_pose
        #TODO try with hmr joints expressed locally to head gt instead of yolo nose 
        hmr_joints_local = hmr_joints[:, :, 1:YoloJoints.NUM_JTS] - hmr_joints[:, :, :1]
        chosen_joints = hmr_joints_local[:, :, self.chosen_jts_local]
        chosen_body_pose = hmr_body_pose[:, :, self.chosen_jts_local]

        embeddings = []

        # HMD embeddings
        for c in range(self.num_hmd_channels):
            pos = hmd_posis[c]
            rot_3x3 = hmd_rots[c]
            pos_delta = hmd_vel[c]
            rot_delta_3x3 = hmd_vel_rots[c]
            rot_6d = matrix_to_two_axis(rot_3x3)
            rot_delta_6d = matrix_to_two_axis(rot_delta_3x3)
            feats = [rot_6d, rot_delta_6d, pos, pos_delta]
            emb = torch.cat([self.hmd_embers[c][i](feat) for i, feat in enumerate(feats)], dim=-1)
            if self.use_rnn_layer_norm:
                emb = self.rnn_layer_norm(emb)
            embeddings.append(emb)

        # Pose estimator embeddings
        for c in range(self.num_chosen_jts):
            pos = chosen_joints[:, :, c]
            feats = [pos]
            if self.use_hmr_velocities:
                pos_delta = self.compute_pos_delta(pos)
                feats.append(pos_delta)
            if self.use_hmr_body_pose:
                rot_3x3 = chosen_body_pose[:, :, c]
                rot_6d = matrix_to_two_axis(rot_3x3)
                feats.append(rot_6d)
                if self.use_hmr_velocities:
                    rot_delta_3x3 = self.compute_rot_delta(rot_3x3)
                    rot_delta_6d = matrix_to_two_axis(rot_delta_3x3)
                    feats.append(rot_delta_6d)
            emb = torch.cat([self.joint_embers[c][i](feat) for i, feat in enumerate(feats)], dim=-1)
            if self.use_rnn_layer_norm:
                emb = self.rnn_layer_norm(emb)
            embeddings.append(emb)

        # Pack into one tensor of shape (batch_size, win_len, self.num_channels, hidden_size)
        feats = torch.stack(embeddings, dim=-2)

        # --- Temporal (LSTM) + Spatial (TransformerEncoder) blocks ---
        for b in range(self.num_blocks):

            feats = feats.reshape(self.batch_size, self.win_len, self.num_channels, -1)

            # --- Temporal ---
            # Run temporal encoder per feature group
            feats_temporal = []
            for c in range(self.num_channels):
                prev_rnn_state = self.prev_rnn_states[b][c]
                rnn_output, rnn_state = self.temporal_encoder[b][c](feats[:, :, c, :], prev_rnn_state)
                self.prev_rnn_states[b][c] = rnn_state
                feats_temporal.append(rnn_output)
            self.feats_temporal = torch.stack(feats_temporal, dim=-2)
            
            # --- Spatial ---
            # Pack frames into the batch dimension for efficiency
            feats_temporal_tok = self.feats_temporal.reshape(self.batch_size * self.win_len, self.num_channels, -1)
            feats = self.spatial_encoder[b](feats_temporal_tok)

        # --- Prediction heads ---
        self.feats = feats.reshape(self.batch_size, self.win_len, -1)
        pose_pred = self.pose_head(self.feats)

        pose_pred = pose_pred.reshape(self.batch_size, self.win_len, SmplxJoints.NUM_JTS, 6)
        self.global_orient_6d_pred = pose_pred[:, :, 0]
        self.body_pose_6d_pred = pose_pred[:, :, 1:]
        betas_pred = self.shape_head(self.feats)
        self.betas_pred = betas_pred

        #global_orient_3x3_pred = two_axis_to_matrix_onnx_friendly(self.global_orient_6d_pred)
        #sbody_pose_3x3_pred = two_axis_to_matrix_onnx_friendly(self.body_pose_6d_pred)
        #global_orient_aa_pred = matrix_to_angle_axis_onnx_friendly(global_orient_3x3_pred)
        #body_pose_aa_pred = matrix_to_angle_axis_onnx_friendly(body_pose_3x3_pred)

        #sq_size = self.batch_size * self.win_len

        """
        bm = self.bm_male if gender_str == 'male' else self.bm_female
        body_parms_pred = {
            'pose_body': body_pose_aa_pred.view(sq_size, (SmplxJoints.NUM_JTS-1)*3),
            'root_orient' : global_orient_aa_pred.view(sq_size, -1)
        }
        # Log shapes of each tensor in the dict
        
        body_pose_local = bm(**{k:v for k, v in body_parms_pred.items() if k in ['pose_body', 'root_orient']})
        
        '''
        smplx_output_pred = self.smplx_layer(
            betas=betas_pred.view(sq_size, self.num_betas),
            global_orient=global_orient_3x3_pred.view(sq_size, 3, 3),
            body_pose=body_pose_3x3_pred.view(sq_size, SmplxJoints.NUM_JTS - 1, 3, 3))
        '''
        
        joints_local_pred = (body_pose_local.Jtr[:, :SmplxJoints.NUM_JTS]
                             .reshape(self.batch_size, self.win_len, -1, 3))

        #vertices_local_pred = smplx_output_pred.vertices.reshape(batch_size, win_len, -1, 3)
        head_pos_local_pred = joints_local_pred[:, :, SmplxJoints.HEAD]
        # Force predictions to agree with head ground truth since HMD 6DoF always available
        correction = head_pos_global - head_pos_local_pred
        #transl_pred = correction
        joints_pred = joints_local_pred + correction[..., None, :]
        #vertices_pred = vertices_local_pred + correction[..., None, :]
        """
        #return betas_pred, global_orient_3x3_pred, body_pose_3x3_pred, joints_pred
        #return betas_pred, global_orient_3x3_pred, body_pose_3x3_pred
        return betas_pred, self.global_orient_6d_pred, self.body_pose_6d_pred

def inference_hmdposer(anim_model, input_anim, onnx_inferencer):

    if onnx_inferencer is not None:
        output = onnx_inferencer(input_anim)
    else:
        output = anim_model(*input_anim) #unsqueeze to create batch dimension

# --- custom ---
def __load_and_preprocess_images_ultrafast(image_paths: typing.List[str], target_size: tuple = (640, 640)):
    """
    Ultra-fast image loading with pre-allocated memory
    """
    batch_size = __NCAM__
    height, width = target_size
    
    # Pre-allocate numpy array for batch
    batch_array = np.zeros((batch_size, 3, height, width), dtype=np.float32)
    
    for i, path in enumerate(image_paths):
        # Load with OpenCV
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (width, height))
        
        # Normalize and transpose in one step
        batch_array[i] = img.astype(np.float32).transpose(2, 0, 1) / 255.0
    
    # Single conversion to tensor and GPU transfer
    return torch.from_numpy(batch_array)

def __check_shape(input_mpl):

    middle_points, closest_points_all, target, rays, meta, joints_2ds = input_mpl

    # Check and print shapes for each variable
    if hasattr(middle_points, 'shape'):
        print(f"middle_points.shape: {middle_points.shape}")
    elif hasattr(middle_points, '__len__'):
        print(f"middle_points (len): {len(middle_points)}")
    else:
        print("middle_points is not a tensor/array")

    if hasattr(closest_points_all, 'shape'):
        print(f"closest_points_all.shape: {closest_points_all.shape}")
    elif hasattr(closest_points_all, '__len__'):
        print(f"closest_points_all (len): {len(closest_points_all)}")
    else:
        print("closest_points_all is not a tensor/array")

    if hasattr(target, 'shape'):
        print(f"target.shape: {target.shape}")
    elif hasattr(target, '__len__'):
        print(f"target (len): {len(target)}")
    else:
        print("target is not a tensor/array")

    if hasattr(rays, 'shape'):
        print(f"rays.shape: {rays.shape}")
    elif hasattr(rays, '__len__'):
        print(f"rays (len): {len(rays)}")
    else:
        print("rays is not a tensor/array")

    if hasattr(meta, 'shape'):
        print(f"meta.shape: {meta.shape}")
    elif hasattr(meta, '__len__'):
        print(f"meta (len): {len(meta)}")
    else:
        print("meta is not a tensor/array")

    if hasattr(joints_2ds, 'shape'):
        print(f"joints_2ds.shape: {joints_2ds.shape}")
    elif hasattr(joints_2ds, '__len__'):
        print(f"joints_2ds (len): {len(joints_2ds)}")
    else:
        print("joints_2ds is not a tensor/array")

def _forward(config, yolo_model, mpl_model, anim_model, batch_images, MVGen, onnx_inferencer):

    mpl_onnx_inferencer, anim_onnx_inferencer = onnx_inferencer
    
    batch_images = batch_images.to(0)
    #get multiview 2D keypoints
    results = yolo_model(batch_images, verbose=False)
    #2D data preprocess
    input_mpl = MVGen.get_from_yolo(results)
    #3D pose lifter
    output_mpl = inference_mpl(config, mpl_model, input_mpl, mpl_onnx_inferencer)
    #Self-avatar animtion
    input_anim = __create_dummy_BaseModeInput(device=batch_images.device, output_mpl=output_mpl)
    inference_hmdposer(anim_model, input_anim, anim_onnx_inferencer)

# --- in main ---
def init_models_and_data(yolo_model_str:str, image_path:str, onnx_conversion:bool=False, use_onnx_inference:bool=True):
    
    # --- Models ---
    #YOLO
    yolo_model = init_yolo(yolo_model_str+'.pt')
    yolo_model.fuse()

    #MPL
    args = parse_args()
    reset_config(config, args)

    _, final_output_dir, _ = create_logger(
        config, args.cfg, 'infer')
    
    mpl_model = eval('__models.' + config.MODEL + '.get_multiview_rumpl_net')(config, is_train=False)
    if args.trained_model:
        mpl_model.load_state_dict(torch.load(args.trained_model))
        logging.info('=> loading model from {}'.format(args.trained_model))

    else:
        if config.TEST.MODEL_FILE:
            if config.TEST.MODEL_FILE == 'no_fine_tuning':
                pass
            else:
                logging.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
                mpl_model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
        else:
            model_path = 'model_best.pth.tar' if config.TEST.STATE.startswith('best') else 'final_state.pth.tar'
            model_path = 'checkpoint.pth.tar' if config.TEST.STATE == 'checkpoint' else model_path
            model_state_file = os.path.join(final_output_dir, model_path)

            logging.info('=> loading model from {}'.format(model_state_file))
            mpl_model.load_state_dict(torch.load(model_state_file),  strict=False)

    mpl_model = mpl_model.to(0)
    mpl_model.eval()

    #HMD-Poser
    anim_model = load_model(MODEL_DIR, CHECKPOINT)
    anim_model = anim_model.to(0)
    anim_model.eval()

    mpl_onnx_filepath = os.path.join(ONNX_PATH, 'mpl_model.onnx')
    anim_onnx_filepath = os.path.join(ONNX_PATH, 'anim_model.onnx')

    if onnx_conversion:
        os.makedirs(ONNX_PATH, exist_ok=True)

        # mpl conversion
        if not os.path.exists(mpl_onnx_filepath):
            dummy_input = torch.ones((1, __NJOINTS__, __NCAM__, 7), dtype=torch.float32).to(0)
            ConversionONNX.convert_mpl_model(
                model       = mpl_model,
                input       = dummy_input,
                onnx_path   = mpl_onnx_filepath
            )
        else:
            print(f'{mpl_onnx_filepath} already exists!')

        # anim conversion
        if not os.path.exists(anim_onnx_filepath):
            dummy_input = __create_dummy_BaseModeInput(device=next(anim_model.parameters()).device)
            ConversionONNX.convert_anim_model(
                model       = anim_model,
                input_tuple = dummy_input,
                onnx_path   = anim_onnx_filepath
            )
        else:
            print(f'{anim_onnx_filepath} already exists!')

    # --- Data --- 
    batch_images = __load_and_preprocess_images_ultrafast(image_path)
    MVGen = MultiViewInference_OpenMPLPoser_RUMPL(config)

    # --- ONNX inferencer ---
    mpl_onnx_inferencer, anim_onnx_inferencer = None, None
    
    if use_onnx_inference:

        if os.path.exists(mpl_onnx_filepath):
            mpl_onnx_filepath = os.path.join(ONNX_PATH, 'mpl_model.onnx')
            mpl_onnx_inferencer = ConversionONNX.ONNXInference(mpl_onnx_filepath)
        if os.path.exists(anim_onnx_filepath):
            anim_onnx_filepath = os.path.join(ONNX_PATH, 'anim_model.onnx')
            #anim_onnx_inferencer = ConversionONNX.ONNXInference(anim_onnx_filepath)

    onnx_inferencer = (mpl_onnx_inferencer, anim_onnx_inferencer)

    return config, yolo_model, mpl_model, anim_model, batch_images, MVGen, onnx_inferencer

def inference(args, n_iter_warmup:int=50, n_iter_inf:int=100)->typing.List[float]:
    
    """
    batch_image is (batch_size, RGB, H, W)
    """
    #format_data_to_mpl(config, results, MVGen)

    #warm_up round
    with torch.inference_mode():
        for _ in range(n_iter_warmup):
            _forward(*args)
    #torch.cuda.synchronize()

    timings = []
    with torch.inference_mode():
        for _ in range(n_iter_inf):
            start_time = time.perf_counter()
            _forward(*args)
            #torch.cuda.synchronize()
            end_time = time.perf_counter()
            timings.append((end_time - start_time) * 1000)  # Convert to ms

    return timings 


def get_stats_from_timings(timings:typing.List):

    # Calculate comprehensive statistics
    timings_np = np.array(timings)
    avg_time = np.mean(timings_np)
    min_time = np.min(timings_np)
    max_time = np.max(timings_np)
    std_time = np.std(timings_np)
    median_time = np.median(timings_np)
    p95_time = np.percentile(timings_np, 95)  # 95th percentile
    p99_time = np.percentile(timings_np, 99)  # 99th percentile
    
    return {
        'avg': avg_time,
        'min': min_time,
        'max': max_time,
        'std': std_time,
        'median': median_time,
        'p95': p95_time,
        'p99': p99_time,
        'all_timings': timings_np,
        'cv': (std_time / avg_time) * 100  # Coefficient of variation (%)
    }

def print_detailed_statistics(stats):
    """Print comprehensive timing statistics"""
    print("=== INFERENCE TIME STATISTICS ===")
    print(f"Average: {stats['avg']:.3f} ms")
    print(f"Std Dev: {stats['std']:.3f} ms (±{stats['std']/stats['avg']*100:.1f}%)")
    print(f"Min/Max: {stats['min']:.3f} / {stats['max']:.3f} ms")
    print(f"Median: {stats['median']:.3f} ms")
    print(f"95th percentile: {stats['p95']:.3f} ms")
    print(f"99th percentile: {stats['p99']:.3f} ms")
    print(f"Coefficient of variation: {stats['cv']:.1f}%")
    
    # Check for outliers
    outlier_threshold = stats['avg'] + 2 * stats['std']
    outliers = stats['all_timings'][stats['all_timings'] > outlier_threshold]
    print(f"Outliers (> mean + 2σ): {len(outliers)} ({len(outliers)/len(stats['all_timings'])*100:.1f}%)")

# --- main ---
def __main():

    packed_info = init_models_and_data(
        yolo_model_str      =   YOLO_MODEL,
        image_path          =   ["body_image_cam_1.png"] * __NCAM__,
        onnx_conversion     =   ONNX_CONVERSION,
        use_onnx_inference  =   USE_ONNX_INF
    )
    
    # forward pass and compute inference time
    timings = inference(packed_info)

    # make stats from inference time
    stats = get_stats_from_timings(timings)
    print_detailed_statistics(stats)

if __name__ == '__main__':
    __main()

