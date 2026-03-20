"""
From https://github.com/zxz267/AvatarJLM
"""
# External
import os
import logging
import argparse
import tomli
import torch
from typing import Optional
from ultralytics import YOLO

# Internal
from anim.data.amass import get_frozen_smplx_layer
from data.utils_data import process
from human_body_prior.body_model.body_model import BodyModel
from data.data_config import YoloJoints
from data.rendering import init_mesh_viewer
from data.camera_config import (
    CAMERA_PATH_PROTOCOL_1,
    CAMERA_PATH_PROTOCOL_2,
    CAMERA_PATH_PROTOCOL_3,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CAMERA_PATHS = {
    1: CAMERA_PATH_PROTOCOL_1,
    2: CAMERA_PATH_PROTOCOL_2,
    3: CAMERA_PATH_PROTOCOL_3,
}

# Dataset subsets per protocol
_PROTOCOL_DATASETS = {
    1: {'train_test': ['BioMotionLab_NTroje', 'CMU', 'MPI_HDM05']},
    2: {'train_test': ['BioMotionLab_NTroje', 'CMU', 'MPI_HDM05']},
    3: {
        'train': [
            'MPI_HDM05', 'BioMotionLab_NTroje', 'CMU', 'ACCAD', 'BMLmovi',
            'EKUT', 'Eyes_Japan_Dataset', 'KIT', 'MPI_Limits', 'MPI_mosh',
            'SFU', 'TotalCapture',
        ],
        'test': ['HumanEva', 'Transitions_mocap'],
    },
}

# SMPLX topology uses different dataset folder names
_SMPLX_NAME_MAP = {
    'BioMotionLab_NTroje': 'BMLrub',
    'MPI_HDM05':           'HDM05',
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

class Config:
    """Simple attribute container populated from TOML + CLI overrides."""
    root: str
    protocol: int
    support_data: str
    data_split: str
    topology: str
    output_dir: str
    yolo_model: Optional[str]


def _load_config(args: argparse.Namespace) -> Config:
    with open(args.config, 'rb') as f:
        toml = tomli.load(f)

    cfg = Config()
    
    dp_cfg   = toml.get('data_path',  {})
    yolo_cfg = toml.get('yolo',       {})
    bm_cfg   = toml.get('body_model', {})
    pr_cfg   = toml.get('protocol',   {})

    cfg.root         = args.root         or dp_cfg.get('root')
    cfg.protocol     = args.protocol     or pr_cfg.get('protocol')
    cfg.support_data = args.support_data or dp_cfg.get('support_data', './data/support_data')
    cfg.data_split   = args.data_split   or dp_cfg.get('data_split',   './data/data_split')
    cfg.topology     = args.topology     or bm_cfg.get('topology',     'smpl')
    cfg.output_dir   = args.output_dir   or dp_cfg.get('output_dir',   './data/keypoints/')
    cfg.yolo_model   = args.yolo_model   or yolo_cfg.get('model')

    if cfg.root is None:
        raise ValueError("'root' must be specified in config or via --root.")
    if cfg.protocol is None:
        raise ValueError("'protocol' must be specified in config or via --protocol.")

    return cfg


def log_config(cfg: Config, args: argparse.Namespace):
    """Log the configuration state in cyan, highlighting CLI overrides."""
    CYAN = "\033[36m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    logging.info(f"{CYAN}--- Configuration Summary ---{RESET}")
    for key in ['root', 'protocol', 'support_data', 'data_split', 'topology', 'output_dir', 'yolo_model']:
        val = getattr(cfg, key)
        # check if it was provided via CLI
        is_overridden = getattr(args, key, None) is not None
        
        label = f"{key:15}"
        if is_overridden:
            logging.info(f"{CYAN}{label}: {BOLD}{val}{RESET}{CYAN} [CLI OVERRIDE]{RESET}")
        else:
            logging.info(f"{CYAN}{label}: {val}{RESET}")
    logging.info(f"{CYAN}-----------------------------{RESET}")

# ---------------------------------------------------------------------------
# init logging 
# ---------------------------------------------------------------------------

def init_logging()->None:
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_camera_path(cfg: Config) -> str:
    try:
        return _CAMERA_PATHS[cfg.protocol]
    except KeyError:
        raise NotImplementedError(f"Protocol {cfg.protocol} does not exist.")


def make_dst_path(cfg: Config, subset: str, phase: str, use_yolo: bool) -> str:
    """
    Build the output directory path for a given subset / phase.

    Args:
        cfg:      Parsed configuration.
        subset:   Dataset subset name.
        phase:    'train' or 'test'.
        use_yolo: Whether YOLO pose estimation is active.

    Returns:
        Absolute (or project-relative) path string.
    """
    if use_yolo:
        logging.info('Pose estimation with YOLO')
        folder = f'{cfg.yolo_model}_protocol_{cfg.protocol}'
    else:
        logging.info('No external 3-D data')
        folder = f'protocol_{cfg.protocol}_{cfg.topology}'
    
    return os.path.join(cfg.output_dir, folder, subset, phase, 'preprocessed')


# ---------------------------------------------------------------------------
# Body-model initialisation
# ---------------------------------------------------------------------------

def _init_body_models(cfg: Config, device: str) -> tuple[dict, str, str]:
    """
    Instantiate the appropriate body models and return (models_dict, BMLrub_name, HDM_name).
    """
    num_betas = 16

    if cfg.topology == 'smpl':
        num_dmpls = 8
        bm_root = os.path.join(cfg.support_data, 'body_models')

        def _load(gender):
            bm_path   = os.path.join(bm_root, f'smplh/{gender}/model.npz')
            dmpl_path = os.path.join(bm_root, f'dmpls/{gender}/model.npz')
            return BodyModel(
                bm_fname=bm_path,
                num_betas=num_betas,
                num_dmpls=num_dmpls,
                dmpl_fname=dmpl_path,
            ).to(device)

        models = {'male': _load('male'), 'female': _load('female')}
        return models, 'BioMotionLab_NTroje', 'MPI_HDM05'

    elif cfg.topology == 'smplx':
        models = {'neutral': get_frozen_smplx_layer(gender='neutral', num_betas=num_betas)}
        return models, 'BMLrub', 'HDM05'

    raise NotImplementedError(f"Topology '{cfg.topology}' is not implemented.")


# ---------------------------------------------------------------------------
# Dataset iteration helpers
# ---------------------------------------------------------------------------

def _iter_subsets(cfg: Config, BMLrub: str, HDM: str):
    """
    Yield (subset, phase) pairs for the configured protocol.
    Applies the SMPLX name map when cfg.topology == 'smplx'.
    """
    proto = cfg.protocol

    if proto in (1, 2):
        subsets = [BMLrub, 'CMU', HDM]
        for subset in subsets:
            for phase in ('train', 'test'):
                yield subset, phase

    elif proto == 3:
        train_set = [HDM, BMLrub, 'CMU', 'ACCAD', 'BMLmovi', 'EKUT',
                     'Eyes_Japan_Dataset', 'KIT', 'MPI_Limits', 'MPI_mosh',
                     'SFU', 'TotalCapture']
        test_set  = ['HumanEva', 'Transitions_mocap']
        for subset in train_set:
            yield subset, 'train'
        for subset in test_set:
            yield subset, 'test'

# ---------------------------------------------------------------------------
# YOLO init
# ---------------------------------------------------------------------------
def init_yolo(yolo_model: str):
    model = YOLO(yolo_model)
    model.to(0)
    return model

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description='Prepare motion-capture data.')
    parser.add_argument('config', type=str, help='Path to TOML configuration file.')
    parser.add_argument('--root',         type=str, help='Path to data root (overrides config).')
    parser.add_argument('--protocol',     type=int, choices=[1, 2, 3], help='Evaluation protocol (overrides config).')
    parser.add_argument('--support_data', type=str, help='Path to support data (overrides config).')
    parser.add_argument('--data_split',   type=str, help='Path to data split (overrides config).')
    parser.add_argument('--topology',     type=str, choices=['smpl', 'smplx'], help='Body topology (overrides config).')
    parser.add_argument('--output_dir',   type=str, help='Path where the processed keypoints will be saved (overrides config).')
    parser.add_argument('--yolo_model',   type=str, help='YOLO model name, e.g. yolov8n-pose (overrides config).')
    args = parser.parse_args()

    init_logging()

    cfg = _load_config(args)
    log_config(cfg, args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    camera_path = get_camera_path(cfg)

    MV = init_mesh_viewer(camera_path=camera_path)
    body_models, BMLrub, HDM = _init_body_models(cfg, device)

    # ---- optional YOLO initialisation ----
    use_yolo = bool(cfg.yolo_model)
    process_kwargs = {}
    if use_yolo:
        process_kwargs['yolo_model']     = init_yolo(f'{cfg.yolo_model}.pt')
        process_kwargs['yolo_model_str'] = cfg.yolo_model
        process_kwargs['yolo_topology']  = YoloJoints

    os.makedirs(cfg.output_dir, exist_ok=True)

    for subset, phase in _iter_subsets(cfg, BMLrub, HDM):
        print(subset, phase)
        dst = make_dst_path(cfg, subset, phase, use_yolo=use_yolo)
        os.makedirs(dst, exist_ok=True)

        process_args = dict(
            src=os.path.join(cfg.root, subset),
            dst=dst,
            body_models=body_models,
            topology=cfg.topology,
            logging=logging,
            camera_path=camera_path,
            MV=MV,
            **process_kwargs,
        )
        if cfg.protocol in (1, 2):
            split_file = os.path.join(cfg.data_split, subset, f'{phase}_split.txt')
            process_args['split_file'] = split_file

        process(**process_args)

if __name__ == '__main__':
    main()
