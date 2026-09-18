"""
Prepare motion-capture data for training/evaluation.

Loads a TOML configuration file describing a dataset root, evaluation
protocol, body-model topology, and (optionally) a YOLO pose-estimation
model, then iterates over the relevant dataset subsets/phases and runs
the preprocessing pipeline (`data.utils_data.process`) for each one,
writing the resulting keypoints to disk.

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

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

class Config:
    """Simple attribute container populated from TOML.

    Attributes:
        root: Path to the root of the motion-capture dataset.
        protocol: Evaluation protocol identifier (1, 2, or 3).
        support_data: Path to support data (e.g. body-model files).
        data_split: Path to the directory containing train/test split files.
        topology: Body-model topology to use, either 'smpl' or 'smplx'.
        output_dir: Directory where processed keypoints will be written.
        yolo_model: Optional YOLO model name used for 2-D pose estimation.
            If not set, no YOLO-based pose estimation is performed.
    """
    root: str
    protocol: int
    support_data: str
    data_split: str
    topology: str
    output_dir: str
    yolo_model: Optional[str]


def _load_config(args: argparse.Namespace) -> Config:
    """Load and validate the configuration from a TOML file.

    Reads the TOML file at `args.config` and populates a `Config` object
    from its `data_path`, `yolo`, `body_model`, and `protocol` sections,
    applying sensible defaults where applicable.

    Args:
        args: Parsed command-line arguments; only `args.config` (the path
            to the TOML configuration file) is used.

    Returns:
        A populated `Config` instance.

    Raises:
        ValueError: If `root` or `protocol` is not specified in the
            config file.
    """
    with open(args.config, 'rb') as f:
        toml = tomli.load(f)

    cfg = Config()

    dp_cfg   = toml.get('data_path',  {})
    yolo_cfg = toml.get('yolo',       {})
    bm_cfg   = toml.get('body_model', {})
    pr_cfg   = toml.get('protocol',   {})

    cfg.root         = dp_cfg.get('root')
    cfg.protocol     = pr_cfg.get('protocol')
    cfg.support_data = dp_cfg.get('support_data', './data/support_data')
    cfg.data_split   = dp_cfg.get('data_split',   './data/data_split')
    cfg.topology     = bm_cfg.get('topology',     'smpl')
    cfg.output_dir   = dp_cfg.get('output_dir',   './data/keypoints/')
    cfg.yolo_model   = yolo_cfg.get('model')

    if cfg.root is None:
        raise ValueError("'root' must be specified in the config file.")
    if cfg.protocol is None:
        raise ValueError("'protocol' must be specified in the config file.")

    return cfg


def log_config(cfg: Config):
    """Log a summary of the resolved configuration.

    Prints each configuration field to the logger, formatted in cyan for
    readability in the console.

    Args:
        cfg: The configuration to log.
    """
    CYAN = "\033[36m"
    RESET = "\033[0m"

    logging.info(f"{CYAN}--- Configuration Summary ---{RESET}")
    for key in ['root', 'protocol', 'support_data', 'data_split', 'topology', 'output_dir', 'yolo_model']:
        val = getattr(cfg, key)
        label = f"{key:15}"
        logging.info(f"{CYAN}{label}: {val}{RESET}")
    logging.info(f"{CYAN}-----------------------------{RESET}")

# ---------------------------------------------------------------------------
# init logging 
# ---------------------------------------------------------------------------

def init_logging()->None:
    """Configure the root logger.

    Sets up basic logging with an INFO level and a timestamped format of
    the form "YYYY-MM-DD HH:MM:SS - LEVEL - message".
    """
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_camera_path(cfg: Config) -> str:
    """Resolve the camera configuration path for the configured protocol.

    Args:
        cfg: The parsed configuration, whose `protocol` field selects
            which camera path to return.

    Returns:
        The camera path string associated with `cfg.protocol`.

    Raises:
        NotImplementedError: If `cfg.protocol` is not one of the
            supported protocols (1, 2, or 3).
    """
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

    Args:
        cfg: Parsed configuration; `cfg.topology` selects between the
            'smpl' and 'smplx' body-model families, and `cfg.support_data`
            locates the on-disk model files for the 'smpl' family.
        device: Torch device string (e.g. 'cuda' or 'cpu') to move the
            instantiated models to.

    Returns:
        A 3-tuple of:
            - A dict mapping gender (or 'neutral') to the loaded body model.
            - The dataset folder name to use in place of 'BioMotionLab_NTroje'.
            - The dataset folder name to use in place of 'MPI_HDM05'.

    Raises:
        NotImplementedError: If `cfg.topology` is neither 'smpl' nor 'smplx'.
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

    Args:
        cfg: Parsed configuration; `cfg.protocol` (1, 2, or 3) selects
            which dataset subsets and phases are yielded.
        BMLrub: Dataset folder name to use for the "BioMotionLab_NTroje"
            subset (topology-dependent naming).
        HDM: Dataset folder name to use for the "MPI_HDM05" subset
            (topology-dependent naming).

    Yields:
        Tuples of `(subset, phase)`, where `phase` is either 'train'
        or 'test'.
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
    """Load a YOLO model and move it to GPU device 0.

    Args:
        yolo_model: Path or identifier of the YOLO model weights to load
            (e.g. 'yolov8n-pose.pt').

    Returns:
        The loaded `YOLO` model instance, moved to CUDA device 0.
    """
    model = YOLO(yolo_model)
    model.to(0)
    return model

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the motion-capture data preparation pipeline.

    Parses the `config` command-line argument, loads and logs the
    resulting configuration, initializes the body models, mesh viewer,
    and (optionally) a YOLO pose-estimation model, then iterates over
    every dataset subset/phase for the configured protocol and runs
    `data.utils_data.process` to generate and save preprocessed
    keypoints for each one.
    """

    parser = argparse.ArgumentParser(description='Prepare motion-capture data.')
    parser.add_argument('config', type=str, help='Path to TOML configuration file.')
    args = parser.parse_args()

    init_logging()

    cfg = _load_config(args)
    log_config(cfg)

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