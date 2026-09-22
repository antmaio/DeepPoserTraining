"""
AMASS dataset loading and preprocessing.
"""
# External
import enum
import logging
import os
import pathlib
from typing import Iterable, List, Union, Optional

import numpy as np
import smplx
import smplx.lbs
import torch
from torch.utils.data import Dataset

# Internal
import data.data_config as bm_config
from data.data_config import YoloJoints
import filter_model
from utils import utils_fix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

_CONF_THRESHOLD = 0.5
_KP_CLAMP       = 5.0

_DATASET_DIR_MAP = {
    'cmu':     'CMU',
    'hdm05':   'MPI_HDM05',
    'bml_rub': 'BioMotionLab_NTroje',
}

_3D_MODES = ('triang', 'openmpl')

_DATA_SPLIT_DIR = os.path.join('anim', 'data', 'avatarposer_data_split')

_SMPLX_SUBSET_NAME_MAP = {
    'CMU':    'CMU',
    'HDM05':  'MPI_HDM05',
    'BMLrub': 'BioMotionLab_NTroje',
}

_CACHE_SUBDIR = 'hmd-poser-ext-amass'

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Gender(enum.IntEnum):
    MALE    = 0
    FEMALE  = 1
    NEUTRAL = 2

# ---------------------------------------------------------------------------
# SMPLX layer helpers
# ---------------------------------------------------------------------------

def get_frozen_smplx_layer(**kwargs) -> smplx.SMPLXLayer:
    """Return a frozen (no-grad) SMPLX layer from the configured model directory."""
    return smplx.SMPLXLayer(bm_config._SMPLX_DIR_, **kwargs).requires_grad_(False)


def get_smpl_or_smplx_rest_joints(
    layer: Union[smplx.SMPLLayer, smplx.SMPLXLayer],
    betas: torch.Tensor,
) -> torch.Tensor:
    """Compute rest-pose joint positions for the given shape parameters."""
    assert betas.ndim == 2 and betas.shape[1] == layer.num_betas
    v_shaped    = layer.v_template + smplx.lbs.blend_shapes(betas, layer.shapedirs)
    rest_joints = smplx.lbs.vertices2joints(layer.J_regressor, v_shaped)
    return rest_joints

# ---------------------------------------------------------------------------
# Gender encoding
# ---------------------------------------------------------------------------

def _gender_tensor(gender_str: str, dtype: torch.dtype) -> torch.Tensor:
    mapping = {'male': Gender.MALE, 'female': Gender.FEMALE}
    return torch.tensor(mapping.get(gender_str, Gender.NEUTRAL), dtype=dtype)

# ---------------------------------------------------------------------------
# Split / recording path helpers
# ---------------------------------------------------------------------------

def _glob_pkl(data_dir: pathlib.Path, pattern: str) -> List[pathlib.Path]:
    return list(data_dir.glob(pattern))


def get_protocol1_split_relative_paths(cfg, split: str) -> List[pathlib.Path]:
    assert split in ('train', 'valid', 'test')
    split = 'test' if split == 'valid' else split
    return _glob_pkl(pathlib.Path(cfg.data_dir), f'*/{split}/**/*.pkl')


def get_protocol2_split_relative_paths(cfg, split: str) -> List[pathlib.Path]:
    assert split in ('train', 'valid', 'test'), f"Invalid split: {split}"
    assert cfg.as_testset in _DATASET_DIR_MAP, \
        f"as_testset '{cfg.as_testset}' not in {list(_DATASET_DIR_MAP)}."

    data_dir     = pathlib.Path(cfg.data_dir)
    actual_split = 'test' if split in ('test', 'valid') else 'train'
    test_dir     = _DATASET_DIR_MAP[cfg.as_testset]

    if actual_split == 'test':
        logging.info(f'test set: {test_dir}')
        return _glob_pkl(data_dir, f'{test_dir}/**/*.pkl')

    paths = []
    for name, path in _DATASET_DIR_MAP.items():
        if name != cfg.as_testset:
            logging.info(f'train set: {path}')
            paths.extend(_glob_pkl(data_dir, f'{path}/**/*.pkl'))
    return paths


def get_protocol3_split_relative_paths(cfg, split: str) -> List[pathlib.Path]:
    assert split in ('train', 'valid', 'test'), f"Invalid split: {split}"
    split = 'test' if split == 'valid' else split
    return _glob_pkl(pathlib.Path(cfg.data_dir), f'*/{split}/**/*.pkl')


def get_protocol1_split_relative_paths_smplx(cfg, split: str) -> List[str]:
    """Load file lists from AvatarPoser split text files for the SMPLX topology."""
    if split == 'val':
        split = 'test'
    assert split in ('train', 'test')

    def _old_to_new(path: str, subset_old: str, subset_new: str) -> str:
        return (
            path.rstrip('\n')
                .replace(f'{subset_old}/', f'{subset_new}/')
                .replace('poses', 'stageii')
                .replace('/', os.sep)
        )

    def _exclude(path: str) -> bool:
        return not any(name in path for name in ('Rory', 'Justin', 'rory', 'justin'))

    paths = []
    for subset_new, subset_old in _SMPLX_SUBSET_NAME_MAP.items():
        split_file = os.path.join(_DATA_SPLIT_DIR, subset_new, f'{split}_split.txt')
        assert os.path.isfile(split_file), f'Split file not found: {split_file}'
        with open(split_file) as fp:
            raw = fp.readlines()
        converted = (_old_to_new(p, subset_old, subset_new) for p in raw)
        paths.extend(filter(_exclude, converted))
    return paths


def get_dataset_recording_names_for_split(
    cfg, dataset_str: str, split: str, topology: str
) -> List:
    """Return the list of .pkl recording paths for the requested dataset/split."""
    if dataset_str == 'amass-p1':
        return get_protocol1_split_relative_paths(cfg, split)
    if dataset_str == 'amass-p2':
        assert getattr(cfg, 'as_testset', None), \
            "cfg.as_testset must be set for amass-p2."
        return get_protocol2_split_relative_paths(cfg, split)
    if dataset_str == 'amass-p3':
        return get_protocol3_split_relative_paths(cfg, split)
    raise NotImplementedError(f"Dataset '{dataset_str}' is not supported.")


# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------

def load_gt(relative_rec_path) -> dict:
    """Load a ground-truth .pkl recording."""
    return np.load(relative_rec_path, allow_pickle=True)


def load_kp(cfg, relative_rec_path: Union[str, pathlib.Path]) -> dict:
    """Load the corresponding triangulated keypoint .npz file."""
    mode = getattr(cfg, 'mode', 'triang')
    assert mode in _3D_MODES, f"mode '{mode}' must be one of {_3D_MODES}."
    rec   = pathlib.Path(relative_rec_path)
    npz   = rec.parents[1] / mode / (rec.stem + '.npz')
    return np.load(npz, allow_pickle=True)


# ---------------------------------------------------------------------------
# Keypoint preprocessing (shared between dataset and load_smpl)
# ---------------------------------------------------------------------------

def _conf_to_tensor(conf, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if isinstance(conf, np.ndarray):
        return torch.tensor(conf, dtype=dtype)
    if isinstance(conf, torch.Tensor):
        return conf.to(dtype=dtype)
    return None


def preprocess_keypoints(
    keypoints: torch.Tensor,
    cfg,
    conf_scores: Optional[torch.Tensor],
    threshold: float = _CONF_THRESHOLD,
) -> torch.Tensor:
    """
    Zero out low-confidence joints, clamp outliers, and optionally Kalman-filter.

    Args:
        keypoints:   (T, J, 3) raw 3-D keypoint tensor.
        cfg:         Run config (checked for with_kalman_filter / filter_name).
        conf_scores: Optional (T, J) confidence scores.
        threshold:   Joints with confidence below this are zeroed.

    Returns:
        Processed (T, J, 3) tensor.
    """
    keypoints = utils_fix.zero_if_confidences_are_below_threshold(
        keypoints, conf_scores, threshold=threshold
    )
    keypoints = torch.clamp(keypoints, -_KP_CLAMP, _KP_CLAMP)

    if getattr(cfg, 'with_kalman_filter', False):
        kf = filter_model.Filters(
            cfg.filter_name,
            kalman_params_path=getattr(cfg, 'kalman_params_path', None),
        )
        keypoints = kf.filter(keypoints, conf_scores=conf_scores)

    return keypoints


def _load_recording(
    cfg,
    relative_rec_path,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """
    Load one recording's ground-truth data and keypoints into a unified dict.

    Returns keys: rotations_local_full_gt_list, hmd_position_global_full_gt_list,
                  head_global_trans_list, betas, gender, keypoints, body_parms_list,
                  conf (optional).
    """
    data_gt  = load_gt(relative_rec_path)
    data_kp  = load_kp(cfg, relative_rec_path)

    num_frames = data_gt['hmd_position_global_full_gt_list'].shape[0]

    rot    = torch.as_tensor(data_gt['rotation_local_full_gt_list'],        dtype=dtype).cpu()
    hmd    = torch.as_tensor(data_gt['hmd_position_global_full_gt_list'],   dtype=dtype).cpu()
    head_t = torch.as_tensor(data_gt['head_global_trans_list'],             dtype=dtype).cpu()
    betas  = torch.tensor(
        data_gt['shape'][None, :].repeat(num_frames, axis=0), dtype=dtype
    )
    body_parms = data_gt['body_parms_list']
    gender     = _gender_tensor(str(data_gt.get('gender', '')), dtype)

    kp          = torch.tensor(data_kp['points3d'], dtype=dtype)
    conf        = _conf_to_tensor(data_kp.get('conf'), dtype)
    kp          = preprocess_keypoints(kp, cfg, conf)

    return {
        'rotations_local_full_gt_list':       rot,
        'hmd_position_global_full_gt_list':   hmd,
        'head_global_trans_list':             head_t,
        'betas':                              betas,
        'gender':                             gender,
        'keypoints':                          kp,
        'body_parms_list':                    body_parms,
        'conf':                               conf,
        'filepath':                           str(relative_rec_path),
    }


# ---------------------------------------------------------------------------
# Disk-cached recording loader (used by render.py)
# ---------------------------------------------------------------------------

def load_smpl(cfg, relative_rec_path: Union[str, pathlib.Path]) -> dict:
    """
    Load a recording, with a disk cache to avoid reprocessing.

    The cache lives under cfg.cache_dir / _CACHE_SUBDIR.
    """
    cache_dir  = os.path.join(getattr(cfg, 'cache_dir', '.cache'), _CACHE_SUBDIR)
    os.makedirs(cache_dir, exist_ok=True)

    rec_str     = str(relative_rec_path)
    cached_name = rec_str.replace(os.sep, '-').replace('.pkl', '') + '.pt'
    cached_path = os.path.join(cache_dir, cached_name)

    if os.path.exists(cached_path):
        return torch.load(cached_path, weights_only=True)

    rec = _load_recording(cfg, relative_rec_path, dtype=torch.float32)

    # Persist everything except conf=None (store None as absent key)
    to_save = {k: v for k, v in rec.items() if v is not None}
    torch.save(to_save, cached_path)
    return rec


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AMASSDatasetWithTransl(Dataset):
    """
    Windowed AMASS dataset.

    In train/valid mode the full sequences are split into fixed-length,
    overlapping windows that are then served one-by-one.

    In test mode each recording is served as a single item (variable length).

    Args:
        cfg:                       Run configuration object.
        relative_recording_paths:  Iterable of paths to .pkl recordings.
        win_len:                   Window length in frames.
        win_overlap:               Overlap between consecutive windows in frames.
        zero_betas:                Replace shape parameters with zeros.
        dtype:                     Tensor dtype.
        phase:                     'train', 'valid', or 'test'.
    """

    def __init__(
        self,
        cfg,
        relative_recording_paths: Iterable,
        win_len:      int            = 40,
        win_overlap:  int            = 5,
        zero_betas:   bool           = True,
        dtype:        torch.dtype    = torch.float32,
        phase:        str            = 'train',
    ):
        assert win_len > 0,              "win_len must be positive."
        assert 0 <= win_overlap < win_len, "win_overlap must be in [0, win_len)."
        assert phase in ('train', 'valid', 'test'), f"Invalid phase: '{phase}'."

        self._phase       = phase
        self._win_len     = win_len
        self._zero_betas  = zero_betas

        # Per-window storage
        self._rot:      list = []
        self._hmd:      list = []
        self._head_t:   list = []
        self._betas:    list = []
        self._gender:   list = []
        self._kp:       list = []
        self._conf:     list = []
        self._parms:    list = []
        self._transl:   list = []
        self._filepath: list = []

        win_step = win_len - win_overlap

        for rec_path in relative_recording_paths:
            try:
                rec = _load_recording(cfg, rec_path, dtype=dtype)
            except Exception as exc:
                logging.warning(f'Skipping {rec_path}: {exc}')
                continue

            T = rec['hmd_position_global_full_gt_list'].shape[0]

            if phase in ('train', 'valid'):
                if T < win_len:
                    continue
                self._append_windows(rec, T, win_len, win_step)
            else:  # test — whole sequence as one item
                self._append_sequence(rec)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_windows(self, rec: dict, T: int, win_len: int, win_step: int):
        for start in range(0, T, win_step):
            end = start + win_len
            if end > T:                  # last partial window: shift left
                start, end = T - win_len, T

            s = slice(start, end)
            self._rot.append(rec['rotations_local_full_gt_list'][s])
            self._hmd.append(rec['hmd_position_global_full_gt_list'][s])
            self._head_t.append(rec['head_global_trans_list'][s])
            self._betas.append(rec['betas'][s])
            self._gender.append(rec['gender'])
            self._kp.append(rec['keypoints'][s])
            self._conf.append(rec['conf'][s] if isinstance(rec['conf'], torch.Tensor) else None)
            self._parms.append(rec['body_parms_list'])

            self._transl.append(rec['body_parms_list']['trans'][s])
            self._filepath.append(rec['filepath'])

    def _append_sequence(self, rec: dict):
        self._rot.append(rec['rotations_local_full_gt_list'])
        self._hmd.append(rec['hmd_position_global_full_gt_list'])
        self._head_t.append(rec['head_global_trans_list'])
        self._betas.append(rec['betas'])
        self._gender.append(rec['gender'])
        self._kp.append(rec['keypoints'])
        self._conf.append(rec['conf'])
        self._parms.append(rec['body_parms_list'])

        self._transl.append(rec['body_parms_list']['trans'])
        self._filepath.append(rec['filepath'])

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._betas)

    def __getitem__(self, idx: int) -> dict:
        betas = self._betas[idx].clone()
        if self._zero_betas:
            betas = torch.zeros_like(betas)

        out = {
            'rotations_local_full_gt_list':     self._rot[idx].clone(),
            'hmd_position_global_full_gt_list':  self._hmd[idx].clone(),
            'head_global_trans_list':            self._head_t[idx].clone(),
            'betas':                             betas,
            'gender':                            self._gender[idx].clone(),
            'keypoints':                         self._kp[idx].clone(),
            'body_parms_list': (
                self._parms[idx] if self._phase == 'test' else -1
            ),
        }

        conf = self._conf[idx]
        if isinstance(conf, torch.Tensor):
            out['conf'] = conf.clone()

        out['transl']   = self._transl[idx].clone()
        out['filepath'] = self._filepath[idx]

        return out


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_dataset(
    cfg,
    dataset_str: str,
    split:       str,
    topology:    str,
    ratio:       float = None,
    **dataset_args,
) -> Dataset:
    """
    Build and return an AMASSDatasetWithTransl for the requested split.

    Args:
        cfg:         Run configuration.
        dataset_str: e.g. 'amass-p1', 'amass-p2', 'amass-p3'.
        split:       'train', 'valid', or 'test'.
        topology:    'smpl' or 'smplx' (used for recording name resolution).
        ratio:       Optional float in (0, 1) to use only a fraction of data.
        **dataset_args: Forwarded to AMASSDatasetWithTransl.
    """
    rec_names = sorted(get_dataset_recording_names_for_split(cfg, dataset_str, split, topology))

    if ratio is not None:
        assert 0.0 < ratio < 1.0
        new_n = int(len(rec_names) * ratio)
        assert new_n > 0, f"ratio={ratio} too small for {len(rec_names)} recordings."
        rec_names = rec_names[:new_n]

    return AMASSDatasetWithTransl(
        cfg,
        rec_names,
        phase=split,
        **dataset_args,
    )


def get_dataset_smplx_layers(dataset_str: str) -> list:
    """Return the SMPLX body-model layer(s) appropriate for the dataset."""
    if dataset_str in ('amass-p1', 'amass-p2', 'amass-p3'):
        return [get_frozen_smplx_layer(gender='neutral', num_betas=bm_config._NUM_BETAS_)]
    raise NotImplementedError(f"Dataset '{dataset_str}' not supported.")
