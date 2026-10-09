"""Evaluation script for trained pose prediction models.

Loads a trained model from a run directory, runs it on the test split of the
configured dataset, and reports/saves the following metrics:

* MPJPE  - mean per-joint position error [cm]
* MPJRE  - mean per-joint rotation error [deg]
* MPJVE  - mean per-joint velocity error [cm/s]
* Jitter - ratio of predicted to ground-truth jitter

Position and velocity metrics are additionally split into upper/lower body,
and rotation metrics into root/local/upper/lower.

Usage:
    python eval_synthetic.py <model_dir> [--checkpoint EPOCH]

Inspired by https://github.com/georgedf1/sfbpe/blob/main/eval_synthetic.py
"""
# External
import argparse
import json
import logging
import os
from typing import Optional, Sequence

import tomli
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Internal
import anim.models as models
import anim.train as train
import data.data_config as dconfig
from anim.data import amass
from anim.train import Config
from utils import utils_transform

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

# Fields collected from model targets / outputs during inference.
EVAL_FIELDS = ('global_orient', 'body_pose', 'joints')

# Convenience alias: a list of per-sequence tensors, each of shape (T, ...).
SeqList = Sequence[torch.Tensor]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _to_float(t: torch.Tensor) -> float:
    """Convert a scalar tensor to a Python float."""
    return float(t.detach().cpu())


def _cat(seqs: SeqList) -> torch.Tensor:
    """Concatenate per-sequence tensors along the time dimension."""
    return torch.cat(list(seqs), dim=0)


def _euclidean_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-joint Euclidean distance between ``pred`` and ``gt`` (last dim = xyz)."""
    return (pred - gt).square().sum(dim=-1).sqrt()


def _rotation_error_deg(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Geodesic angle [deg] between predicted and ground-truth rotation matrices."""
    delta = pred @ torch.linalg.inv(gt)
    return torch.rad2deg(utils_transform.rotation_angle_radians(delta))


def _velocity(seqs: SeqList) -> torch.Tensor:
    """Finite-difference velocity [units/s] of each sequence, concatenated."""
    return _cat([dconfig.FPS * (s[1:] - s[:-1]) for s in seqs])


def _jitter(seqs: SeqList) -> torch.Tensor:
    """Mean jitter (norm of the third finite difference, scaled by FPS^3).

    Sequences shorter than 4 frames are skipped. Returns 0 if none remain.
    """
    chunks = [
        (dconfig.FPS ** 3)
        * (s[3:] - 3 * s[2:-1] + 3 * s[1:-2] - s[:-3]).norm(dim=-1)
        for s in seqs if s.shape[0] >= 4
    ]
    return torch.cat(chunks).mean() if chunks else torch.tensor(0.0)


def _split_mean(
    err: torch.Tensor,
    upper_idx: Sequence[int],
    lower_idx: Sequence[int],
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean of ``err`` over all, upper-body and lower-body joints.

    Args:
        err: Error tensor whose last dimension indexes joints.
        upper_idx: Joint indices belonging to the upper body.
        lower_idx: Joint indices belonging to the lower body.
        scale: Multiplicative factor (e.g. 100 to convert m -> cm).

    Returns:
        Tuple ``(overall, upper, lower)`` of scalar tensors.
    """
    return (
        scale * err.mean(),
        scale * err[..., upper_idx].mean(),
        scale * err[..., lower_idx].mean(),
    )


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    gt: dict[str, SeqList],
    pred: dict[str, SeqList],
    upper_body_pose_joints: list[int],
    lower_body_pose_joints: list[int],
) -> dict[str, float]:
    """Compute all evaluation metrics from per-sequence predictions.

    Args:
        gt: Ground-truth tensors per sequence, keyed by ``EVAL_FIELDS``.
        pred: Predicted tensors per sequence, keyed by ``EVAL_FIELDS``.
        upper_body_pose_joints: Upper-body indices into ``body_pose``
            (which excludes the root joint).
        lower_body_pose_joints: Lower-body indices into ``body_pose``.

    Returns:
        Dict mapping metric name to a Python float.
    """
    num_jts = dconfig.SmplxJoints.NUM_JTS

    # ---- MPJPE [cm] ----
    pos_err = _euclidean_error(_cat(pred['joints']), _cat(gt['joints']))
    mpjpe, mpjpe_upper, mpjpe_lower = _split_mean(
        pos_err, dconfig.SMPLX_UPPER_JOINTS, dconfig.SMPLX_LOWER_JOINTS, scale=100)

    # ---- MPJRE [deg] ----
    go_err = _rotation_error_deg(_cat(pred['global_orient']), _cat(gt['global_orient']))
    bp_err = _rotation_error_deg(_cat(pred['body_pose']), _cat(gt['body_pose']))

    mpjre_root  = go_err.mean()
    mpjre_local = bp_err.mean()
    mpjre_local_upper = bp_err[..., upper_body_pose_joints].mean()
    mpjre_local_lower = bp_err[..., lower_body_pose_joints].mean()
    mpjre = (mpjre_root + (num_jts - 1) * mpjre_local) / num_jts

    # ---- MPJVE [cm/s] ----
    vel_err = _euclidean_error(_velocity(pred['joints']), _velocity(gt['joints']))
    mpjve, mpjve_upper, mpjve_lower = _split_mean(
        vel_err, dconfig.SMPLX_UPPER_JOINTS, dconfig.SMPLX_LOWER_JOINTS, scale=100)

    # ---- Jitter ----
    jitter_pred = _jitter(pred['joints'])
    jitter_gt   = _jitter(gt['joints'])
    jitter_ratio = jitter_pred / jitter_gt if jitter_gt > 0 else torch.tensor(0.0)

    metrics = {
        'MPJPE':             mpjpe,
        'MPJPE_Upper':       mpjpe_upper,
        'MPJPE_Lower':       mpjpe_lower,
        'MPJRE':             mpjre,
        'MPJRE_Root':        mpjre_root,
        'MPJRE_Local':       mpjre_local,
        'MPJRE_Local_Upper': mpjre_local_upper,
        'MPJRE_Local_Lower': mpjre_local_lower,
        'MPJVE':             mpjve,
        'MPJVE_Upper':       mpjve_upper,
        'MPJVE_Lower':       mpjve_lower,
        'Jitter_Pred':       jitter_pred,
        'Jitter_Gt':         jitter_gt,
        'Jitter':            jitter_ratio,
    }
    return {k: _to_float(v) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Results saving / printing
# ---------------------------------------------------------------------------

def pprint_and_save(
    results: dict[str, float],
    model_dir: str,
    dataset_str: str,
    split: str,
) -> None:
    """Save metrics to JSON in ``model_dir`` and log a formatted summary.

    Args:
        results: Metric dict as returned by :func:`compute_metrics`.
        model_dir: Directory the JSON file is written to.
        dataset_str: Dataset name (used in the filename and log header).
        split: Dataset split name (used in the filename and log header).
    """
    path = os.path.join(model_dir, f'eval_synthetic_{dataset_str}_{split}.json')
    with open(path, 'w') as fp:
        json.dump(results, fp, indent=4)
    logging.info(f'Results saved → {path}')

    r = results
    logging.info(
        f'--- Eval: {dataset_str} {split} ---\n'
        f'MPJPE  = {r["MPJPE"]:.3f} cm  '
        f'(upper {r["MPJPE_Upper"]:.3f} | lower {r["MPJPE_Lower"]:.3f})\n'
        f'MPJRE  = {r["MPJRE"]:.3f} °  '
        f'(root {r["MPJRE_Root"]:.3f} | local {r["MPJRE_Local"]:.3f} | '
        f'upper {r["MPJRE_Local_Upper"]:.3f} | lower {r["MPJRE_Local_Lower"]:.3f})\n'
        f'MPJVE  = {r["MPJVE"]:.3f} cm/s  '
        f'(upper {r["MPJVE_Upper"]:.3f} | lower {r["MPJVE_Lower"]:.3f})\n'
        f'Jitter = {r["Jitter"]:.3f}  '
        f'(pred {r["Jitter_Pred"]:.3f} | gt {r["Jitter_Gt"]:.3f})'
    )


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _load_config(model_dir: str) -> Config:
    """Load the run config of a trained model and fill in missing defaults.

    Args:
        model_dir: Directory containing ``run_config.json`` and
            ``model_config.toml``.

    Returns:
        Populated :class:`Config` (``data_dir`` and ``cache_dir`` are set
        if they were absent from the run config).

    Raises:
        FileNotFoundError: If ``run_config.json`` is missing.
    """
    cfg_dict = train.get_model_run_config(model_dir)
    if cfg_dict is None:
        raise FileNotFoundError(f'Missing run_config.json in {model_dir}')

    cfg_dict['model_config'] = os.path.join(model_dir, 'model_config.toml')
    cfg = Config(cfg_dict)

    if not hasattr(cfg, 'data_dir'):
        protocol = getattr(cfg, 'protocol', 1)
        str_prot = 1 if protocol in (1, 2) else 3
        y_model  = getattr(cfg, 'yolo_model', 'yolov8n-pose')
        data_dir = f'./data/keypoints/{y_model}_protocol_{str_prot}'
        if getattr(cfg, 'skinned_mesh_topology', 'smpl') == 'smplx':
            data_dir += '_smplx'
        cfg.data_dir = data_dir

    if not hasattr(cfg, 'cache_dir'):
        cfg.cache_dir = '.cache'

    return cfg


def _load_model_config_dict(path: str) -> Optional[dict]:
    """Read a TOML model config, returning ``None`` if it cannot be parsed."""
    try:
        with open(path, 'rb') as fp:
            return tomli.load(fp)
    except Exception:
        return None


def _body_pose_joint_sets() -> tuple[list[int], list[int]]:
    """Upper/lower-body joint indices relative to ``body_pose``.

    ``body_pose`` excludes the root, so all indices are shifted by -1 and the
    first lower-body joint (the pelvis/root) is dropped.
    """
    upper = [j - 1 for j in dconfig.SMPLX_UPPER_JOINTS]
    lower = [j - 1 for j in dconfig.SMPLX_LOWER_JOINTS[1:]]
    return upper, lower


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments, run inference on the test split, and report metrics."""
    torch.autograd.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description='Evaluate a trained pose prediction model.')
    parser.add_argument('model_dir', type=str, help='Directory of the trained model run.')
    parser.add_argument('--checkpoint', type=int, default=None,
                        help='Checkpoint epoch (default: latest).')
    args = parser.parse_args()

    # ---- config ----
    cfg = _load_config(args.model_dir)
    train.log_config(
        cfg.__dict__, is_train=False,
        model_cfg_dict=_load_model_config_dict(cfg.model_config),
    )

    device = torch.device(getattr(cfg, 'device_str', 'cuda'))
    dtype  = torch.float32

    dataset_str = getattr(cfg, 'dataset', 'amass-p1')
    split       = 'test'

    # ---- dataset / dataloader ----
    dataset = amass.get_dataset(
        cfg, dataset_str, split,
        topology=getattr(cfg, 'skinned_mesh_topology', 'smpl'),
        win_len=getattr(cfg, 'win_len', 40),
        win_overlap=getattr(cfg, 'win_overlap', 5),
        zero_betas=getattr(cfg, 'zero_betas', False),
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # ---- model ----
    checkpoint = args.checkpoint or models.get_last_model_epoch(args.model_dir)
    model = models.load_model(args.model_dir, checkpoint, len(dataloader), device=device)
    model = model.to(device, dtype)
    model.eval()
    if not getattr(model, 'nbatch', None):
        model.set_nbatch(len(dataloader))
    if not getattr(model, 'lr_scheduler', None):
        model.set_scheduler()

    # ---- inference loop ----
    gt:   dict[str, list[torch.Tensor]] = {f: [] for f in EVAL_FIELDS}
    pred: dict[str, list[torch.Tensor]] = {f: [] for f in EVAL_FIELDS}

    logging.info(f'Evaluating on {split.upper()} split…')
    for batch in tqdm(dataloader):
        model_input, model_target = models.batch_to_model_input_and_target(
            batch, device, dtype, mode3d=model.mode3d,
        )
        model.reset()
        model_output = model(model_input)

        for field in EVAL_FIELDS:
            gt[field].append(getattr(model_target, field).squeeze(0).to(device))
            pred[field].append(getattr(model_output, field).squeeze(0).to(device))

    # ---- metrics ----
    upper_joints, lower_joints = _body_pose_joint_sets()
    results = compute_metrics(gt, pred, upper_joints, lower_joints)
    pprint_and_save(results, args.model_dir, dataset_str, split)


if __name__ == '__main__':
    main()