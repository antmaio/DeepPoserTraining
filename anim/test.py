"""
Evaluation script.
Inspired by https://github.com/georgedf1/sfbpe/blob/main/eval_synthetic.py
"""
# External
import argparse
import json
import logging
import os
import pathlib
import tomli

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# Internal
import anim.models as models
import anim.train as train
from anim.data import amass
from anim.data.amass import SmplxJoints, get_dataset_recording_names_for_split, load_gt
from anim.train import Config
from utils import utils_transform

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)



# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _tensor_float(t: torch.Tensor) -> float:
    return float(t.detach().cpu())


def compute_metrics(
    joints_gt_list,
    joints_pred_list,
    global_orient_gt_list,
    global_orient_pred_list,
    body_pose_gt_list,
    body_pose_pred_list,
    betas_gt_list,
    betas_pred_list,
    upper_body_pose_joints: list[int],
    lower_body_pose_joints: list[int],
) -> dict:
    """Concatenate per-sequence lists and compute all evaluation metrics."""

    def _cat(lst):
        return torch.cat(lst, dim=0) if lst else None

    go_gt   = _cat(global_orient_gt_list)
    go_pred = _cat(global_orient_pred_list)
    bp_gt   = _cat(body_pose_gt_list)
    bp_pred = _cat(body_pose_pred_list)
    j_gt    = _cat(joints_gt_list)
    j_pred  = _cat(joints_pred_list)
    b_gt    = _cat(betas_gt_list)
    b_pred  = _cat(betas_pred_list)

    # ---- MPJPE [cm] ----
    pos_err    = (j_pred - j_gt).square().sum(dim=-1).sqrt()
    mpjpe      = 100 * pos_err.mean()
    mpjpe_upper = 100 * pos_err[..., amass.SMPLX_UPPER_JOINTS].mean()
    mpjpe_lower = 100 * pos_err[..., amass.SMPLX_LOWER_JOINTS].mean()

    # ---- MPJRE [°] ----
    num_jts = SmplxJoints.NUM_JTS

    go_delta = go_pred @ torch.linalg.inv(go_gt)
    go_err   = torch.rad2deg(utils_transform.rotation_angle_radians(go_delta))

    bp_delta = bp_pred @ torch.linalg.inv(bp_gt)
    bp_err   = torch.rad2deg(utils_transform.rotation_angle_radians(bp_delta))

    mpjre_root        = go_err.mean()
    mpjre_local       = bp_err.mean()
    mpjre_local_upper = bp_err[..., upper_body_pose_joints].mean()
    mpjre_local_lower = bp_err[..., lower_body_pose_joints].mean()
    mpjre = (mpjre_root + (num_jts - 1) * mpjre_local) / num_jts

    # ---- MPJVE [cm/s] ----
    vel_gt   = torch.cat([amass.FPS * (j[1:] - j[:-1]) for j in joints_gt_list])
    vel_pred = torch.cat([amass.FPS * (j[1:] - j[:-1]) for j in joints_pred_list])
    vel_err  = (vel_pred - vel_gt).square().sum(dim=-1).sqrt()
    mpjve       = 100 * vel_err.mean()
    mpjve_upper = 100 * vel_err[..., amass.SMPLX_UPPER_JOINTS].mean()
    mpjve_lower = 100 * vel_err[..., amass.SMPLX_LOWER_JOINTS].mean()

    # ---- Jitter ----
    def _jitter(seqs):
        chunks = [
            (amass.FPS ** 3) * (s[3:] - 3*s[2:-1] + 3*s[1:-2] - s[:-3]).norm(dim=-1)
            for s in seqs if s.shape[0] >= 4
        ]
        return torch.cat(chunks).mean() if chunks else torch.tensor(0.0)

    j_pred_val = _jitter(joints_pred_list)
    j_gt_val   = _jitter(joints_gt_list)
    j_ratio    = j_pred_val / j_gt_val if j_gt_val > 0 else torch.tensor(0.0)

    # ---- Shape ----
    if b_pred is not None and b_gt is not None and b_pred.shape[-1] > 0:
        shape_err = torch.abs(b_pred - b_gt).sum(dim=-1).mean()
    else:
        shape_err = torch.tensor(0.0)

    return {
        'MPJPE':             _tensor_float(mpjpe),
        'MPJPE_Upper':       _tensor_float(mpjpe_upper),
        'MPJPE_Lower':       _tensor_float(mpjpe_lower),
        'MPJRE':             _tensor_float(mpjre),
        'MPJRE_Root':        _tensor_float(mpjre_root),
        'MPJRE_Local':       _tensor_float(mpjre_local),
        'MPJRE_Local_Upper': _tensor_float(mpjre_local_upper),
        'MPJRE_Local_Lower': _tensor_float(mpjre_local_lower),
        'MPJVE':             _tensor_float(mpjve),
        'MPJVE_Upper':       _tensor_float(mpjve_upper),
        'MPJVE_Lower':       _tensor_float(mpjve_lower),
        'Jitter_Pred':       _tensor_float(j_pred_val),
        'Jitter_Gt':         _tensor_float(j_gt_val),
        'Jitter':            _tensor_float(j_ratio),
        'Shape':             _tensor_float(shape_err),
    }


# ---------------------------------------------------------------------------
# Results saving / printing
# ---------------------------------------------------------------------------

def pprint_and_save(
    results: dict,
    model_dir: str,
    dataset_str: str,
    split: str,
):
    filename = f'eval_synthetic_{dataset_str}_{split}.json'
    path     = os.path.join(model_dir, filename)
    with open(path, 'w') as fp:
        json.dump(results, fp, indent=4)
    logging.info(f'Results saved → {path}')

    r = results
    logging.info(
        f'--- Eval: {dataset_str} {split} ---\n'
        f'MPJPE          = {r["MPJPE"]:.3f} cm  '
        f'(upper {r["MPJPE_Upper"]:.3f} | lower {r["MPJPE_Lower"]:.3f})\n'
        f'MPJRE          = {r["MPJRE"]:.3f} °  '
        f'(root {r["MPJRE_Root"]:.3f} | local {r["MPJRE_Local"]:.3f} | '
        f'upper {r["MPJRE_Local_Upper"]:.3f} | lower {r["MPJRE_Local_Lower"]:.3f})\n'
        f'MPJVE          = {r["MPJVE"]:.3f} cm/s  '
        f'(upper {r["MPJVE_Upper"]:.3f} | lower {r["MPJVE_Lower"]:.3f})\n'
        f'Jitter         = {r["Jitter"]:.3f}  '
        f'(pred {r["Jitter_Pred"]:.3f} | gt {r["Jitter_Gt"]:.3f})\n'
        f'Shape          = {r["Shape"]:.3f}'
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    torch.autograd.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description='Evaluate a trained pose prediction model.')
    parser.add_argument('model_dir',    type=str)
    parser.add_argument('--checkpoint', type=int, default=None)
    args = parser.parse_args()

    # ---- load run config ----
    cfg_dict = train.get_model_run_config(args.model_dir)
    if cfg_dict is None:
        raise FileNotFoundError(f'Missing run_config.json in {args.model_dir}')
    
    cfg_dict['model_config'] = os.path.join(args.model_dir, 'model_config.toml')
    cfg = Config(cfg_dict)

    # ---- device & dtype ----
    device = torch.device(getattr(cfg, 'device_str', 'cuda'))
    dtype  = torch.float32

    dataset_str  = getattr(cfg, 'dataset',               'amass-p1')
    topology     = getattr(cfg, 'skinned_mesh_topology',  'smpl')
    win_len      = getattr(cfg, 'win_len',                40)
    win_overlap  = getattr(cfg, 'win_overlap',            5)
    zero_betas   = getattr(cfg, 'zero_betas',             False)

    # ---- default data_dir if missing ----
    if not hasattr(cfg, 'data_dir'):
        protocol  = getattr(cfg, 'protocol', 1)
        str_prot  = 1 if protocol in (1, 2) else 3
        y_model   = getattr(cfg, 'yolo_model', 'yolov8n-pose')
        data_dir  = f'./data/keypoints/{y_model}_protocol_{str_prot}'
        if topology == 'smplx':
            data_dir += '_smplx'
        cfg.data_dir = data_dir

    if not hasattr(cfg, 'cache_dir'):
        cfg.cache_dir = '.cache'

    try:
        with open(cfg.model_config, 'rb') as fp:
            model_cfg_dict = tomli.load(fp)
    except Exception:
        model_cfg_dict = None

    train.log_config(cfg.__dict__, is_train=False, model_cfg_dict=model_cfg_dict)


    # ---- dataset / dataloader ----
    dataset    = amass.get_dataset(cfg, dataset_str, 'test',
                                   topology=topology, win_len=win_len,
                                   win_overlap=win_overlap, zero_betas=zero_betas)
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

    # ---- body-pose joint index sets ----
    upper_body_pose_joints = [jt - 1 for jt in amass.SMPLX_UPPER_JOINTS]
    lower_body_pose_joints = [amass.SMPLX_LOWER_JOINTS[i] - 1
                              for i in range(1, len(amass.SMPLX_LOWER_JOINTS))]

    # ---- inference loop ----
    go_gt_list   = []
    go_pred_list = []
    bp_gt_list   = []
    bp_pred_list = []
    j_gt_list    = []
    j_pred_list  = []
    b_gt_list    = []
    b_pred_list  = []

    logging.info('Evaluating on TEST split…')
    for idx, batch in tqdm(enumerate(dataloader)):
        model_input, model_target = models.batch_to_model_input_and_target(
            batch, device, dtype, mode3d=model.mode3d
        )

        model.reset()
        model_output = model(model_input)


        def _s(t):
            return t.squeeze(0).to(device) if t is not None else None

        go_gt_list.append(_s(model_target.global_orient))
        go_pred_list.append(_s(model_output.global_orient))
        bp_gt_list.append(_s(model_target.body_pose))
        bp_pred_list.append(_s(model_output.body_pose))
        j_gt_list.append(_s(model_target.joints))
        j_pred_list.append(_s(model_output.joints))
        if model_target.betas is not None:
            b_gt_list.append(_s(model_target.betas))
        if model_output.betas is not None:
            b_pred_list.append(_s(model_output.betas))

    results = compute_metrics(
        j_gt_list, j_pred_list,
        go_gt_list, go_pred_list,
        bp_gt_list, bp_pred_list,
        b_gt_list, b_pred_list,
        upper_body_pose_joints,
        lower_body_pose_joints,
    )
    pprint_and_save(results, args.model_dir, dataset_str, 'test')


if __name__ == '__main__':
    main()
