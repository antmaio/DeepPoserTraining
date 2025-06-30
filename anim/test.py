
"""
Inspired from https://github.com/georgedf1/sfbpe/blob/main/eval_synthetic.py
"""

# External
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import json
import os
# Internal
import train
from anim.data import amass 
import anim.models as models 
import utils

def __main():
    
    _ = torch.autograd.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str)
    parser.add_argument('dataset', type=str, choices=('amass-p1', 'amass-p2', 'egobody'),
                        help="Override train_args with a TOML file path")
    parser.add_argument('split', type=str, choices=('train', 'val', 'test', 'full'))
    parser.add_argument('--checkpoint', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--win_len', type=int, default=None)
    parser.add_argument('--win_overlap', type=int, default=None)
    parser.add_argument('--zero_betas', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--model_device_str', type=str, default='cuda')
    parser.add_argument('--stats_device_str', type=str, default='cuda')

    args = parser.parse_args()
    model_dir = args.model_dir
    dataset_str = args.dataset
    split = args.split
    checkpoint = args.checkpoint
    batch_size = args.batch_size
    win_len = args.win_len
    win_overlap = args.win_overlap
    zero_betas = args.zero_betas
    model_device = torch.device(args.model_device_str)
    stats_device = torch.device(args.stats_device_str)
    dtype = torch.float32

    # Use same parameters as from last training run of model if unspecified
    train_info = train.get_model_train_info(model_dir)
    if batch_size is None:
        batch_size = train_info['batch_size']
    if win_len is None:
        win_len = train_info['win_len']
    if win_overlap is None:
        win_overlap = train_info['win_overlap']
    if zero_betas is None:
        zero_betas = train_info['zero_betas']
    
    if checkpoint is None:
        checkpoint = models.get_last_model_epoch(model_dir)
    model = models.load_model(model_dir, checkpoint)
    model = model.to(model_device, dtype)
    model.eval()

    dataset = amass.get_dataset(dataset_str, split,
                               win_len=win_len, win_overlap=win_overlap, zero_betas=zero_betas)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    upper_body_pose_joints = [jt - 1 for jt in amass.SMPLX_UPPER_JOINTS]
    lower_body_pose_joints = [amass.SMPLX_LOWER_JOINTS[i] - 1 for i in range(1, len(amass.SMPLX_LOWER_JOINTS))]

    global_orient_gt_list = []
    global_orient_pred_list = []
    body_pose_gt_list = []
    body_pose_pred_list = []
    joints_gt_list = []
    joints_pred_list = []
    for batch in tqdm(iter(dataloader)):
        model_input, model_target = models.batch_to_model_input_and_target(
            batch, model_device, dtype, mode3d=model.mode3d)
                    
        model.reset()
        model_output = model(model_input)

        global_orient_gt_list.append(model_target.global_orient.to(stats_device))
        global_orient_pred_list.append(model_output.global_orient.to(stats_device))
        body_pose_gt_list.append(model_target.body_pose.to(stats_device))
        body_pose_pred_list.append(model_output.body_pose.to(stats_device))
        joints_gt_list.append(model_target.joints.to(stats_device))
        joints_pred_list.append(model_output.joints.to(stats_device))

    global_orient_gt_cat = torch.cat(global_orient_gt_list, dim=0)
    global_orient_pred_cat = torch.cat(global_orient_pred_list, dim=0)
    body_pose_gt_cat = torch.cat(body_pose_gt_list, dim=0)
    body_pose_pred_cat = torch.cat(body_pose_pred_list, dim=0)

    # NB: Good reference metric implementations can be found in HMD-Poser's utils/metrics.py

    # Mean per joint positional error
    joints_gt_cat = torch.cat(joints_gt_list, dim=0)
    joints_pred_cat = torch.cat(joints_pred_list, dim=0)
    pos_err = (joints_pred_cat - joints_gt_cat).square().sum(dim=-1).sqrt()
    mpjpe = 100 * pos_err.mean()
    mpjpe_upper = 100 * pos_err[..., amass.SMPLX_UPPER_JOINTS].mean()
    mpjpe_lower = 100 * pos_err[..., amass.SMPLX_LOWER_JOINTS].mean()

    # Mean per joint rotational error
    global_orient_delta_cat = global_orient_pred_cat @ torch.linalg.inv(global_orient_gt_cat)
    global_orient_rot_err = utils.rotation_angle_radians(global_orient_delta_cat)
    global_orient_rot_err = torch.rad2deg(global_orient_rot_err)
    body_pose_delta_cat = body_pose_pred_cat @ torch.linalg.inv(body_pose_gt_cat)
    body_pose_rot_err = utils.rotation_angle_radians(body_pose_delta_cat)
    body_pose_rot_err = torch.rad2deg(body_pose_rot_err)
    num_jts = amass.SmplxJoints.NUM_JTS
    mpjre_local = body_pose_rot_err.mean()
    mpjre_local_upper = body_pose_rot_err[..., upper_body_pose_joints].mean()
    mpjre_local_lower = body_pose_rot_err[..., lower_body_pose_joints].mean()
    mpjre_root = global_orient_rot_err.mean()
    mpjre = (1 / num_jts) * mpjre_root + ((num_jts - 1) / num_jts) * mpjre_local

    # Mean per joint velocity error
    vel_gt = amass.FPS * (joints_gt_cat[:, 1:] - joints_gt_cat[:, :-1])
    vel_pred = amass.FPS * (joints_pred_cat[:, 1:] - joints_pred_cat[:, :-1])
    vel_err = (vel_pred - vel_gt).square().sum(dim=-1).sqrt()
    mpjve = 100 * vel_err.mean()
    mpjve_upper = 100 * vel_err[..., amass.SMPLX_UPPER_JOINTS].mean()
    mpjve_lower = 100 * vel_err[..., amass.SMPLX_LOWER_JOINTS].mean()

    # TODO If you use jitter, make sure it is what previous works do, or redefine it
    jitter_pred = (joints_pred_cat[:, 3:] - 3 * joints_pred_cat[:, 2:-1] + 3 * joints_pred_cat[:, 1:-2] - joints_pred_cat[:, :-3])
    jitter_gt = joints_gt_cat[:, 3:] - 3 * joints_gt_cat[:, 2:-1] + 3 * joints_gt_cat[:, 1:-2] - joints_gt_cat[:, :-3]
    jitter = 0.01 * (amass.FPS ** 3) * (jitter_pred - jitter_gt).square().sum(dim=-1).sqrt().mean()

    # Convert tensors to float values
    results = {
        "MPJPE": float(mpjpe.detach().cpu().numpy()),
        "MPJPE_Upper": float(mpjpe_upper.detach().cpu().numpy()),
        "MPJPE_Lower": float(mpjpe_lower.detach().cpu().numpy()),
        "MPJRE": float(mpjre.detach().cpu().numpy()),
        "MPJRE_Root": float(mpjre_root.detach().cpu().numpy()),
        "MPJRE_Local": float(mpjre_local.detach().cpu().numpy()),
        "MPJRE_Local_Upper": float(mpjre_local_upper.detach().cpu().numpy()),
        "MPJRE_Local_Lower": float(mpjre_local_lower.detach().cpu().numpy()),
        "MPJVE": float(mpjve.detach().cpu().numpy()),
        "MPJVE_Upper": float(mpjve_upper.detach().cpu().numpy()),
        "MPJVE_Lower": float(mpjve_lower.detach().cpu().numpy())
    }
    # Save to a JSON file
    with open(os.path.join(model_dir, f"eval_synthetic_{dataset_str}_{split}.json"), "w") as fp:
        json.dump(results, fp, indent=4)
    print("Results saved to eval_synthetic_results.json")

    print(f"--- Eval with synthetic pose estimation data on {dataset_str} {split} ---")
    print(f"MPJPE = {results['MPJPE']:.3f} (cm)\n"
          f"  Upper = {results['MPJPE_Upper']:.3f} (cm)\n"
          f"  Lower = {results['MPJPE_Lower']:.3f} (cm)\n"
          f"MPJRE = {results['MPJRE']:.3f} (°)\n"
          f"  Root = {results['MPJRE_Root']:.3f} (°)\n"
          f"  Local = {results['MPJRE_Local']:.3f} (°)\n"
          f"    Upper = {results['MPJRE_Local_Upper']:.3f} (°)\n"
          f"    Lower = {results['MPJRE_Local_Lower']:.3f} (°)\n"
          f"MPJVE = {results['MPJVE']:.3f} (cm/s)"
          f"  Upper = {results['MPJVE_Upper']:.3f} (cm/s)\n"
          f"  Lower = {results['MPJVE_Lower']:.3f} (cm/s)")

if __name__ == '__main__':
    __main()