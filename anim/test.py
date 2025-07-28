
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
import logging


# Internal
import anim.train as train
from anim.data import amass
import anim.models as models
from utils import utils_transform
import _debug as DEBUG
from anim.data.amass import SmplxJoints, YoloJoints
import config


def pprint_and_save(results:dict, model_dir:str, dataset_str:str, split:str)->None:

    # Save to a JSON file
    with open(os.path.join(model_dir, f"eval_synthetic_{dataset_str}_{split}.json"), "w") as fp:
        json.dump(results, fp, indent=4)
    logging.info("Results saved to eval_synthetic_results.json")

    logging.info(f"--- Eval with synthetic pose estimation data on {dataset_str} {split} ---")
    logging.info(f"MPJPE = {results['MPJPE']:.3f} (cm)\n"
          f"  Upper = {results['MPJPE_Upper']:.3f} (cm)\n"
          f"  Lower = {results['MPJPE_Lower']:.3f} (cm)\n"
          f"MPJRE = {results['MPJRE']:.3f} (°)\n"
          f"  Root = {results['MPJRE_Root']:.3f} (°)\n"
          f"  Local = {results['MPJRE_Local']:.3f} (°)\n"
          f"    Upper = {results['MPJRE_Local_Upper']:.3f} (°)\n"
          f"    Lower = {results['MPJRE_Local_Lower']:.3f} (°)\n"
          f"MPJVE = {results['MPJVE']:.3f} (cm/s)"
          f"  Upper = {results['MPJVE_Upper']:.3f} (cm/s)\n"
          f"  Lower = {results['MPJVE_Lower']:.3f} (cm/s) \n"
          f"Jitter_Pred = {results['Jitter_Pred']:.3f} \n" 
          f"Jitter_Gt = {results['Jitter_Gt']:.3f} \n"
          f"Jitter = {results['Jitter']:.3f}" 
    )

def __main():

    # Overwrite log file every time the script runs
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

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
    config_info = train.get_model_config_info(model_dir)

    if train_info is not None:
        if batch_size is None:
            batch_size = train_info['batch_size']
        if win_len is None:
            win_len = train_info['win_len']
        if win_overlap is None:
            win_overlap = train_info['win_overlap']
        if zero_betas is None:
            zero_betas = train_info['zero_betas']
    else:
        if any(v is None for v in [batch_size, win_len, win_overlap, zero_betas]):
            raise ValueError('train info not provided in file nor in args')

    # Use parameters in config_json if the file exists, parameters in config.py otherwise  
    if config_info is not None: 
        config.YOLO_MODEL   = config_info['yolo_model']
        config.PROTOCOL     = config_info['protocol']
        config.AS_TESTSET   = config_info['as_testset']
        config.MODE         = config_info['mode']
        config.CACHE_DIR    = config_info['cache_dir']
        config.DATA_DIR     = config_info['data_dir']
    logging.info('Config parameters:')
    logging.info(f'  YOLO_MODEL : {config.YOLO_MODEL}')
    logging.info(f'  PROTOCOL   : {config.PROTOCOL}')
    logging.info(f'  AS_TESTSET : {config.AS_TESTSET}')
    logging.info(f'  MODE       : {config.MODE}')
    logging.info(f'  CACHE_DIR  : {config.CACHE_DIR}')
    logging.info(f'  DATA_DIR   : {config.DATA_DIR}')
    
    if checkpoint is None:
        checkpoint = models.get_last_model_epoch(model_dir)
    model = models.load_model(model_dir, checkpoint)
    model = model.to(model_device, dtype)
    model.eval()

    dataset = amass.get_dataset(config, dataset_str, split,
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

    #
    # --- batched evaluation for 40 frames animation ---
    #
    if split in ('train', 'val'):

        logging.info(f'Evaluating batch of size {batch_size} of {win_len}-frames motion length')
        
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

            #TODO remove debug if ok
            #DEBUG.scatter_plot_3d_pose(model_target.joints, model_output.joints, b=59, f=2)
            #DEBUG.plot_rot(model_target.body_pose, model_output.body_pose, b=59, j=SmplxJoints.RIGHT_FOOT-1)
            #DEBUG.subplot_rot(model_target.body_pose, model_output.body_pose, b=59, j=SmplxJoints.RIGHT_FOOT-1)
            
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
        global_orient_rot_err = utils_transform.rotation_angle_radians(global_orient_delta_cat)
        global_orient_rot_err = torch.rad2deg(global_orient_rot_err)
        body_pose_delta_cat = body_pose_pred_cat @ torch.linalg.inv(body_pose_gt_cat)
        body_pose_rot_err = utils_transform.rotation_angle_radians(body_pose_delta_cat)
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
        jitter_pred = ((amass.FPS ** 3) * (joints_pred_cat[:, 3:] - 3 * joints_pred_cat[:, 2:-1]  
            + 3 * joints_pred_cat[:, 1:-2]  - joints_pred_cat[:, :-3])).norm(dim=-1)
        jitter_gt   = ((amass.FPS ** 3) * (joints_gt_cat[:, 3:]   - 3 * joints_gt_cat[:, 2:-1]    
            + 3 * joints_gt_cat[:, 1:-2]    - joints_gt_cat[:, :-3])).norm(dim=-1)

        jitter_pred = jitter_pred.mean()
        jitter_gt = jitter_gt.mean()
        jitter = jitter_pred / jitter_gt

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
            "MPJVE_Lower": float(mpjve_lower.detach().cpu().numpy()),
            "Jitter_Pred": float(jitter_pred.mean().detach().cpu().numpy()),
            "Jitter_Gt": float(jitter_gt.mean().detach().cpu().numpy()),
            "Jitter": float(jitter.detach().cpu().numpy())
        }
        # Save to a JSON file
        pprint_and_save(results, model_dir, dataset_str, split)
    
    #
    # --- unbatched evaluation for n-frames animation ---
    #
    
    elif split == 'test':
        logging.info(f'Evaluating unbatched tensor of various motion length')

        for batch in tqdm(iter(dataloader)):
            model_input, model_target = models.batch_to_model_input_and_target(
                batch, model_device, dtype, mode3d=model.mode3d)
                        
            model.reset()
            model_output = model(model_input)

            #Remove batch dim
            global_orient_gt_list.append(model_target.global_orient.squeeze(0).to(stats_device))
            global_orient_pred_list.append(model_output.global_orient.squeeze(0).to(stats_device))
            body_pose_gt_list.append(model_target.body_pose.squeeze(0).to(stats_device))
            body_pose_pred_list.append(model_output.body_pose.squeeze(0).to(stats_device))
            joints_gt_list.append(model_target.joints.squeeze(0).to(stats_device))
            joints_pred_list.append(model_output.joints.squeeze(0).to(stats_device))

        
        global_orient_gt_cat = torch.cat(global_orient_gt_list, dim=0)
        global_orient_pred_cat = torch.cat(global_orient_pred_list, dim=0)
        body_pose_gt_cat = torch.cat(body_pose_gt_list, dim=0)
        body_pose_pred_cat = torch.cat(body_pose_pred_list, dim=0)
        joints_gt_cat = torch.cat(joints_gt_list, dim=0)
        joints_pred_cat = torch.cat(joints_pred_list, dim=0)

        #MPJPE [cm]
        pos_err = (joints_pred_cat - joints_gt_cat).square().sum(dim=-1).sqrt()
        mpjpe = 100 * pos_err.mean()
        mpjpe_upper = 100 * pos_err[..., amass.SMPLX_UPPER_JOINTS].mean()
        mpjpe_lower = 100 * pos_err[..., amass.SMPLX_LOWER_JOINTS].mean()

        #MPJRE [°]
        global_orient_delta_cat = global_orient_pred_cat @ torch.linalg.inv(global_orient_gt_cat)
        global_orient_rot_err = utils_transform.rotation_angle_radians(global_orient_delta_cat)
        global_orient_rot_err = torch.rad2deg(global_orient_rot_err)
        body_pose_delta_cat = body_pose_pred_cat @ torch.linalg.inv(body_pose_gt_cat)
        body_pose_rot_err = utils_transform.rotation_angle_radians(body_pose_delta_cat)
        body_pose_rot_err = torch.rad2deg(body_pose_rot_err)
        num_jts = amass.SmplxJoints.NUM_JTS
        mpjre_local = body_pose_rot_err.mean()
        mpjre_local_upper = body_pose_rot_err[..., upper_body_pose_joints].mean()
        mpjre_local_lower = body_pose_rot_err[..., lower_body_pose_joints].mean()
        mpjre_root = global_orient_rot_err.mean()
        mpjre = (1 / num_jts) * mpjre_root + ((num_jts - 1) / num_jts) * mpjre_local

        #MPJVE [cm/s]
        vel_gt_list = []
        vel_pred_list = []
        for j_gt, j_pred in zip(joints_gt_list, joints_pred_list):
            vel_gt_list.append(amass.FPS * (j_gt[1:] - j_gt[:-1]))
            vel_pred_list.append(amass.FPS * (j_pred[1:] - j_pred[:-1]))

        vel_gt_cat = torch.cat(vel_gt_list, dim=0)
        vel_pred_cat = torch.cat(vel_pred_list, dim=0)
        vel_err = (vel_pred_cat - vel_gt_cat).square().sum(dim=-1).sqrt()
        mpjve = 100 * vel_err.mean()
        mpjve_upper = 100 * vel_err[..., amass.SMPLX_UPPER_JOINTS].mean()
        mpjve_lower = 100 * vel_err[..., amass.SMPLX_LOWER_JOINTS].mean()
        
        #Jitter
        jitter_pred_list = []
        jitter_gt_list = []
        for j_gt, j_pred in zip(joints_gt_list, joints_pred_list):
            if j_gt.shape[0] < 4:
                continue  # skip too short sequences
            jitter_pred_list.append(
                ((amass.FPS ** 3) * (j_pred[3:] - 3 * j_pred[2:-1] + 3 * j_pred[1:-2] - j_pred[:-3])).norm(dim=-1)
            )
            jitter_gt_list.append(
                ((amass.FPS ** 3) * (j_gt[3:] - 3 * j_gt[2:-1] + 3 * j_gt[1:-2] - j_gt[:-3])).norm(dim=-1)
            )

        jitter_pred_cat = torch.cat(jitter_pred_list, dim=0)
        jitter_gt_cat = torch.cat(jitter_gt_list, dim=0)

        jitter_pred = jitter_pred_cat.mean()
        jitter_gt = jitter_gt_cat.mean()
        jitter = jitter_pred / jitter_gt

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
            "MPJVE_Lower": float(mpjve_lower.detach().cpu().numpy()),
            "Jitter_Pred": float(jitter_pred.mean().detach().cpu().numpy()),
            "Jitter_Gt": float(jitter_gt.mean().detach().cpu().numpy()),
            "Jitter": float(jitter.detach().cpu().numpy())
        }

        pprint_and_save(results, model_dir, dataset_str, split)

if __name__ == '__main__':
    __main()