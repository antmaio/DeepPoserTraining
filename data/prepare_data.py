"""
From https://github.com/zxz267/AvatarJLM
"""
#External
import os
import argparse
import torch
import logging

#Internal
from data.utils_data import process
from human_body_prior.body_model.body_model import BodyModel
from data.data_config import OUTPUT_DIR, YoloJoints
from data.utils_mmpose import _MODEL_STR_, KEYPOINTS_TOPOLOGY

def make_dst_path(cfg, subset:str,phase:str, **kwargs) -> str:
    """
    Constructs a destination path for output files based on the given protocol,
    subset, phase, and optional model specifications.

    The function uses the provided `protocol`, `subset`, and `phase` to build
    the path under the global `OUTPUT_DIR`. It conditionally includes a model
    name in the path based on the presence of `mm_pose_model` or `yolo_model`
    in `kwargs`.

    - If `mm_pose_model` is truthy, it uses a global `_MODEL_STR_`.
    - Else if `yolo_model` is not None, it uses the provided `yolo_model`.
    - Else it falls back to using `cfg.protocol`.

    Args:
        protocol (int): The protocol number to include in the path.
        subset (str): The dataset subset (e.g., "train", "test", "val").
        phase (str): The phase or subfolder name (e.g., "images", "labels").
        **kwargs: Optional keyword arguments.
            - yolo_model (str): Name of the YOLO model.
            - mm_pose_model (bool): Flag indicating if MMPose model is used.

    Returns:
        str: The constructed destination path.

    """
    assert hasattr(cfg, 'protocol'), "cfg has no attribute 'protocol'" 
    
    yolo_model = kwargs.get('yolo_model')
    mm_pose_model = kwargs.get('mm_pose_model')

    if mm_pose_model:
        logging.info('Pose Estimation with MMpose')
        dst = os.path.join(f'{OUTPUT_DIR}', f"{_MODEL_STR_}_protocol_{cfg.protocol}", subset, phase)
    elif yolo_model is not None:
        logging.info('Pose Estimation with Yolo')
        assert hasattr(cfg, 'yolo_model'), "cfg has no attribute 'yolo_model'" 
        dst = os.path.join(f'{OUTPUT_DIR}', f"{cfg.yolo_model}_protocol_{cfg.protocol}", subset, phase)
    else:
        logging.info('No external 3D data')
        dst = os.path.join(f'{OUTPUT_DIR}', f"protocol_{cfg.protocol}", subset ,phase)
    
    return dst

if __name__ == '__main__':

    # Overwrite log file every time the script runs
    logging.basicConfig(
        # filename='data_process_log.txt',
        # filemode='w',  # 'w' = overwrite
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    # --- Parser ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Path to data root.')
    parser.add_argument('--protocol', type=int, default=1, choices=[1, 2, 3], help='Prepare data mode.')
    #parser.add_argument('--yolo_model', type=str, default=None, choices=['yolov8n-pose','yolov8s-pose','yolov8l-pose','yolov8m-pose','yolov8x-pose','yolov8x-pose-p6', 'ground_truth'])
    parser.add_argument('--support_data', type=str, default='./support_data', help='Path to support data.')
    parser.add_argument('--data_split', type=str, default='./data/data_split', help='Path to data split.')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--yolo_model', type=str, default=None)
    group.add_argument('--mm_pose_model', action='store_true', default=False, help="Apply pose estimation with model in utils_mmpose if True")
    
    cfg = parser.parse_args() 

    # --- body models ---
    bm_fname_male = os.path.join(cfg.support_data, 'body_models/smplh/{}/model.npz'.format('male'))
    dmpl_fname_male = os.path.join(cfg.support_data, 'body_models/dmpls/{}/model.npz'.format('male'))
    bm_fname_female = os.path.join(cfg.support_data, 'body_models/smplh/{}/model.npz'.format('female'))
    dmpl_fname_female = os.path.join(cfg.support_data, 'body_models/dmpls/{}/model.npz'.format('female'))
    
    num_betas = 16 # number of body parameters
    num_dmpls = 8 # number of DMPL parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device= 'cpu'
    bm_male = BodyModel(bm_fname=bm_fname_male, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname_male).to(device)
    bm_female = BodyModel(bm_fname=bm_fname_female, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname_female).to(device)
    body_models = {'male': bm_male, 'female': bm_female}

    #Pose estimation model init
    kwargs = {}
    if cfg.yolo_model:
        from data.utils_yolo import init_yolo
        yolo_model = init_yolo(f'{cfg.yolo_model}.pt',)
        kwargs['yolo_model'] = yolo_model
        kwargs['yolo_model_str'] = cfg.yolo_model
        kwargs['topology'] = YoloJoints
    elif cfg.mm_pose_model:
        #ok
        from mmpose.apis import MMPoseInferencer
        inferencer = MMPoseInferencer(_MODEL_STR_, device=device)
        kwargs['mm_pose_model'] = inferencer
        kwargs['topology'] = KEYPOINTS_TOPOLOGY

    os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    if cfg.protocol in [1, 2]:
        dataset = ['BioMotionLab_NTroje', 'CMU', 'MPI_HDM05']
        for subset in dataset: 
            for phase in ['train', 'test']:
                print(subset, phase)
                split_file = os.path.join(cfg.data_split, subset, phase + "_split.txt")
                src = os.path.join(cfg.root, subset)

                dst = make_dst_path(cfg, subset, phase, **kwargs)
                """
                if cfg.yolo_model is not None:
                    dst = os.path.join(f'{OUTPUT_DIR}', f"{cfg.yolo_model}_protocol_{cfg.protocol}", subset, phase)
                else:
                    dst = os.path.join(f'{OUTPUT_DIR}', f"protocol_{cfg.protocol}", subset ,phase)
                """
                os.makedirs(dst, exist_ok=True)
                process(src, dst, body_models, logging=logging, split_file=split_file, **kwargs)

    elif cfg.protocol in [3]:
        #TODO implement :)
        raise NotImplementedError
        train_set = ['MPI_HDM05', 'BioMotionLab_NTroje', 'CMU', 'ACCAD', 'BMLmovi', 'EKUT', 'Eyes_Japan_Dataset', 'KIT', 'MPI_Limits', 'MPI_mosh', 'SFU', 'TotalCapture']
        test_set = ['HumanEva', 'Transitions_mocap']
        all_data = {**{k: 'train' for k in train_set}, **{k: 'test' for k in test_set}}
        for subset, phase in all_data.items(): 
            print(subset, phase)
            src = os.path.join(cfg.root, subset)
            dst = os.path.join(f"./data/protocol_{cfg.protocol}", subset, phase)
            os.makedirs(dst, exist_ok=True)
            process(src, dst, body_models, logging=logging, yolo_model=yolo_model, mm_pose_model=mm_pose_model)
