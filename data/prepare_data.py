"""
From https://github.com/zxz267/AvatarJLM
"""

import os
import argparse
import torch

from data.utils_data import process
from data.utils_yolo import init_yolo
from human_body_prior.body_model.body_model import BodyModel
from data.data_config import OUTPUT_DIR


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Path to data root.')
    parser.add_argument('--protocol', type=int, default=1, choices=[1, 2, 3], help='Prepare data mode.')
    parser.add_argument('--yolo_model', type=str, default=None, choices=['yolov8n-pose','yolov8s-pose','yolov8l-pose','yolov8m-pose','yolov8x-pose','yolov8x-pose-p6', 'ground_truth'])
    parser.add_argument('--support_data', type=str, default='./support_data', help='Path to support data.')
    parser.add_argument('--data_split', type=str, default='./data/data_split', help='Path to data split.')
    cfg = parser.parse_args() 

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

    if cfg.yolo_model is not None and cfg.yolo_model != 'ground_truth':
        yolo_model = init_yolo(f'{cfg.yolo_model}.pt',)
    else:
        yolo_model=None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    if cfg.protocol in [1, 2]:
        dataset = ['BioMotionLab_NTroje', 'CMU', 'MPI_HDM05']
        for subset in dataset: 
            for phase in ['train', 'test']:
                print(subset, phase)
                split_file = os.path.join(cfg.data_split, subset, phase + "_split.txt")
                src = os.path.join(cfg.root, subset)
                if cfg.yolo_model is not None:
                    dst = os.path.join(f'{OUTPUT_DIR}', f"{cfg.yolo_model}_protocol_{cfg.protocol}", subset, phase)
                else:
                    dst = os.path.join(f'{OUTPUT_DIR}', f"protocol_{cfg.protocol}", subset ,phase)
                os.makedirs(dst, exist_ok=True)
                process(src, dst, body_models, split_file, yolo_model)

    elif cfg.protocol in [3]:
        train_set = ['MPI_HDM05', 'BioMotionLab_NTroje', 'CMU', 'ACCAD', 'BMLmovi', 'EKUT', 'Eyes_Japan_Dataset', 'KIT', 'MPI_Limits', 'MPI_mosh', 'SFU', 'TotalCapture']
        test_set = ['HumanEva', 'Transitions_mocap']
        all_data = {**{k: 'train' for k in train_set}, **{k: 'test' for k in test_set}}
        for subset, phase in all_data.items(): 
            print(subset, phase)
            src = os.path.join(cfg.root, subset)
            dst = os.path.join(f"./data/protocol_{cfg.protocol}", subset, phase)
            os.makedirs(dst, exist_ok=True)
            process(src, dst, body_models, yolo_model)