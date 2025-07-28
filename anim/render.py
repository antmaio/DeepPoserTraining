# External
import pathlib
import pyrender
import random
import torch
import cv2
import argparse
import numpy as np
import os
import trimesh
from tqdm import tqdm
import logging
# Internal
import anim.models as models
import anim.train as train
import anim.data.amass as amass
import anim.bm_config as bm_C
from anim.models.base import BaseModel
from human_body_prior.body_model.body_model import BodyModel
from utils.utils_transform import matrix_to_angle_axis
import config 

os.environ['PYOPENGL_PLATFORM'] = 'egl'

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def get_vertices_and_faces(model_base:BaseModel, bm:BodyModel)->torch.Tensor:

    global_orient_aa = matrix_to_angle_axis(model_base.global_orient)
    body_pose_aa = matrix_to_angle_axis(model_base.body_pose)
    num_frames = global_orient_aa.shape[1] #assuming batch tensors
    body_parms = {
        'pose_body': body_pose_aa.view(num_frames, (amass.SmplxJoints.NUM_JTS-1)*3),
        'root_orient' : global_orient_aa.view(num_frames, -1)
    }
    body_pose_local= bm(**{k:v for k, v in body_parms.items() if k in ['pose_body', 'root_orient']})
    return body_pose_local.v, body_pose_local.f

def __main():
    _ = torch.autograd.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str)
    parser.add_argument('dataset', type=str, choices=('egobody', 'amass-p1', 'amass-p2'))
    parser.add_argument('split', type=str, choices=('train', 'val', 'test', 'full'),
                        help="Split of dataset to evaluate and/or render")
    parser.add_argument('--render_dir', type=str, default='./renders')
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--rec_idx', type=int, default=-1)
    parser.add_argument('--wearer', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--num_frames', type=int, default=-1)
    parser.add_argument('--zero_betas', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--render_time_scale', type=float, default=1.0,
                        help="Factor for scaling FPS of rendered video")
    parser.add_argument('--device_str', type=str, default='cuda')

    args = parser.parse_args()
    model_dir = args.model_dir
    dataset_str = args.dataset
    split = args.split
    render_dir = args.render_dir
    epoch = args.epoch
    zero_betas = args.zero_betas
    rec_idx = args.rec_idx
    wearer = args.wearer
    num_frames = args.num_frames
    device = torch.device(args.device_str)
    dtype = torch.float32

    # Use relevant parameters from last training run if otherwise unspecified
    train_info = train.get_model_train_info(model_dir)
    if zero_betas is None:
        zero_betas = train_info['zero_betas']
    config_info = train.get_model_config_info(model_dir)
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

    # --- Load model ---

    if epoch is None:
        epoch = models.get_last_model_epoch(model_dir)
    model = models.load_model(model_dir, epoch)
    model = model.to(device, dtype)
    model.eval()

    rec_names = amass.get_dataset_recording_names_for_split(config, dataset_str, split)
    if rec_idx < 0:
        rec_idx = random.randrange(0, len(rec_names))
    assert 0 <= rec_idx < len(rec_names)
    rec_name = rec_names[rec_idx]

    if 'amass' in dataset_str:
        batch = amass.load_smpl(config, rec_name)
    elif dataset_str == 'egobody':
        raise NotImplementedError(f"Unknown dataset: {dataset_str}") 
        import egobody
        batch = egobody.load_smplx_30fps(rec_name, wearer=wearer)
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset_str}")


    body_parms_list = batch['body_parms_list']
    max_num_frames = batch['rotations_local_full_gt_list'].shape[0]
    if (num_frames <= 0) or (num_frames > max_num_frames):
        num_frames = max_num_frames

    # Make into proper batch
    batch['betas'] = batch['betas'][None, :num_frames]
    if zero_betas:
        batch['betas'] = torch.zeros_like(batch['betas'])
    batch['rotations_local_full_gt_list'] = batch['rotations_local_full_gt_list'][None, :num_frames]
    batch['hmd_position_global_full_gt_list'] = batch['hmd_position_global_full_gt_list'][None, :num_frames]
    batch['head_global_trans_list'] = batch['head_global_trans_list'][None, :num_frames]
    batch['gender'] = batch['gender'][None]
    batch['keypoints'] = batch['keypoints'][None, :num_frames]
    batch['conf'] = batch['conf'][None, :num_frames]
     
    bm_male = BodyModel(bm_fname=bm_C._BM_FNAME_MALE_, num_betas=bm_C._NUM_BETAS_, num_dmpls=bm_C._NUM_DMPLS_, dmpl_fname=bm_C._DMPL_FNAME_MALE_).to(device)
    bm_female = BodyModel(bm_fname=bm_C._BM_FNAME_MALE_, num_betas=bm_C._NUM_BETAS_, num_dmpls=bm_C._NUM_DMPLS_, dmpl_fname=bm_C._DMPL_FNAME_FEMALE_).to(device)

    # Inference and process output and target
    model_input, model_target = models.batch_to_model_input_and_target(batch, device, dtype, mode3d=model.mode3d)
    model_pred = model(model_input)  # no 'reset' needed
    bm = bm_male if model_input.gender == amass.Gender.MALE else bm_female
    #Pred 
    vertices_pred, faces_pred = get_vertices_and_faces(model_pred, bm)
    #Target
    vertices_target, faces_target = get_vertices_and_faces(model_target, bm)
    
    vertices_target_np = vertices_target.cpu().numpy()
    vertices_pred_np = vertices_pred.cpu().numpy()
    faces_target = faces_target.cpu().numpy()
    faces_pred = faces_pred.cpu().numpy()

    fourcc = 0x7634706d  # mp4v
    fps = amass.FPS
    render_fps = int(args.render_time_scale * fps)
    model_dir_name = pathlib.Path(model_dir).name
    video_path = os.path.join(
        render_dir, f"render_{dataset_str}_{split}_{rec_idx}_{num_frames}_{model_dir_name}.avi")
    os.makedirs(render_dir, exist_ok=True)

    kinect = None
    start_frame = -1
    if dataset_str == 'egobody':
        import egobody
        rec = egobody.get_recording_by_name(rec_name)
        kinect_idx = 0  # Pick master; TODO Could parameterise/randomise
        kinect = rec.kinects[kinect_idx]
        start_frame = rec.start_frame
        camera, camera_pose = egobody.get_kinect_pyrender_camera_and_pose(kinect)
        video_width = egobody.COLOR_WIDTH
        video_height = egobody.COLOR_HEIGHT
    else:
        camera = pyrender.PerspectiveCamera(np.deg2rad(60.0))
        #transl_target = model_target.transl[0]
        #TODO fix camera_pose
        transl_target = body_parms_list['trans']      
        transl_target_avg_np = transl_target.cpu().numpy().mean(axis=0)
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = transl_target_avg_np
        camera_pose[2, 3] += 3.0
        video_width = 512  # TODO Could increase resolution
        video_height = 512
    video_writer = cv2.VideoWriter(video_path, fourcc, render_fps, (video_width, video_height))

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4)
    pred_color = (0.0, 1.0, 1.0, 1.0)
    pred_mat = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=pred_color)
    target_color = (0.0, 1.0, 0.0, 1.0)
    target_mat = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=target_color)
    renderer = pyrender.OffscreenRenderer(video_width, video_height)

    for frame_idx in tqdm(range(num_frames)):
        mesh_target = trimesh.Trimesh(vertices_target_np[frame_idx], faces_target, process=False)
        mesh_target = pyrender.Mesh.from_trimesh(mesh_target, target_mat)
        mesh_pred = trimesh.Trimesh(vertices_pred_np[frame_idx], faces_pred, process=False)
        mesh_pred = pyrender.Mesh.from_trimesh(mesh_pred, pred_mat)

        scene = pyrender.Scene(bg_color=(0, 0, 0, 0), ambient_light=(1, 1, 1))
        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        scene.add(mesh_target)
        scene.add(mesh_pred)

        img_rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        if dataset_str == 'egobody':
            import egobody
            img_background_bgr = cv2.undistort(
                egobody.load_color_bgr(rec_name, kinect.name, egobody.frame_to_string(start_frame + frame_idx)),
                kinect.intrinsics.camera_mtx,
                kinect.intrinsics.k)
            alpha = 0.6
            mask = img_rgba[..., 3] > 0
            img_bgr = img_background_bgr.copy()
            img_bgr[mask] = alpha * img_rgba[mask][..., (2, 1, 0)] + (1 - alpha) * img_background_bgr[mask]
        else:
            img_bgr = img_rgba[..., (2, 1, 0)]

        video_writer.write(img_bgr)

    video_writer.release()
    print(f"Rendered '{video_path}'")


if __name__ == "__main__":
    __main()