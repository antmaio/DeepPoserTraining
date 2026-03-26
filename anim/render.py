"""
Rendering script — runs model inference and writes side-by-side AVI videos
comparing ground-truth and predicted body meshes.
"""
# External
import argparse
import logging
import os
import pathlib
import pathlib
import random
import tomli

import cv2
import numpy as np
import pyrender
import torch
import torch.nn.functional as F
import trimesh
from torch.utils.data import DataLoader
from tqdm import tqdm

# Internal
import data.data_config as bm_C
import anim.data.amass as amass
import anim.models as models
import anim.train as train
from anim.models.base import BaseModelOutput
from anim.train import Config
from human_body_prior.body_model.body_model import BodyModel
from utils.utils_transform import matrix_to_angle_axis

os.environ['PYOPENGL_PLATFORM'] = 'egl'

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

# ---------------------------------------------------------------------------
# Rendering constants
# ---------------------------------------------------------------------------

_PRED_COLOR   = (0.0, 1.0, 1.0, 1.0)   # cyan
_TARGET_COLOR = (0.0, 1.0, 0.0, 1.0)   # green
_FOURCC       = 0x7634706d              # mp4v / AVI

# ---------------------------------------------------------------------------
# Jitter utility
# ---------------------------------------------------------------------------

def add_jitter(
    joint_pred: torch.Tensor,
    jitter_strength: float = 1e-3,
) -> torch.Tensor:
    """
    Add high-frequency positional noise to joint trajectories.

    Args:
        joint_pred:       (B, T, J, 3)
        jitter_strength:  Amplitude of the noise.

    Returns:
        Jittered tensor with the same shape.
    """
    noise      = torch.randn_like(joint_pred) * jitter_strength
    noise_diff = noise[:, 1:] - noise[:, :-1]
    noise_diff = F.pad(noise_diff, (0, 0, 0, 0, 1, 0))   # restore T dimension
    return joint_pred + noise_diff


# ---------------------------------------------------------------------------
# Body-model utilities
# ---------------------------------------------------------------------------

def get_vertices_and_faces(
    model_output: BaseModelOutput,
    bm: BodyModel,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run a body-model forward pass and return (vertices, faces).

    Args:
        model_output: Contains global_orient, body_pose, transl (all squeezed to (T, ...)).
        bm:           BodyModel instance on the correct device.

    Returns:
        vertices: (T, V, 3) tensor.
        faces:    (F, 3) tensor.
    """
    go_aa = matrix_to_angle_axis(model_output.global_orient)
    bp_aa = matrix_to_angle_axis(model_output.body_pose)
    T     = go_aa.shape[1]

    body_out = bm(
        pose_body=bp_aa.view(T, (amass.SmplxJoints.NUM_JTS - 1) * 3),
        root_orient=go_aa.view(T, -1),
        trans=model_output.transl.squeeze(),
    )
    return body_out.v, body_out.f


# ---------------------------------------------------------------------------
# Per-frame rendering
# ---------------------------------------------------------------------------

def _make_materials():
    pred_mat   = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=_PRED_COLOR)
    target_mat = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=_TARGET_COLOR)
    return pred_mat, target_mat


def _default_camera_pose(transl_target: torch.Tensor) -> np.ndarray:
    """Position camera facing the average character position, offset back in Y."""
    avg_pos    = transl_target.cpu().numpy().mean(axis=0)
    cam_pose   = np.eye(4)
    cam_pose[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])   # +90° around X
    cam_pose[:3, 3]  = avg_pos
    cam_pose[1,   3] -= 3.0   # step back
    return cam_pose


def render_recording(
    vertices_pred:   np.ndarray,
    vertices_target: np.ndarray,
    faces:           np.ndarray,
    camera:          pyrender.Camera,
    camera_pose:     np.ndarray,
    video_path:      str,
    video_width:     int,
    video_height:    int,
    render_fps:      int,
):
    """Write one AVI video comparing predicted and target meshes frame by frame."""
    pred_mat, target_mat = _make_materials()
    light    = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    renderer = pyrender.OffscreenRenderer(video_width, video_height)
    writer   = cv2.VideoWriter(video_path, _FOURCC, render_fps, (video_width, video_height))

    for frame_idx in range(len(vertices_pred)):
        mesh_target = pyrender.Mesh.from_trimesh(
            trimesh.Trimesh(vertices_target[frame_idx], faces, process=False), target_mat)
        mesh_pred   = pyrender.Mesh.from_trimesh(
            trimesh.Trimesh(vertices_pred[frame_idx],   faces, process=False), pred_mat)

        scene = pyrender.Scene(bg_color=(0, 0, 0, 0), ambient_light=(1, 1, 1))
        scene.add(camera, pose=camera_pose)
        scene.add(light,  pose=camera_pose)
        scene.add(mesh_target)
        scene.add(mesh_pred)

        img_rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        writer.write(img_rgba[..., (2, 1, 0)])   # RGBA → BGR

    writer.release()
    renderer.delete()
    logging.info(f"Rendered → {video_path}")





# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _build_render_config(args, model_dir: str) -> Config:
    """
    Build a Config from the saved run_config and print the unified summary.
    """
    cfg_dict = train.get_model_run_config(model_dir)
    if cfg_dict is None:
        raise FileNotFoundError(f'Missing run_config.json in {model_dir}')
        
    cfg_dict['model_config'] = os.path.join(model_dir, 'model_config.toml')
    cfg      = Config(cfg_dict)

    if not hasattr(cfg, 'data_dir'):
        protocol = getattr(cfg, 'protocol', 1)
        str_prot = 1 if protocol in (1, 2) else 3
        y_model  = getattr(cfg, 'yolo_model', 'yolov8n-pose')
        cfg.data_dir = f'./data/keypoints/{y_model}_protocol_{str_prot}'
        if getattr(cfg, 'skinned_mesh_topology', 'smpl') == 'smplx':
            cfg.data_dir += '_smplx'
            
    if not hasattr(cfg, 'cache_dir'):
        cfg.cache_dir = '.cache'

    try:
        with open(cfg.model_config, 'rb') as fp:
            model_cfg_dict = tomli.load(fp)
    except Exception:
        model_cfg_dict = None

    train.log_config(cfg.__dict__, is_train=False, model_cfg_dict=model_cfg_dict)

    return cfg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    torch.autograd.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description='Render model predictions vs. ground truth.')
    parser.add_argument('model_dir',  type=str)
    parser.add_argument('split',      type=str, choices=('train', 'val', 'test', 'full'))
    parser.add_argument('--render_dir',          type=str,   default='./renders')
    parser.add_argument('--epoch',               type=int,   default=None)
    parser.add_argument('--rec_idx',             type=int,   nargs='+', default=None)
    parser.add_argument('--num_frames',          type=int,   default=-1)
    parser.add_argument('--render_time_scale',   type=float, default=1.0,
                        help='Scale factor applied to FPS of the rendered video.')
    args = parser.parse_args()

    cfg = _build_render_config(args, args.model_dir)

    device   = torch.device(getattr(cfg, 'device_str', 'cuda'))
    dtype    = torch.float32

    dataset_str  = getattr(cfg, 'dataset',               'amass-p1')
    topology     = getattr(cfg, 'skinned_mesh_topology',  'smpl')
    win_len      = getattr(cfg, 'win_len',                40)
    win_overlap  = getattr(cfg, 'win_overlap',            5)
    zero_betas   = getattr(cfg, 'zero_betas',             False)

    # ---- dataset ----
    dataset    = amass.get_dataset(cfg, dataset_str, args.split,
                                   topology=topology,
                                   win_len=win_len,
                                   win_overlap=win_overlap,
                                   zero_betas=zero_betas)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # ---- model ----
    epoch = args.epoch or models.get_last_model_epoch(args.model_dir)
    model = models.load_model(args.model_dir, epoch, len(dataloader), device=device)
    model = model.to(device, dtype)
    model.eval()

    # ---- recording selection ----
    rec_names = sorted(amass.get_dataset_recording_names_for_split(
        cfg, dataset_str, args.split, topology=topology))
    if args.rec_idx is not None:
        #if rec_idx == -1, select a random recording
        rec_idx = (
            [random.randrange(len(rec_names))]
            if len(args.rec_idx) == 1 and args.rec_idx[0] < 0
            else args.rec_idx 
        )
        for r in rec_idx:
            assert 0 <= r < len(rec_names), f'rec_idx {r} out of range.'
            logging.info(f'Processing recording {rec_names[r]}')
    else:
        #if rec_idx is None, select all recordings
        rec_idx = list(range(len(rec_names)))
        logging.info('Processing all recordings.')

    # ---- body models ----
    bm_male   = BodyModel(bm_fname=bm_C._BM_FNAME_MALE_,
                          num_betas=bm_C._NUM_BETAS_, num_dmpls=bm_C._NUM_DMPLS_,
                          dmpl_fname=bm_C._DMPL_FNAME_MALE_).to(device)
    bm_female = BodyModel(bm_fname=bm_C._BM_FNAME_FEMALE_,
                          num_betas=bm_C._NUM_BETAS_, num_dmpls=bm_C._NUM_DMPLS_,
                          dmpl_fname=bm_C._DMPL_FNAME_FEMALE_).to(device)

    render_fps = int(args.render_time_scale * amass.FPS)
    os.makedirs(args.render_dir, exist_ok=True)

    for r_id, batch in tqdm(enumerate(dataloader)):
        if r_id not in rec_idx:
            continue

        with torch.no_grad():
            model_input, model_target = models.batch_to_model_input_and_target(
                batch, device, dtype, mode3d=model.mode3d)
            model.reset()
            model_pred = model(model_input)

        n_frames = model_pred.joints.shape[1]
        bm = bm_male if model_input.gender == amass.Gender.MALE else bm_female

        v_pred,   f_pred   = get_vertices_and_faces(model_pred,   bm)
        v_target, f_target = get_vertices_and_faces(model_target, bm)

        # ---- camera ----
        camera      = pyrender.PerspectiveCamera(np.deg2rad(90.0))
        transl      = batch['body_parms_list']['trans'].squeeze()
        camera_pose = _default_camera_pose(transl)
        video_size  = (1024, 1024)

        # ---- output path ----
        model_name = pathlib.Path(args.model_dir).name
        video_path = os.path.join(
            args.render_dir,
            f'render_{dataset_str}_{args.split}_{r_id}_{n_frames}_{model_name}.avi',
        )

        render_recording(
            vertices_pred=v_pred.cpu().numpy(),
            vertices_target=v_target.cpu().numpy(),
            faces=f_pred.cpu().numpy(),
            camera=camera,
            camera_pose=camera_pose,
            video_path=video_path,
            video_width=video_size[0],
            video_height=video_size[1],
            render_fps=render_fps,
        )


if __name__ == '__main__':
    main()
