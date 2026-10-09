"""Render ground-truth vs. predicted body meshes as side-by-side comparison videos.

Loads a trained animation model and a dataset split, runs inference on each
selected recording, converts the ground-truth and predicted SMPL(-X) parameters
into body meshes, and renders both meshes in the same scene (target in green,
prediction in cyan). One AVI video is written per recording.

Example:
    python render.py runs/my_model test --rec_idx 0 5 12 --render_dir ./renders
    python render.py runs/my_model test --rec_idx -1      # one random recording
"""
# Must be set before pyrender / OpenGL are imported to select headless rendering.
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# External
import argparse
import logging
import pathlib
import random
import tomli

import cv2
import numpy as np
import pyrender
import torch
import trimesh
from torch.utils.data import DataLoader
from tqdm import tqdm

# Internal
import data.data_config as dconfig
import anim.data.amass as amass
import anim.models as models
import anim.train as train
from anim.models.base import BaseModelOutput
from anim.train import Config
from human_body_prior.body_model.body_model import BodyModel
from utils.utils_transform import matrix_to_angle_axis

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
# Body-model utilities
# ---------------------------------------------------------------------------

def get_vertices_and_faces(
    model_output: BaseModelOutput,
    bm: BodyModel,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a body-model forward pass and return the resulting mesh.

    Rotations are converted from rotation matrices to axis-angle before being
    passed to the body model.

    Args:
        model_output: Model prediction or ground-truth target, with a leading
            batch dimension of 1. Uses `global_orient` (root rotation matrices),
            `body_pose` (rotation matrices of the body joints, root excluded)
            and `transl` (root translation per frame).
        bm: SMPL body model (male or female) on the same device as the tensors.

    Returns:
        A tuple `(vertices, faces)`:
            - vertices: Tensor of shape (T, V, 3), mesh vertices for each of
              the T frames.
            - faces: Tensor of shape (F, 3), triangle indices shared by all
              frames.
    """
    go_aa = matrix_to_angle_axis(model_output.global_orient)
    bp_aa = matrix_to_angle_axis(model_output.body_pose)
    T     = go_aa.shape[1]

    body_out = bm(
        pose_body=bp_aa.view(T, (dconfig.SmplxJoints.NUM_JTS - 1) * 3),
        root_orient=go_aa.view(T, -1),
        trans=model_output.transl.squeeze(),
    )
    return body_out.v, body_out.f


# ---------------------------------------------------------------------------
# Per-frame rendering
# ---------------------------------------------------------------------------

def _make_materials() -> tuple[pyrender.MetallicRoughnessMaterial,
                               pyrender.MetallicRoughnessMaterial]:
    """Create the pyrender materials used to tell the two meshes apart.

    Returns:
        A tuple `(pred_mat, target_mat)` of opaque, non-metallic materials,
        colored cyan for the prediction and green for the ground truth.
    """
    pred_mat   = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=_PRED_COLOR)
    target_mat = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=_TARGET_COLOR)
    return pred_mat, target_mat


def _default_camera_pose(transl_target: torch.Tensor) -> np.ndarray:
    """Compute a fixed camera pose that frames the whole recording.

    The camera is placed at the mean root position of the target motion,
    rotated +90° about the X axis, then moved back 3 m along the Y axis.

    Args:
        transl_target: Ground-truth root translations, shape (T, 3).

    Returns:
        A 4x4 camera pose matrix as a NumPy array.
    """
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
) -> None:
    """Render a recording frame by frame and write it to a video file.

    For each frame, the target mesh (green) and predicted mesh (cyan) are added
    to the same scene, rendered off-screen with one directional light, and
    appended to the video. The renderer and the video writer are released at
    the end.

    Args:
        vertices_pred: Predicted mesh vertices, shape (T, V, 3).
        vertices_target: Ground-truth mesh vertices, shape (T, V, 3).
        faces: Triangle indices shared by both meshes, shape (F, 3).
        camera: pyrender camera used for the whole video.
        camera_pose: 4x4 camera pose matrix.
        video_path: Output path of the AVI file.
        video_width: Frame width in pixels.
        video_height: Frame height in pixels.
        render_fps: Frame rate of the output video.
    """
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

def _build_render_config(model_dir: str) -> Config:
    """Rebuild the training configuration of a saved model for rendering.

    Reads `run_config.json` from the model directory and points `model_config`
    to the saved `model_config.toml`. Fills in defaults for fields that older
    runs may lack: `data_dir` (derived from the protocol, YOLO model and mesh
    topology) and `cache_dir` (`.cache`). The resulting configuration is
    logged.

    Args:
        model_dir: Directory containing the saved model, `run_config.json` and
            `model_config.toml`.

    Returns:
        The reconstructed `Config` object.

    Raises:
        FileNotFoundError: If `run_config.json` is missing from `model_dir`.
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

def main() -> None:
    """Command-line entry point: render model predictions vs. ground truth.

    Pipeline:
        1. Parse arguments and rebuild the saved model configuration.
        2. Load the requested dataset split and the model checkpoint (the last
           epoch unless `--epoch` is given).
        3. Select recordings: those listed in `--rec_idx`, one random recording
           if a single negative index is given, or all recordings if omitted.
        4. For each selected recording, run inference, build predicted and
           target meshes with the matching male or female body model, and
           render them with `render_recording`.
        5. Save the videos as
           `render_<dataset>_<split>_<id>_<n_frames>_<model_name>.avi` in
           `--render_dir`.

    Command-line arguments:
        model_dir: Directory of the trained model.
        split: Dataset split to render (`train`, `val`, `test` or `full`).
        --render_dir: Output directory for the videos (default `./renders`).
        --epoch: Checkpoint epoch to load (default: last available).
        --rec_idx: Indices of the recordings to render; a single negative
            value picks a random one (default: all).
        --num_frames: Maximum number of frames to render per recording
            (default -1: all frames).
        --render_time_scale: Multiplier applied to the dataset FPS for the
            output video, e.g. 0.5 for slow motion (default 1.0).
    """
    torch.autograd.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description='Render model predictions vs. ground truth.')
    parser.add_argument('model_dir',  type=str)
    parser.add_argument('split',      type=str, choices=('train', 'val', 'test', 'full'))
    parser.add_argument('--render_dir',          type=str,   default='./renders')
    parser.add_argument('--epoch',               type=int,   default=None)
    parser.add_argument('--rec_idx',             type=int,   nargs='+', default=None)
    parser.add_argument('--num_frames',          type=int,   default=-1,
                        help='Maximum number of frames to render per recording (-1: all).')
    parser.add_argument('--render_time_scale',   type=float, default=1.0,
                        help='Scale factor applied to FPS of the rendered video.')
    args = parser.parse_args()

    cfg = _build_render_config(args.model_dir)

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
    epoch = args.epoch if args.epoch is not None else models.get_last_model_epoch(args.model_dir)
    model = models.load_model(args.model_dir, epoch, len(dataloader), device=device)
    model = model.to(device, dtype)
    model.eval()

    # ---- recording selection ----
    rec_names = sorted(amass.get_dataset_recording_names_for_split(
        cfg, dataset_str, args.split, topology=topology))
    if args.rec_idx is not None:
        # A single negative index selects a random recording.
        rec_idx = (
            [random.randrange(len(rec_names))]
            if len(args.rec_idx) == 1 and args.rec_idx[0] < 0
            else args.rec_idx
        )
        for r in rec_idx:
            assert 0 <= r < len(rec_names), f'rec_idx {r} out of range.'
            logging.info(f'Processing recording {rec_names[r]}')
    else:
        # No index given: process all recordings.
        rec_idx = list(range(len(rec_names)))
        logging.info('Processing all recordings.')

    # ---- body models ----
    bm_male   = BodyModel(bm_fname=dconfig._BM_FNAME_MALE_,
                          num_betas=dconfig._NUM_BETAS_, num_dmpls=dconfig._NUM_DMPLS_,
                          dmpl_fname=dconfig._DMPL_FNAME_MALE_).to(device)
    bm_female = BodyModel(bm_fname=dconfig._BM_FNAME_FEMALE_,
                          num_betas=dconfig._NUM_BETAS_, num_dmpls=dconfig._NUM_DMPLS_,
                          dmpl_fname=dconfig._DMPL_FNAME_FEMALE_).to(device)

    render_fps = int(args.render_time_scale * dconfig.FPS)
    os.makedirs(args.render_dir, exist_ok=True)

    for r_id, batch in tqdm(enumerate(dataloader)):
        if r_id not in rec_idx:
            continue

        with torch.no_grad():
            model_input, model_target = models.batch_to_model_input_and_target(
                batch, device, dtype, mode3d=model.mode3d)
            model.reset()
            model_pred = model(model_input)

        bm = bm_male if model_input.gender == amass.Gender.MALE else bm_female

        v_pred,   faces = get_vertices_and_faces(model_pred,   bm)
        v_target, _     = get_vertices_and_faces(model_target, bm)

        # ---- optional frame limit ----
        if args.num_frames > 0:
            v_pred   = v_pred[:args.num_frames]
            v_target = v_target[:args.num_frames]
        n_frames = v_pred.shape[0]

        # ---- camera ----
        camera      = pyrender.PerspectiveCamera(np.deg2rad(90.0))
        transl      = batch['body_parms_list']['trans'].squeeze()
        camera_pose = _default_camera_pose(transl[:n_frames])
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
            faces=faces.cpu().numpy(),
            camera=camera,
            camera_pose=camera_pose,
            video_path=video_path,
            video_width=video_size[0],
            video_height=video_size[1],
            render_fps=render_fps,
        )


if __name__ == '__main__':
    main()