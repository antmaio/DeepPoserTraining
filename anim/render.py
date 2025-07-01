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
# Internal
import models
import data
import train


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

    # --- Load model ---

    if epoch is None:
        epoch = models.get_last_model_epoch(model_dir)
    model = models.load_model(model_dir, epoch)
    model = model.to(device, dtype)
    model.eval()

    dataset_smplx_layers = data.get_dataset_smplx_layers(dataset_str)
    for layer_idx in range(len(dataset_smplx_layers)):
        dataset_smplx_layers[layer_idx] = dataset_smplx_layers[layer_idx].to(device, dtype)

    rec_names = data.get_dataset_recording_names_for_split(dataset_str, split)
    if rec_idx < 0:
        rec_idx = random.randrange(0, len(rec_names))
    assert 0 <= rec_idx < len(rec_names)
    rec_name = rec_names[rec_idx]

    if dataset_str == 'amass':
        import amass
        batch = amass.load_smplx(rec_name)
    elif dataset_str == 'egobody':
        raise NotImplementedError(f"Unknown dataset: {dataset_str}") 
        import egobody
        batch = egobody.load_smplx_30fps(rec_name, wearer=wearer)
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset_str}")


    max_num_frames = batch['transl'].shape[0]
    if (num_frames <= 0) or (num_frames > max_num_frames):
        num_frames = max_num_frames
    # Make into proper batch
    # TODO Could likely handle smplx_layer_idx more gracefully, e.g. via load_smplx* above
    batch['smplx_layer_idx'] = torch.Tensor([1 if batch['gender'] == 'female' else 0])
    batch['betas'] = batch['betas'][None, :num_frames]
    batch['transl'] = batch['transl'][None, :num_frames]
    batch['global_orient'] = batch['global_orient'][None, :num_frames]
    batch['body_pose'] = batch['body_pose'][None, :num_frames]
    if zero_betas:
        batch['betas'] = torch.zeros_like(batch['betas'])

    model_input, model_target = models.batch_to_model_input_and_target(batch, dataset_smplx_layers, device, dtype)
    model_pred = model(model_input)  # no 'reset' needed

    vertices_target_np = model_target.vertices[0].cpu().numpy()
    vertices_pred_np = model_pred.vertices[0].cpu().numpy()
    faces_target = model_target.faces
    faces_pred = model_pred.faces

    fourcc = 0x7634706d  # mp4v
    fps = data.FPS
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
        transl_target = model_target.transl[0]
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
    smplx_faces = dataset_smplx_layers[0].faces

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