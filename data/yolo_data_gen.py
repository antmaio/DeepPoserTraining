# External
import os
import glob
import re
import xml.etree.ElementTree as ET

import numpy as np
import torch
import trimesh
from PIL import Image

# Internal
from body_visualizer.tools.vis_tools import colors
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from data.rendering import CheckerBoard, MeshViewer2
from data.data_config import YoloJoints

os.environ['PYOPENGL_PLATFORM'] = 'egl'

# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def extract_from_xml(file_path: str):
    """
    Parse a camera XML file and return intrinsics, extrinsics, and image size.

    Returns:
        K            (3x3 ndarray) – intrinsic matrix.
        camera_pose  (3x4 ndarray) – extrinsic / camera-matrix rows.
        image_size   (int, int)    – (width, height).
    """
    root = ET.parse(file_path).getroot()

    def _parse_matrix(tag: str) -> np.ndarray:
        text = root.find(tag).find('data').text
        return np.array([
            list(map(float, row.split()))
            for row in text.strip().split('\n')
        ])

    K = _parse_matrix('Intrinsics')
    camera_pose = _parse_matrix('CameraMatrix')
    width = int(root.find('image_width').text)
    height = int(root.find('image_height').text)
    return K, camera_pose, (width, height)


# ---------------------------------------------------------------------------
# Debug / visualisation helpers
# ---------------------------------------------------------------------------

def generate_checker_mesh() -> trimesh.Trimesh:
    generator = CheckerBoard()
    checker = generator.gen_checker_xy(generator.black, generator.white)
    return trimesh.Trimesh(checker.v, checker.f, process=False, face_colors=checker.fc)


def generate_camera_mesh(camera_path: str, camera_id: int):
    """Return a (camera_mesh, axes_mesh) tuple for the requested camera."""
    K, camera_pose, _ = extract_from_xml(f'{camera_path}Camera_{camera_id}.xml')
    camera_pose = np.vstack((camera_pose, [0, 0, 0, 1]))

    body = trimesh.creation.box(extents=(0.1, 0.05, 0.05))
    lens = trimesh.creation.cylinder(radius=0.05, height=0.08, sections=32)
    lens.apply_translation([0, 0, 0.05])
    button = trimesh.creation.icosphere(subdivisions=3, radius=0.01)
    button.apply_translation([0.02, 0.04, 0.06])

    camera_mesh = trimesh.util.concatenate([body, lens, button])
    camera_mesh.apply_transform(camera_pose)

    axes_mesh = trimesh.creation.axis(axis_length=0.2)
    axes_mesh.apply_transform(camera_pose)
    return camera_mesh, axes_mesh


def generate_gt3d_joints(global_positions: np.ndarray) -> list:
    """Return a list of red spheres representing ground-truth 3-D joint positions."""
    sphere_color = (1, 0, 0)
    spheres = []
    for joint_pos in global_positions.squeeze():
        sphere = trimesh.primitives.Sphere(radius=0.05, center=joint_pos)
        sphere.visual.face_colors = np.array(sphere_color)
        spheres.append(sphere)
    return spheres


def save_body_image(body_image: np.ndarray, cam: str):
    Image.fromarray(body_image).save(f'body_image_{cam}.png')


# ---------------------------------------------------------------------------
# Pose estimation
# ---------------------------------------------------------------------------

def _save_bad_frame(frame_path: str, fId: int, orig_file: str):
    with open('./yolo_bad_frames.txt', 'a') as f:
        f.write(f'{orig_file} {frame_path} {fId}\n')


def inference(body_image: np.ndarray, fId: int, frame_path: str, orig_file: str, **kwargs):
    """
    Run pose estimation on *body_image* and return (keypoints, confidences).

    Expects ``yolo_model`` and ``yolo_topology`` in *kwargs*.
    """
    yolo_model = kwargs.get('yolo_model')
    yolo_topology = kwargs.get('yolo_topology')

    if yolo_model is None:
        raise ValueError("No pose estimation model provided. Pass yolo_model in kwargs.")

    ret, conf = [], []
    results = yolo_model.predict(body_image, device=0, verbose=False, stream=True)
    for res in results:
        if len(res) > 0:
            ret.append(res[0].keypoints.data.cpu().numpy())
            conf.append(res[0].keypoints.conf.cpu().numpy())
        else:
            _save_bad_frame(frame_path, fId, orig_file)
            ret.append(np.zeros((1, yolo_topology.NUM_JTS, 2), dtype=float))
            conf.append(np.zeros((1, yolo_topology.NUM_JTS), dtype=float))

    return ret, conf


def _get_keypoints_by_cam(mv: MeshViewer2, cam: str, fId: int,
                          frame_path: str, orig_file: str, **kwargs):
    """Render current scene and run pose estimation for a single camera view."""
    body_image = mv.render(render_wireframe=False)
    KMat = mv.viewer._renderer._get_camera_matrices(mv.scene)
    PMat = mv.get_projection_matrix()

    ret, conf = inference(
        body_image=body_image,
        fId=fId,
        frame_path=frame_path,
        orig_file=orig_file,
        **kwargs,
    )
    return ret, conf, KMat, PMat


def _sorted_camera_files(camera_path: str) -> list:
    """Return Camera_*.xml files sorted by camera index."""
    def _cam_id(path):
        m = re.search(r'Camera_(\d+)\.xml', os.path.basename(path))
        return int(m.group(1)) if m else float('inf')

    files = glob.glob(os.path.join(camera_path, '*.xml'))
    return sorted(files, key=_cam_id)


def get_keypoints(fId: int, mv: MeshViewer2, body_pose_hand, faces: np.ndarray,
                  frame_path: str, orig_file: str, camera_path: str, **kwargs):
    """
    Render the body at frame *fId* from every camera and return 2-D keypoints.

    Returns:
        keypoints   – list of per-camera arrays.
        confidences – list of per-camera confidence arrays.
        KMats       – list of camera projection matrices.
        PMats       – list of projection matrices.
    """
    body_mesh = trimesh.Trimesh(
        vertices=c2c(body_pose_hand.v[fId]),
        faces=faces,
        vertex_colors=np.tile(colors['grey'], (6890, 1)),
    )
    mv.set_dynamic_meshes([body_mesh])

    keypoints, confidences, KMats, PMats = [], [], [], []

    for camera_file in _sorted_camera_files(camera_path):
        K, camera_pose, _ = extract_from_xml(camera_file)
        camera_pose = np.vstack((camera_pose, [0, 0, 0, 1]))
        cam = os.path.splitext(os.path.basename(camera_file))[0]

        mv.updateCam(camera_pose, K)

        results, confs, KMat, PMat = _get_keypoints_by_cam(
            mv=mv, cam=cam, fId=fId,
            frame_path=frame_path, orig_file=orig_file,
            **kwargs,
        )
        keypoints.append(results[0])
        confidences.append(confs[0])
        KMats.append(KMat)
        PMats.append(PMat)

    return keypoints, confidences, KMats, PMats


def run_yolo(mv: MeshViewer2, bm: BodyModel, body_pose_world,
             nb_frames: int, orig_file: str, frame_path: str,
             idx: int, camera_path: str, **kwargs):
    """
    Run YOLO pose estimation on every frame and return per-camera 2-D keypoints.

    Returns:
        cam_2d_arrays  – list of (T, J, 2) arrays, one per camera.
        Confidences    – (T, J, num_cameras) array.
    """
    faces = c2c(bm.f)
    cam_2d: list[list] = []
    confidences_per_frame = []

    print(f'Processing {nb_frames} frames ...')
    for frame_id in range(nb_frames + 1):
        frame_path_id = f"{frame_path}/{idx}.pkl"

        kpts, confs, _, _ = get_keypoints(
            fId=frame_id,
            mv=mv,
            body_pose_hand=body_pose_world,
            faces=faces,
            frame_path=frame_path_id,
            orig_file=orig_file,
            camera_path=camera_path,
            **kwargs,
        )
        confidences_per_frame.append(confs)

        for cam_idx, cam_kpts in enumerate(kpts):
            if len(cam_2d) <= cam_idx:
                cam_2d.append([])
            cam_2d[cam_idx].append(cam_kpts[0][:, :2])  # keep only (x, y)

    Confidences = np.array(confidences_per_frame).squeeze()
    Confidences = np.transpose(Confidences, (0, 2, 1))
    cam_2d_arrays = [np.array(cam_data) for cam_data in cam_2d]

    return cam_2d_arrays, Confidences
