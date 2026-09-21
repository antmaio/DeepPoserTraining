"""
Multi-camera rendering and 2D pose-estimation helpers.

This module renders a posed SMPL/SMPLX body mesh from a set of calibrated
virtual cameras (defined by per-camera XML calibration files) and runs a
YOLO pose-estimation model on each rendered view to produce 2D keypoints
and confidences. It also contains small debug/visualization helpers (a
checkerboard ground plane, camera frustum meshes, ground-truth joint
spheres) used when inspecting a scene.

Typical usage, per processed sequence::

    cam_2d_arrays, confidences = run_yolo(
        mv=mesh_viewer,
        bm=body_model,
        body_pose_world=body_model_output,
        nb_frames=num_frames,
        orig_file=npz_path,
        frame_path=output_dir,
        idx=sequence_index,
        camera_path=camera_xml_dir,
        yolo_model=yolo_model,
        yolo_topology=yolo_topology,
    )
"""

# External
import os
import glob
import re
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from PIL import Image

# Internal
from body_visualizer.tools.vis_tools import colors
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from data.rendering import CheckerBoard, MeshViewer2

os.environ['PYOPENGL_PLATFORM'] = 'egl'

# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def extract_from_xml(file_path: str):
    """Parse a camera XML file and return intrinsics, extrinsics, and image size.

    Args:
        file_path: Path to a camera calibration ``.xml`` file containing
            ``Intrinsics`` and ``CameraMatrix`` elements (each with a nested
            ``data`` element of whitespace/newline-separated numbers) plus
            ``image_width`` and ``image_height`` elements.

    Returns:
        A tuple of:
            K: (3, 3) ndarray, the camera intrinsic matrix.
            camera_pose: (3, 4) ndarray, the extrinsic camera-matrix rows
                (rotation + translation, without the trailing homogeneous row).
            image_size: ``(width, height)`` tuple of ints.
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
    """Build a black-and-white checkerboard ground-plane mesh.

    Useful as a visual reference plane when debugging scene/camera setup.

    Returns:
        A ``trimesh.Trimesh`` of the checkerboard, with per-face colors
        already baked in.
    """
    generator = CheckerBoard()
    checker = generator.gen_checker_xy(generator.black, generator.white)
    return trimesh.Trimesh(checker.v, checker.f, process=False, face_colors=checker.fc)


def generate_camera_mesh(camera_path: str, camera_id: int):
    """Build a simple visual mesh (body + lens + button) for a calibrated camera.

    Reads the camera's pose from its XML calibration file and builds a small
    stylized camera model (box body, cylindrical lens, spherical button) plus
    a set of coordinate axes, both transformed into the camera's world pose.
    Useful for visually verifying camera placement/orientation in a scene.

    Args:
        camera_path: Directory prefix under which ``Camera_{camera_id}.xml``
            lives (note: no path separator is inserted, so this should
            already end in one if needed, matching the original call
            convention ``f'{camera_path}Camera_{camera_id}.xml'``).
        camera_id: Integer camera index used to build the XML filename.

    Returns:
        A tuple of:
            camera_mesh: ``trimesh.Trimesh``, the stylized camera body model,
                transformed to the camera's world pose.
            axes_mesh: ``trimesh.Trimesh``, a set of RGB coordinate axes at
                the camera's world pose.
    """
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
    """Build a set of red sphere meshes marking ground-truth 3D joint positions.

    Args:
        global_positions: Array of joint positions, any leading shape that
            squeezes down to ``(J, 3)`` (e.g. ``(1, J, 3)`` or ``(J, 3)``).

    Returns:
        A list of ``trimesh.primitives.Sphere`` objects, one per joint,
        each colored red and centered at its corresponding joint position.
    """
    sphere_color = (1, 0, 0)
    spheres = []
    for joint_pos in global_positions.squeeze():
        sphere = trimesh.primitives.Sphere(radius=0.05, center=joint_pos)
        sphere.visual.face_colors = np.array(sphere_color)
        spheres.append(sphere)
    return spheres


def save_body_image(body_image: np.ndarray, cam: str):
    """Save a rendered body image to disk for debugging.

    Args:
        body_image: (H, W, 3) or (H, W, 4) uint8 image array, as returned by
            the mesh viewer's render call.
        cam: Camera identifier used to build the output filename
            (``body_image_{cam}.png``), written to the current working
            directory.
    """
    Image.fromarray(body_image).save(f'body_image_{cam}.png')


# ---------------------------------------------------------------------------
# Pose estimation
# ---------------------------------------------------------------------------

def _save_bad_frame(frame_path: str, fId: int, orig_file: str):
    """Append a record of a frame with no detected pose to a bad-frames log.

    Args:
        frame_path: Path identifying the output frame/sequence being processed.
        fId: Frame index within the sequence.
        orig_file: Path to the original source mocap file, for traceability.

    The record is appended (one line per bad frame) to ``./yolo_bad_frames.txt``
    in the current working directory.
    """
    with open('./yolo_bad_frames.txt', 'a') as f:
        f.write(f'{orig_file} {frame_path} {fId}\n')


def inference(body_image: np.ndarray, fId: int, frame_path: str, orig_file: str, **kwargs):
    """Run YOLO pose estimation on a single rendered image.

    Args:
        body_image: Rendered RGB(A) image to run pose estimation on.
        fId: Frame index, used only for bad-frame logging.
        frame_path: Frame/sequence identifier, used only for bad-frame logging.
        orig_file: Source mocap file path, used only for bad-frame logging.
        **kwargs: Must include:
            yolo_model: An Ultralytics-style YOLO pose model exposing
                ``.predict(...)``.
            yolo_topology: Object exposing ``NUM_JTS`` (number of keypoints
                expected), used to build a zero-filled fallback when no
                detection is found.

    Returns:
        A tuple of:
            ret: List (length 1, since one image is passed per call) of
                (1, NUM_JTS, 2) keypoint arrays. If no detection was found,
                this is an all-zero placeholder and the frame is logged as
                a bad frame.
            conf: List (length 1) of (1, NUM_JTS) confidence arrays,
                similarly zero-filled on missed detections.

    Raises:
        ValueError: If ``yolo_model`` is not supplied in ``kwargs``.
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
    """Render the current scene from the viewer's active camera and run pose estimation.

    Args:
        mv: The ``MeshViewer2`` instance, already positioned at the desired
            camera (via ``mv.updateCam``) and holding the current scene.
        cam: Camera identifier, forwarded to ``inference`` for bad-frame logging.
        fId: Frame index, forwarded to ``inference`` for bad-frame logging.
        frame_path: Frame/sequence identifier, forwarded to ``inference``.
        orig_file: Source mocap file path, forwarded to ``inference``.
        **kwargs: Forwarded to ``inference`` (must include ``yolo_model``,
            ``yolo_topology``).

    Returns:
        A tuple of:
            ret: Keypoints returned by ``inference`` for this single view.
            conf: Confidences returned by ``inference`` for this single view.
            KMat: Camera matrices as reported by the viewer's renderer.
            PMat: The viewer's current projection matrix.
    """
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
    """List a directory's ``Camera_*.xml`` calibration files in camera-index order.

    Args:
        camera_path: Directory to search for ``*.xml`` calibration files.

    Returns:
        List of file paths, sorted numerically by the integer camera index
        parsed from each filename (``Camera_{index}.xml``). Files that don't
        match the expected naming pattern are sorted last.
    """
    def _cam_id(path):
        m = re.search(r'Camera_(\d+)\.xml', os.path.basename(path))
        return int(m.group(1)) if m else float('inf')

    files = glob.glob(os.path.join(camera_path, '*.xml'))
    return sorted(files, key=_cam_id)


def get_keypoints(fId: int, mv: MeshViewer2, body_pose_hand, faces: np.ndarray,
                   frame_path: str, orig_file: str, camera_path: str, **kwargs):
    """Render the body at one frame from every calibrated camera and run pose estimation.

    Builds a colored body mesh for frame ``fId``, then for each camera XML
    file found under ``camera_path`` (in sorted order), repositions the
    viewer to that camera and runs pose estimation on the resulting render.

    Args:
        fId: Frame index into ``body_pose_hand.v`` to render.
        mv: The ``MeshViewer2`` instance used for rendering.
        body_pose_hand: Body-model forward-pass output exposing per-frame
            vertices via ``.v[fId]``.
        faces: (F, 3) array of mesh face indices, shared across frames.
        frame_path: Frame/sequence identifier, forwarded for bad-frame logging.
        orig_file: Source mocap file path, forwarded for bad-frame logging.
        camera_path: Directory containing ``Camera_*.xml`` calibration files.
        **kwargs: Forwarded to pose inference (must include ``yolo_model``,
            ``yolo_topology``).

    Returns:
        A tuple of:
            keypoints: List of per-camera keypoint arrays (one entry per
                camera file, in sorted camera order).
            confidences: List of per-camera confidence arrays.
            KMats: List of per-camera camera matrices from the renderer.
            PMats: List of per-camera projection matrices.
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
    """Run YOLO pose estimation across every frame and every calibrated camera.

    Iterates over ``nb_frames + 1`` frames of ``body_pose_world`` (i.e.
    frames ``0`` through ``nb_frames`` inclusive), rendering and running pose
    estimation from each camera under ``camera_path`` at every frame, and
    collects the resulting 2D keypoints and confidences per camera.

    Args:
        mv: The ``MeshViewer2`` instance used for rendering.
        bm: Body model whose face topology (``bm.f``) defines the mesh
            connectivity for every frame.
        body_pose_world: Body-model forward-pass output exposing per-frame
            vertices via ``.v[frame_id]``, used by ``get_keypoints``.
        nb_frames: Number of frames to process (the loop runs
            ``nb_frames + 1`` times, i.e. frames ``0..nb_frames`` inclusive).
        orig_file: Source mocap file path, forwarded for bad-frame logging.
        frame_path: Output directory for the current sequence; combined with
            ``idx`` to build a per-sequence identifier for bad-frame logging.
        idx: Integer index of the current sequence, used to build the
            per-sequence identifier (``{frame_path}/{idx}.pkl``).
        camera_path: Directory containing ``Camera_*.xml`` calibration files.
        **kwargs: Forwarded to pose inference (must include ``yolo_model``,
            ``yolo_topology``).

    Returns:
        A tuple of:
            cam_2d_arrays: List of (T, J, 2) arrays (x, y keypoints only,
                confidence dropped), one array per camera, where T is the
                number of frames processed.
            Confidences: (T, num_cameras, J) array of per-frame, per-camera,
                per-joint confidences.
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