"""
Inspired by https://github.com/zxz267/AvatarJLM
"""
# External
import os
import glob
import pickle

import numpy as np
import torch
from sklearn.cluster import DBSCAN

# Internal
from human_body_prior.tools.rotation_tools import aa2matrot, local2global_pose
from data.yolo_data_gen import run_yolo
from data.data_config import SMPL_JOINTS
from utils import utils_transform
from utils import _utils_transform
from anim.data.amass import SmplxJoints

# ---------------------------------------------------------------------------
# Global thresholds
# ---------------------------------------------------------------------------

DISCARD_TERRAIN_SEQUENCES = True   # discard sequences where person steps onto objects
DISCARD_SHORTER_THAN = 1.0         # seconds

FLOOR_VEL_THRESH = 0.005
FLOOR_HEIGHT_OFFSET = 0.01

CONTACT_VEL_THRESH = 0.005
CONTACT_TOE_HEIGHT_THRESH = 0.04
CONTACT_ANKLE_HEIGHT_THRESH = 0.08

TERRAIN_HEIGHT_THRESH = 0.04
ROOT_HEIGHT_THRESH = 0.04
CLUSTER_SIZE_THRESH = 0.25         # fraction of FPS


# ---------------------------------------------------------------------------
# File-path helpers
# ---------------------------------------------------------------------------

def old_to_new_func(path: str) -> str:
    """Translate legacy SMPL file-path conventions to SMPLX conventions."""
    return (
        path.rstrip('\n')
            .replace('BioMotionLab_NTroje', 'BMLrub')
            .replace('MPI_HDM05', 'HDM05')
            .replace('poses', 'stageii')
            .replace('/', os.sep)
    )


def exclusion_func(path: str, verbose: bool = False) -> bool:
    """Return False (exclude) if *path* contains a known bad subject name."""
    for name in ('Rory', 'Justin', 'rory', 'justin', 'neutral'):
        if name in path:
            if verbose:
                print(f"Skipped '{path}'")
            return False
    return True


# ---------------------------------------------------------------------------
# Contact / floor helpers
# ---------------------------------------------------------------------------

def _joint_velocity(seq: np.ndarray) -> np.ndarray:
    """Frame-wise L2 velocity; last frame duplicates the previous delta."""
    vel = np.linalg.norm(seq[1:] - seq[:-1], axis=1)
    return np.append(vel, vel[-1])


def detect_joint_contact(body_joint_seq: np.ndarray, joint_name: str,
                         floor_height: float, vel_thresh: float,
                         height_thresh: float) -> np.ndarray:
    """Binary contact mask for a single joint based on velocity + height."""
    joint_seq = body_joint_seq[:, SMPL_JOINTS[joint_name], :]
    vel = _joint_velocity(joint_seq)
    heights = joint_seq[:, 2] - floor_height
    return np.logical_and(vel < vel_thresh, heights < height_thresh)


def determine_floor_height_and_contacts(body_joint_seq: np.ndarray, fps: float):
    """
    Estimate floor height via foot-velocity clustering and compute foot contacts.

    Args:
        body_joint_seq: (N, 22, 3) array of joint positions.
        fps:            Recording frame rate.

    Returns:
        offset_floor_height (float),
        contacts            (N, 22) binary array,
        discard_seq         (bool).
    """
    num_frames = body_joint_seq.shape[0]

    # ---- toe velocities and heights ----
    left_toe_seq  = body_joint_seq[:, SMPL_JOINTS['leftToeBase'], :]
    right_toe_seq = body_joint_seq[:, SMPL_JOINTS['rightToeBase'], :]
    root_seq      = body_joint_seq[:, SMPL_JOINTS['hips'], :]

    left_toe_vel  = _joint_velocity(left_toe_seq)
    right_toe_vel = _joint_velocity(right_toe_seq)

    left_toe_heights  = left_toe_seq[:, 2]
    right_toe_heights = right_toe_seq[:, 2]
    root_heights      = root_seq[:, 2]

    # ---- cluster static foot heights to find the floor ----
    all_inds = np.arange(num_frames)
    left_static_mask  = left_toe_vel < FLOOR_VEL_THRESH
    right_static_mask = right_toe_vel < FLOOR_VEL_THRESH

    all_static_heights = np.concatenate([
        left_toe_heights[left_static_mask],
        right_toe_heights[right_static_mask],
    ])
    all_static_inds = np.concatenate([
        all_inds[left_static_mask],
        all_inds[right_static_mask],
    ])

    discard_seq = False
    if all_static_heights.size > 0:
        clustering = DBSCAN(eps=0.005, min_samples=3).fit(all_static_heights.reshape(-1, 1))
        labels = clustering.labels_

        min_median = min_root_median = float('inf')
        cluster_info = []  # (height_median, root_median, size)

        for label in np.unique(labels):
            mask = labels == label
            clust_heights = all_static_heights[mask]
            clust_inds    = np.unique(all_static_inds[mask])

            h_median    = np.median(clust_heights)
            root_median = np.median(root_heights[clust_inds])
            size        = clust_heights.size

            cluster_info.append((h_median, root_median, size))

            if h_median < min_median:
                min_median     = h_median
                min_root_median = root_median

        floor_height        = min_median
        offset_floor_height = floor_height - FLOOR_HEIGHT_OFFSET

        if DISCARD_TERRAIN_SEQUENCES:
            for h_med, root_med, size in cluster_info:
                if (
                    root_med > min_root_median + ROOT_HEIGHT_THRESH
                    and h_med  > min_median     + TERRAIN_HEIGHT_THRESH
                    and size   > int(CLUSTER_SIZE_THRESH * fps)
                ):
                    discard_seq = True
                    print('DISCARDING sequence based on terrain interaction!')
                    break
    else:
        floor_height = offset_floor_height = 0.0

    # ---- heel velocities and heights ----
    left_heel_seq  = body_joint_seq[:, SMPL_JOINTS['leftFoot'], :]
    right_heel_seq = body_joint_seq[:, SMPL_JOINTS['rightFoot'], :]
    left_heel_vel  = _joint_velocity(left_heel_seq)
    right_heel_vel = _joint_velocity(right_heel_seq)

    left_heel_heights  = left_heel_seq[:, 2]  - floor_height
    right_heel_heights = right_heel_seq[:, 2] - floor_height
    left_toe_heights   = left_toe_heights      - floor_height
    right_toe_heights  = right_toe_heights     - floor_height

    # ---- contact masks (velocity + height) ----
    contacts = np.zeros((num_frames, len(SMPL_JOINTS)))
    contacts[:, SMPL_JOINTS['leftFoot']]     = np.logical_and(left_heel_vel  < CONTACT_VEL_THRESH, left_heel_heights  < CONTACT_ANKLE_HEIGHT_THRESH)
    contacts[:, SMPL_JOINTS['rightFoot']]    = np.logical_and(right_heel_vel < CONTACT_VEL_THRESH, right_heel_heights < CONTACT_ANKLE_HEIGHT_THRESH)
    contacts[:, SMPL_JOINTS['leftToeBase']]  = np.logical_and(left_toe_vel   < CONTACT_VEL_THRESH, left_toe_heights   < CONTACT_TOE_HEIGHT_THRESH)
    contacts[:, SMPL_JOINTS['rightToeBase']] = np.logical_and(right_toe_vel  < CONTACT_VEL_THRESH, right_toe_heights  < CONTACT_TOE_HEIGHT_THRESH)

    for joint_name, col in [('leftHand', 'leftHand'), ('rightHand', 'rightHand'),
                             ('leftLeg',  'leftLeg'),  ('rightLeg',  'rightLeg')]:
        contacts[:, SMPL_JOINTS[col]] = detect_joint_contact(
            body_joint_seq, joint_name, floor_height,
            CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH,
        )

    return offset_floor_height, contacts, discard_seq


# ---------------------------------------------------------------------------
# IMU synthesis
# ---------------------------------------------------------------------------

def syn_acc(v: torch.Tensor, smooth_n: int = 4) -> torch.Tensor:
    """Synthesise accelerations from vertex positions (simple finite differences)."""
    mid = smooth_n // 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600
                       for i in range(v.shape[0] - 2)])
    acc = torch.cat([torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])])
    if mid != 0 and v.shape[0] >= 8:
        acc[smooth_n:-smooth_n] = torch.stack([
            (v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
            for i in range(v.shape[0] - smooth_n * 2)
        ])
    return acc


# ---------------------------------------------------------------------------
# SMPL / SMPLX body-parameter extraction
# ---------------------------------------------------------------------------

_JREG_PATH = os.path.join('data', 'J_regressor_coco.npy')
_IMU_JOINT_MASK   = [18, 19, 4, 5, 15, 0]
_IMU_VERTEX_MASK  = [1961, 5424, 1176, 4662, 411, 3021]


def _load_jregressor() -> torch.Tensor:
    return torch.tensor(np.load(_JREG_PATH), dtype=torch.float32)


def _extract_smpl(bdata, body_models: dict, stride: int, device: str):
    """Extract body parameters and run forward pass for SMPL topology."""
    body_model = body_models['male']  # gender-agnostic for now
    poses = bdata['poses'][::stride]
    trans = bdata['trans'][::stride]
    gender = bdata['gender']
    
    body_parms = {
        'root_orient': torch.tensor(poses[:, :3],   dtype=torch.float32, device=device),
        'pose_body':   torch.tensor(poses[:, 3:66], dtype=torch.float32, device=device),
        'trans':       torch.tensor(trans,          dtype=torch.float32, device=device),
    }
    with torch.no_grad():
        output = body_model(**body_parms)

    return poses, output.v.cpu(), output.Jtr.cpu(), body_parms, gender


def _extract_smplx(bdata, body_models: dict, stride: int, device: str):
    """Extract body parameters and run forward pass for SMPLX topology."""
    body_model = body_models['neutral'].to(device)
    trans        = torch.tensor(bdata['trans'],       dtype=torch.float32)[::stride].to(device)
    root_aa      = torch.tensor(bdata['root_orient'][::stride], dtype=torch.float32)
    body_aa      = torch.tensor(bdata['pose_body'][::stride],   dtype=torch.float32)

    global_orient = utils_transform.angle_axis_to_matrix(root_aa).to(device)
    body_pose     = utils_transform.angle_axis_to_matrix(
        body_aa.reshape(trans.shape[0], -1, 3)
    ).to(device)

    body_parms = {
        'global_orient': global_orient,
        'pose_body':     body_pose,
        'trans':         trans,
    }
    with torch.no_grad():
        output = body_model(global_orient=global_orient, body_pose=body_pose, transl=trans)

    # Re-pack poses as a single tensor to mirror the SMPL branch
    poses = torch.cat([root_aa, body_aa], dim=-1).numpy()
    return poses, output.vertices.cpu(), output.joints.cpu(), body_parms, 'neutral'


def _compute_rotations(poses, topology: str, body_model, global_orient=None, body_pose=None):
    """Return (rotation_local_6d, rotation_global_matrot) for either topology."""
    if topology == 'smpl':
        full_rot_aa = torch.tensor(poses[:, :66])
    else:
        full_rot_aa = torch.tensor(poses)

    output_6d = _utils_transform.aa2sixd(full_rot_aa.reshape(-1, 3))
    rotation_local_6d = output_6d.reshape(poses.shape[0], -1)[1:]

    rotation_local_matrot = aa2matrot(torch.tensor(poses).reshape(-1, 3)).reshape(
        poses.shape[0], -1, 9
    )
    if topology == 'smpl':
        rotation_global_matrot = local2global_pose(
            rotation_local_matrot, body_model.kintree_table[0].long()
        )
    else:
        rotation_global_matrot = utils_transform.rotational_fk(global_orient, body_pose)

    return rotation_local_6d, rotation_global_matrot


# ---------------------------------------------------------------------------
# Main processing entry point
# ---------------------------------------------------------------------------

def process(src: str, dst: str, body_models: dict, topology: str,
            logging, camera_path: str, MV, split_file=None, **kwargs):
    """
    Process motion-capture files from *src* and write per-sequence .pkl files to *dst*.

    Args:
        src:         Root directory of the raw dataset.
        dst:         Output directory for processed .pkl files.
        body_models: Dict mapping gender/topology keys to BodyModel instances.
        topology:    'smpl' or 'smplx'.
        logging:     Python logging module (or compatible).
        camera_path: Path to camera XML directory.
        MV:          Initialised MeshViewer2 instance.
        split_file:  Optional path to a text file listing relative file paths.
        **kwargs:    Forwarded to run_yolo (e.g. yolo_model, yolo_model_str, topology).
    """
    assert src and dst

    # ---- build file list ----
    if split_file is None:
        all_files = sorted(glob.glob(os.path.join(src, '**', '*.npz'), recursive=True))
    else:
        with open(split_file, 'r') as f:
            prefix = '/'.join(src.split('/')[:-1])
            all_files = [f'{prefix}/{line.rstrip()}' for line in f]
        if topology == 'smplx':
            all_files = list(filter(exclusion_func, map(old_to_new_func, all_files)))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    jregressor = _load_jregressor()
    yolo_model     = kwargs.get('yolo_model')
    yolo_model_str = kwargs.get('yolo_model_str')

    mocap_rate_key = 'mocap_framerate' if topology == 'smpl' else 'mocap_frame_rate'

    idx = 0
    for filepath in sorted(all_files):

        # ---- skip already-processed files ----
        idx += 1
        out_path = os.path.join(dst, f'{idx}.pkl')
        if os.path.exists(out_path):
            logging.info(f'File {out_path} exists – skipping.')
            continue

        bdata = np.load(filepath, allow_pickle=True)

        try:
            framerate = float(bdata[mocap_rate_key])
        except KeyError:
            logging.info(f'Missing frame-rate key in {filepath} – keys: {list(bdata.keys())}')
            continue

        stride = round(framerate / 60)

        # ---- forward pass ----
        if topology == 'smpl':
            poses, vertices, joints, body_parms, gender = _extract_smpl(
                bdata, body_models, stride, device
            )
            rotation_global_matrot_args = dict(body_model=body_models['male'])
        else:
            poses, vertices, joints, body_parms, gender = _extract_smplx(
                bdata, body_models, stride, device
            )
            rotation_global_matrot_args = dict(
                body_model=None,
                global_orient=body_parms['global_orient'],
                body_pose=body_parms['pose_body'],
            )

        if vertices.shape[0] <= 10:
            continue  # skip very short sequences

        # ---- rotations ----
        rotation_local_6d, rotation_global_matrot = _compute_rotations(
            poses, topology,
            body_model=rotation_global_matrot_args.get('body_model'),
            global_orient=rotation_global_matrot_args.get('global_orient'),
            body_pose=rotation_global_matrot_args.get('body_pose'),
        )

        # ---- IMU synthesis ----
        out_grot = rotation_global_matrot[:, _IMU_JOINT_MASK]
        out_gacc = syn_acc(vertices[:, _IMU_VERTEX_MASK])

        # ---- global 6-D rotations and velocities ----
        head_rot_global = rotation_global_matrot[:, [15], :, :]
        rotation_global_6d = _utils_transform.matrot2sixd(
            rotation_global_matrot.reshape(-1, 3, 3)
        ).reshape(*rotation_global_matrot.shape[:2], 6)

        rotation_vel_matrot = torch.matmul(
            torch.inverse(rotation_global_matrot[:-1]),
            rotation_global_matrot[1:],
        )
        rotation_vel_6d = _utils_transform.matrot2sixd(
            rotation_vel_matrot.reshape(-1, 3, 3)
        ).reshape(*rotation_vel_matrot.shape[:2], 6)

        # ---- joint positions ----
        position_global = joints[:, :22, :]
        position_head   = position_global[:, 15, :]

        head_global_trans = torch.eye(4).repeat(position_head.shape[0], 1, 1)
        head_global_trans[:, :3, :3] = head_rot_global.squeeze()
        head_global_trans[:, :3,  3] = position_global[:, 15, :]

        num_frames = position_global.shape[0] - 1

        hmd_features = torch.cat([
            rotation_global_6d[1:, :22].reshape(num_frames, -1),
            rotation_vel_6d[:, :22].reshape(num_frames, -1),
            position_global[1:, :22].reshape(num_frames, -1),
            (position_global[1:, :22] - position_global[:-1, :22]).reshape(num_frames, -1),
        ], dim=-1)

        # ---- COCO joint projection ----
        joints_coco = torch.einsum('bik,ji->bjk', vertices, jregressor)

        # ---- assemble output dict ----
        out = {
            'rotation_local_full_gt_list':      rotation_local_6d.cpu(),
            'hmd_position_global_full_gt_list': hmd_features.cpu(),
            'body_parms_list':                  {k: v[1:].cpu() for k, v in body_parms.items()},
            'head_global_trans_list':           head_global_trans[1:].cpu(),
            'framerate':                        60,
            'gender':                           gender,
            'filepath':                         filepath,
            'IMU_global_rotation':              out_grot.cpu()[1:],
            'IMU_global_acceleration':          out_gacc.cpu()[1:],
            'shape':                            bdata['betas'],
            'pose_estimation_keypoints':        {},
        }

        # ---- optional YOLO pose estimation ----
        if yolo_model is not None:
            vcam_kp, confidences = run_yolo(
                mv=MV,
                bm=body_models.get('male') or body_models.get('neutral'),
                body_pose_world=_get_body_pose_world(bdata, body_models, topology, stride, device),
                nb_frames=num_frames,
                orig_file=filepath,
                frame_path=dst,
                idx=idx,
                camera_path=camera_path,
                **kwargs,
            )
            kp_dict = {
                'model_version': yolo_model_str,
                'confidences':   torch.tensor(confidences[1:]),
                'ground_truth':  joints_coco[1:],
            }
            for cam_idx, cam_data in enumerate(vcam_kp):
                kp_dict[f'vcam{cam_idx}'] = torch.tensor(cam_data[1:])
            out['pose_estimation_keypoints'] = kp_dict

        # ---- save ----
        logging.info(f'Saving {out_path}')
        with open(out_path, 'wb') as f:
            pickle.dump(out, f)

        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _get_body_pose_world(bdata, body_models, topology, stride, device):
    """Re-run the forward pass to obtain the body-pose mesh object for rendering."""
    if topology == 'smpl':
        poses = bdata['poses'][::stride]
        parms = {
            'root_orient': torch.tensor(poses[:, :3],               dtype=torch.float32, device=device),
            'pose_body':   torch.tensor(poses[:, 3:66],             dtype=torch.float32, device=device),
            'trans':       torch.tensor(bdata['trans'][::stride],   dtype=torch.float32, device=device),
        }
        with torch.no_grad():
            return body_models['male'](**parms)
    else:
        body_model = body_models['neutral'].to(device)
        trans      = torch.tensor(bdata['trans'][::stride],         dtype=torch.float32, device=device)
        root_aa    = torch.tensor(bdata['root_orient'][::stride],   dtype=torch.float32, device=device)
        body_aa    = torch.tensor(bdata['pose_body'][::stride],     dtype=torch.float32, device=device)
        global_orient = utils_transform.angle_axis_to_matrix(root_aa).to(device)
        body_pose     = utils_transform.angle_axis_to_matrix(
            body_aa.reshape(trans.shape[0], -1, 3)
        ).to(device)
        with torch.no_grad():
            return body_model(global_orient=global_orient, body_pose=body_pose, transl=trans)
