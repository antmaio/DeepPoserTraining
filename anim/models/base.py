"""
Inspired from https://github.com/georgedf1/sfbpe/tree/main
"""
# External
import abc
import typing

import torch
from dataclasses import dataclass
from torch.utils.data import Dataset

# Internal
import utils.utils_transform as transform
from anim.data.amass import SmplxJoints


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class StopTrainingException(Exception):
    """Raise inside forward_pass (when optimise=True) to interrupt training."""
    pass


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BaseModelInput:
    batch_size:       int
    win_len:          int
    head_pos_global:  torch.Tensor   # (B, T, 3)
    lh_pos_global:    torch.Tensor   # (B, T, 3)
    rh_pos_global:    torch.Tensor   # (B, T, 3)
    head_rot_global:  torch.Tensor   # (B, T, 3, 3)
    lh_rot_global:    torch.Tensor   # (B, T, 3, 3)
    rh_rot_global:    torch.Tensor   # (B, T, 3, 3)
    hmr_joints:       torch.Tensor   # (B, T, NUM_JTS, 3)
    hmr_body_pose:    torch.Tensor   # (B, T, NUM_JTS-1, 3, 3)
    hmr_global_orient: torch.Tensor  # (B, T, 3, 3)
    # Optional
    betas:   typing.Optional[torch.Tensor]  # (B, T, num_betas)
    gender:  typing.Optional[int]
    conf:    typing.Optional[torch.Tensor]  # (B, T, NUM_JTS, 1)


@dataclass
class BaseModelOutput:
    betas:         typing.Optional[torch.Tensor]  # (B, T, num_betas)
    global_orient: torch.Tensor                   # (B, T, 3, 3)
    body_pose:     torch.Tensor                   # (B, T, NUM_JTS-1, 3, 3)
    joints:        typing.Optional[torch.Tensor]  # (B, T, NUM_JTS, 3)
    gender:        typing.Optional[int]
    transl:        typing.Optional[torch.Tensor]  # (B, T, 3)


# ---------------------------------------------------------------------------
# Abstract base model
# ---------------------------------------------------------------------------

class BaseModel(torch.nn.Module):
    """Abstract base class for all pose prediction models."""

    @staticmethod
    @abc.abstractmethod
    def model_str() -> str:
        """Unique identifier for the model class; must not contain underscores."""
        pass

    @abc.abstractmethod
    def dataset_pass(self, dataset: Dataset, device, dtype):
        """
        Compute dataset-dependent parameters before training (e.g. normalisation).
        Called only for freshly created models; store results as registered buffers
        or parameters so they are included in state_dict().
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """
        Reset all temporal state (e.g. RNN hidden states).

        Consecutive forward() calls are assumed to provide temporally consecutive
        frames.  Calling reset() signals that the next batch is unrelated to the
        previous one.
        """
        pass

    @abc.abstractmethod
    def forward(self, model_input: BaseModelInput) -> BaseModelOutput:
        """Predict for one batch; maintains temporal state across calls."""
        pass

    @abc.abstractmethod
    def forward_pass(
        self,
        model_input: BaseModelInput,
        model_target: BaseModelOutput,
        optimise: bool = False,
    ) -> dict:
        """
        Forward pass + loss computation + optional parameter update.

        Returns a dict of named scalar losses (must include 'loss').
        """
        pass

    @abc.abstractmethod
    def epoch_end(self, epoch: int, train_losses: dict, val_losses: dict):
        """Called by the training loop at the end of every epoch."""
        pass


# ---------------------------------------------------------------------------
# Batch conversion utility
# ---------------------------------------------------------------------------

def batch_to_model_input_and_target(
    batch: dict,
    device,
    dtype,
    mode3d: str = 'gt',
) -> tuple[BaseModelInput, BaseModelOutput]:
    """
    Convert a raw data-loader batch dict into (BaseModelInput, BaseModelOutput).

    Args:
        batch:   Dict from the DataLoader.
        device:  Target torch device.
        dtype:   Target torch dtype.
        mode3d:  Which 3-D source to use for hmr_joints:
                 'gt'       – ground-truth SMPLX joints,
                 'external' – external pose estimator keypoints,
                 None       – ground truth (not used by model).
    """
    def _to(t):
        return t.to(device, dtype)

    betas                            = _to(batch['betas'])
    rotations_local_full_gt          = _to(batch['rotations_local_full_gt_list'])
    hmd_position_global_full_gt      = _to(batch['hmd_position_global_full_gt_list'])
    gender                           = _to(batch['gender'])
    keypoints                        = _to(batch['keypoints'])

    conf   = batch.get('conf')
    transl = batch.get('transl')
    if isinstance(conf,   torch.Tensor): conf   = conf.to(device=device, dtype=dtype)
    if isinstance(transl, torch.Tensor): transl = transl.to(device=device, dtype=dtype)

    batch_size, win_len = betas.shape[:2]
    N = SmplxJoints.NUM_JTS

    # ---- decode rotations / positions from packed representation ----
    rotation = hmd_position_global_full_gt[:, :, :N * 6].reshape(batch_size, win_len, N, 6)
    position = hmd_position_global_full_gt[
        :, :, N * 6 * 2 : N * 6 * 2 + N * 3
    ].reshape(batch_size, win_len, N, 3)

    body_pose = transform.two_axis_to_matrix(
        rotations_local_full_gt.reshape(batch_size, win_len, N, 6)
    )

    if mode3d == 'external':
        hmr_position = keypoints.clone()
    elif mode3d in ('gt', None):
        hmr_position = position.clone()
    else:
        raise NotImplementedError(f"mode3d='{mode3d}' is not supported.")

    def _rot(joint_idx: int) -> torch.Tensor:
        return transform.two_axis_to_matrix(rotation[:, :, joint_idx])

    model_input = BaseModelInput(
        batch_size        = batch_size,
        win_len           = win_len,
        head_pos_global   = position[:, :, SmplxJoints.HEAD],
        head_rot_global   = _rot(SmplxJoints.HEAD),
        lh_pos_global     = position[:, :, SmplxJoints.LEFT_WRIST],
        lh_rot_global     = _rot(SmplxJoints.LEFT_WRIST),
        rh_pos_global     = position[:, :, SmplxJoints.RIGHT_WRIST],
        rh_rot_global     = _rot(SmplxJoints.RIGHT_WRIST),
        hmr_joints        = hmr_position,
        hmr_body_pose     = body_pose[:, :, 1:],
        hmr_global_orient = body_pose[:, :, 0],
        betas             = betas,
        gender            = gender,
        conf              = conf,
    )

    model_target = BaseModelOutput(
        betas         = betas,
        transl        = transl,
        global_orient = _rot(SmplxJoints.PELVIS),
        body_pose     = body_pose[:, :, 1:],
        joints        = position,
        gender        = gender,
    )

    return model_input, model_target
