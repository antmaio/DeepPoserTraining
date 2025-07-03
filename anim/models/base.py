"""
Inspired from https://github.com/georgedf1/sfbpe/tree/main
"""

# External
import typing
import torch
import abc
from torch.utils.data import Dataset
from dataclasses import dataclass

# Internal
#import utils
import utils.utils_transform as transform
from anim.data.amass import SmplxJoints
import _debug as DEBUG 
# You can raise this in your model to interrupt training in train.py
#   Specifically, raise in forward_pass only when optimise=True
class StopTrainingException(Exception):
    pass

@dataclass
class BaseModelInput:
    batch_size: int
    win_len: int
    head_pos_global: torch.Tensor  # (batch_size, win_len, 3)
    lh_pos_global: torch.Tensor  # ditto
    rh_pos_global: torch.Tensor  # ditto
    head_rot_global: torch.Tensor  # (batch_size, win_len, 3, 3)
    lh_rot_global: torch.Tensor  # ditto
    rh_rot_global: torch.Tensor  # ditto
    # TODO Weaken the below assumption that we only use HMR to allow for MediaPipe compatibility
    # hmr_presence: torch.BoolTensor  # TODO Utilise for sparsity
    hmr_joints: torch.Tensor  # (batch_size, win_len, SmplxJoints.NUM_JTS, 3) or (batch_size, win_len, YoloJoints.NUM_JTS, 3) 
    hmr_body_pose: torch.Tensor  # (batch_size, win_len, SmplxJoints.NUM_JTS - 1, 3, 3)
    hmr_global_orient: torch.Tensor  # (batch_size, win_len, 3, 3)
    #Optional
    betas: typing.Optional[torch.Tensor]#(batch_size, win_len, num_betas)
    gender: typing.Optional[int] #(batch_size,)
    conf: typing.Optional[torch.Tensor] #(batch_size, win_len, YoloJoints.NUM_JTS, 2)



@dataclass
class BaseModelOutput:
    # betas for SMPLX model, can be omitted to signal different model (e.g. aligner)
    betas: typing.Optional[torch.Tensor]
    # TODO Do we make other params optional?
    #transl: torch.Tensor  # (batch_size, win_len, 3)
    global_orient: torch.Tensor  # (batch_size, win_len, 3, 3)
    body_pose: torch.Tensor  # (batch_size, win_len, SmplxJoints.NUM_JTS-1, 3, 3)
    # Redundant but packing in here anyway for convenience:
    joints: torch.Tensor   # (batch_size, win_len, SmplxJoints.NUM_JTS, 3)
    #vertices: torch.Tensor
    #faces: list[int]  # for rendering; same set of indices for all vertices
    gender: typing.Optional[int] #(batch_size,)


class BaseModel(torch.nn.Module):
#class BaseModel(torch.jit.ScriptModule):
    """ Base class for synthetic_models """

    @staticmethod
    @abc.abstractmethod
    def model_str() -> str:
        """ Return unique string representing model class; cannot contain underscores """
        pass

    @abc.abstractmethod
    def dataset_pass(self, dataset: Dataset, device, dtype):
        """
        Compute dataset dependent parameters prior to training (e.g. normalisation).
        Only called for fresh models, so copy into torch parameters made in __init__ for them to save!
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """
        Reset any temporal state held by the model.

        Specifically, 'forward' and 'forward_pass' must assume consecutive calls provide consecutive frames.
        Thus, a call to reset signals to the model that new, unrelated, windows of data are inbound.

        For example, we might reset accumulated RNN hidden state.

        To clarify, consider a batch of windows B, containing data with shapes (batch_size, win_len, ...), and that
          we split these windows in consecutive halves B1 and B2, each with data shapes (batch_size, win_len/2, ...).
        Then, calling 'forward(B1)' and then 'forward(B2)', with no intermediate 'reset', is equivalent to 'forward(B)'.
        On the other hand, an intermediate 'reset' here would signal that B1 and B2 are temporally unrelated.
        """
        pass

    @abc.abstractmethod
    def forward(self, model_input: BaseModelInput) -> BaseModelOutput:
        """
        Predict for one batch of input data (BaseModelInput).

        As described by the 'reset' docstring, consecutive calls of 'forward' should be considered
          temporally consecutive, so you should maintain model temporal state (e.g. RNN history).

        Returns prediction (BaseModelInput).
        """
        pass

    # TODO This might be doing too many things? what about a train_pass that accepts model_output, model_target?
    @abc.abstractmethod
    def forward_pass(self, model_input: BaseModelInput, model_target: BaseModelOutput, optimise: bool = False) -> dict:
        """
        Predict on one batch of input data (i.e. forward), compute losses, and update parameters if optimise true.

        This method would typically call 'forward' internally is expected to have the same 'reset' semantics.

        Returns dict with model losses on that batch.
        """
        pass

    @abc.abstractmethod
    def epoch_end(self, epoch: int, train_losses: dict, val_losses: dict):
        """
        Called by training loop after an epoch of training with 'epoch' index passed.
        Train losses and val losses are averaged numpy losses with same keys as returned 'forward_pass'.
        Useful for updating per-epoch state, such as LR schedulers.
        """
        pass


def batch_to_model_input_and_target(
        batch: dict, device, dtype, mode3d:str='gt') -> (BaseModelInput, BaseModelOutput):
    
    '''
    return {
        'rotations_local_full_gt_list' : self._rotations_local_full_gt_list[idx].clone(),
        'hmd_position_global_full_gt_list': self._hmd_position_global_full_gt_list[idx].clone(),
        'head_global_trans_list': self._head_global_trans_list[idx].clone(),
        'betas': betas, 
        'gender': self._gender[idx].clone(),
        'framerate' : self._framerate[idx].clone(),
        'filepath': self._filepath[idx].clone(),
        'body_parms_list': self._body_parms_list[idx].clone()
    }
    '''

    betas = batch['betas'].to(device, dtype)
    rotations_local_full_gt_list = batch['rotations_local_full_gt_list'].to(device, dtype)
    hmd_position_global_full_gt_list = batch['hmd_position_global_full_gt_list'].to(device, dtype)
    gender = batch['gender'].to(device, dtype)
    #data from external pose estimation
    keypoints = batch['keypoints'].to(device, dtype)
    conf = batch['conf'].to(device, dtype)

    batch_size, win_len, *_ = betas.shape

    rotation = hmd_position_global_full_gt_list[:,:,:SmplxJoints.NUM_JTS*6].reshape(batch_size, win_len, SmplxJoints.NUM_JTS, 6)
    body_pose = transform.two_axis_to_matrix(rotations_local_full_gt_list.reshape(batch_size, win_len, SmplxJoints.NUM_JTS, 6))
    position =  hmd_position_global_full_gt_list[:,:,SmplxJoints.NUM_JTS*6*2:SmplxJoints.NUM_JTS*6*2+3*SmplxJoints.NUM_JTS].reshape(batch_size, win_len, SmplxJoints.NUM_JTS, 3)

    #TODO REMOVE
    #DEBUG.scatter_plot_3d_pose(position, b=0,f=0) #ok

    # Mode3d indicates what 3D positions are used to guide body tracking
    if mode3d is not None:
        if mode3d == 'external': 
            hmr_position = keypoints.clone()
        elif mode3d == 'gt':
            hmr_position = position.clone()
        else:
            raise NotImplementedError(f"{mode3d} is not available for mode3d")
    else:
        hmr_position = position.clone() #but not used

    model_input = BaseModelInput(
        batch_size          = batch_size,
        win_len             = win_len, 
        head_pos_global     = position[:,:,SmplxJoints.HEAD],
        head_rot_global     = transform.two_axis_to_matrix(rotation[:,:,SmplxJoints.HEAD]),
        lh_pos_global       = position[:,:,SmplxJoints.LEFT_WRIST],
        lh_rot_global       = transform.two_axis_to_matrix(rotation[:,:,SmplxJoints.LEFT_WRIST]),
        rh_pos_global       = position[:,:,SmplxJoints.RIGHT_WRIST],           
        rh_rot_global       = transform.two_axis_to_matrix(rotation[:,:,SmplxJoints.RIGHT_WRIST]),
        # Emulate HMR using ground truth data (thus synthetic); omitting betas due to incompatibility across models
        hmr_joints          = hmr_position,
        hmr_body_pose       = body_pose[:,:,1:],
        hmr_global_orient   = body_pose[:,:,0],
        betas               = betas,
        gender              = gender,
        conf                = conf
    )

    model_target = BaseModelOutput(
        betas               = betas,
        global_orient       = transform.two_axis_to_matrix(rotation[:,:,SmplxJoints.PELVIS]),
        body_pose           = body_pose[:,:,1:],
        joints              = position,
        gender              = gender
    )

    # --- Dataset API (see amass.AMASSNeutralDataset and egobody.EgoBodyDataset for implementations) ---
    """
    batch_size, win_len, *_ = betas.shape

    joints = None
    vertices = None
    for layer_idx, smplx_layer in enumerate(smplx_layers):
        idxs = smplx_layer_idx == layer_idx
        layer_batch_size = int(torch.count_nonzero(idxs))
        if layer_batch_size == 0:
            continue
        squashed_size = layer_batch_size * win_len
        smplx_output = smplx_layers[layer_idx](
            betas=betas[idxs].reshape(squashed_size, -1),
            transl=transl[idxs].reshape(squashed_size, 3),
            global_orient=global_orient[idxs].reshape(squashed_size, 3, 3),
            body_pose=body_pose[idxs].reshape(squashed_size, SmplxJoints.NUM_JTS - 1, 3, 3)
        )
        layer_joints = smplx_output.joints[:, :SmplxJoints.NUM_JTS].view(
            layer_batch_size, win_len, SmplxJoints.NUM_JTS, 3)
        layer_vertices = smplx_output.vertices.view(layer_batch_size, win_len, -1, 3)

        if joints is None:
            joints = torch.empty(batch_size, win_len, SmplxJoints.NUM_JTS, 3, device=device, dtype=dtype)
        joints[idxs] = layer_joints
        if vertices is None:
            num_vertices = smplx_output.vertices.shape[1]
            vertices = torch.empty(batch_size, win_len, num_vertices, 3, device=device, dtype=dtype)
        vertices[idxs] = layer_vertices
    assert joints is not None
    assert vertices is not None

    faces = smplx_layers[0].faces

    body_pose_global = utils.rotational_fk(global_orient, body_pose)

    model_input = BaseModelInput(
        batch_size=batch_size,
        win_len=win_len,
        head_pos_global=joints[:, :, SmplxJoints.HEAD],
        head_rot_global=body_pose_global[:, :, SmplxJoints.HEAD - 1],
        lh_pos_global=joints[:, :, SmplxJoints.LEFT_WRIST],
        lh_rot_global=body_pose_global[:, :, SmplxJoints.LEFT_WRIST - 1],
        rh_pos_global=joints[:, :, SmplxJoints.RIGHT_WRIST],
        rh_rot_global=body_pose_global[:, :, SmplxJoints.RIGHT_WRIST - 1],
        # Emulate HMR using ground truth data (thus synthetic); omitting betas due to incompatibility across models
        hmr_body_pose=body_pose,
        hmr_joints=joints,
        hmr_global_orient=global_orient,
        hmr_vertices=vertices,
        hmr_faces=faces,
    )
    model_target = BaseModelOutput(
        betas=betas,
        transl=transl,
        global_orient=global_orient,
        body_pose=body_pose,
        joints=joints,
        vertices=vertices,
        faces=faces,
    )
    """
    return model_input, model_target
