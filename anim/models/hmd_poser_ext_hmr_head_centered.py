"""
Inspired from https://github.com/georgedf1/sfbpe/tree/main
"""
# External
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Internal
from . import base, BaseModelInput, BaseModelOutput
from .architecture import (
    build_hmd_embedding,
    build_joint_embedding,
    build_temporal_encoder,
    build_spatial_encoder,
    build_pose_head,
    build_shape_head,
)
from .losses import LossWeights, compute_losses
from anim.data.amass import SmplxJoints, YoloJoints, FPS
from utils.utils_transform import (
    two_axis_to_matrix,
    matrix_to_two_axis,
    matrix_to_angle_axis,
)
from utils.utils_ukf import UKF, filter_poses, TransitionStateFunctions
from human_body_prior.body_model.body_model import BodyModel
import data.data_config as bm_C

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)


class HMDPoserExtHeadCentered(nn.Module):
    """Extension of HMD-Poser that exploits external pose-estimator joints / body pose."""

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @staticmethod
    def model_str() -> str:
        return 'hmd-poser-ext-head-centered'

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        # Input selection
        chosen_jts: list[int]               = None,
        mode3d: str                         = None,
        use_hmr_body_pose: bool             = False,
        use_hmr_velocities: bool            = False,
        use_rnn_layer_norm: bool            = False,
        use_conf_score: bool                = False,
        # Body model
        num_betas: int                      = bm_C._NUM_BETAS_,
        num_dmpls: int                      = bm_C._NUM_DMPLS_,
        # Architecture
        hidden_size: int                    = 256,
        num_blocks: int                     = 2,
        rnn_type: str                       = 'lstm',
        num_rnn_layers: int                 = 1,
        num_transformer_heads: int          = 8,
        num_transformer_layers: int         = 3,
        # Loss
        loss_func: str                      = 'l1',
        global_orient_loss_weight: float    = 1.0,
        body_pose_loss_weight: float        = 5.0,
        body_pose_global_loss_weight: float = 1.0,
        joints_loss_weight: float           = 1.0,
        smooth_loss_weight: float           = 0.5,
        shape_loss_weight: float            = 0.1,
        extra_shape_loss_weight: float      = 0.0,
        orth_loss: float                    = 0.1,
        jitter_loss: float                  = 0.0,
        extra_hand_pose_loss_weight: float          = 0.0,
        extra_hand_pose_global_loss_weight: float   = 0.0,
        extra_hand_joints_loss_weight: float        = 0.0,
        orthonormality_loss_weight: float           = 1.0,
        # Optimiser
        lr: float                           = 1e-3
    ):
        super().__init__()
        self._validate_args(hidden_size, num_blocks, rnn_type, num_rnn_layers,
                            num_transformer_heads, num_transformer_layers,
                            loss_func, lr, num_betas)

        # ---- topology & chosen joints ----
        self.topology = SmplxJoints if mode3d == 'gt' else YoloJoints
        if mode3d not in ('gt', 'external', None):
            raise NotImplementedError(f"mode3d='{mode3d}' not recognised.")
        self.mode3d = mode3d

        chosen_jts = self._resolve_chosen_joints(chosen_jts, self.topology)
        logging.info(f"Training with {len(chosen_jts)} additional body joints.")

        # ---- body models ----
        self.num_betas = num_betas
        self.num_dmpls = num_dmpls
        self.bm_male   = BodyModel(bm_fname=bm_C._BM_FNAME_MALE_,
                                   num_betas=num_betas, num_dmpls=num_dmpls,
                                   dmpl_fname=bm_C._DMPL_FNAME_MALE_)
        self.bm_female = BodyModel(bm_fname=bm_C._BM_FNAME_MALE_,
                                   num_betas=num_betas, num_dmpls=num_dmpls,
                                   dmpl_fname=bm_C._DMPL_FNAME_FEMALE_)

        # ---- joint selection ----
        self.chosen_jts_local   = torch.tensor(chosen_jts, dtype=torch.int32) - 1
        self.num_chosen_jts     = len(chosen_jts)
        self.use_hmr_body_pose  = use_hmr_body_pose
        self.use_hmr_velocities = use_hmr_velocities

        # ---- embeddings ----
        num_hmd_channels     = 5  # head, lh, rh, lh_local, rh_local
        self.num_hmd_channels = num_hmd_channels
        self.hmd_embers = nn.ModuleList([
            build_hmd_embedding(hidden_size) for _ in range(num_hmd_channels)
        ])
        self.joint_embers = nn.ModuleList([
            build_joint_embedding(hidden_size, use_hmr_velocities, use_hmr_body_pose)
            for _ in range(self.num_chosen_jts)
        ])

        # ---- optional layer norm ----
        self.use_rnn_layer_norm = use_rnn_layer_norm
        if use_rnn_layer_norm:
            self.rnn_layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        # ---- temporal + spatial encoders ----
        self.num_channels = num_hmd_channels + self.num_chosen_jts
        self.num_blocks   = num_blocks
        self.temporal_encoder = build_temporal_encoder(
            hidden_size, self.num_channels, num_blocks, rnn_type, num_rnn_layers
        )
        self.spatial_encoder  = build_spatial_encoder(
            hidden_size, num_blocks, num_transformer_heads, num_transformer_layers
        )
        self.prev_rnn_states = [
            [None] * self.num_channels for _ in range(num_blocks)
        ]

        # ---- prediction heads ----
        self.pose_head  = build_pose_head(hidden_size, self.num_channels)
        self.shape_head = build_shape_head(hidden_size, self.num_channels, num_betas)

        # ---- losses ----
        self.loss_weights = LossWeights(
            loss_func=loss_func,
            global_orient_loss_weight=global_orient_loss_weight,
            body_pose_loss_weight=body_pose_loss_weight,
            body_pose_global_loss_weight=body_pose_global_loss_weight,
            joints_loss_weight=joints_loss_weight,
            smooth_loss_weight=smooth_loss_weight,
            shape_loss_weight=shape_loss_weight,
            extra_shape_loss_weight=extra_shape_loss_weight,
            orth_loss=orth_loss,
            jitter_loss_weight=jitter_loss,
            extra_hand_pose_loss_weight=extra_hand_pose_loss_weight,
            extra_hand_pose_global_loss_weight=extra_hand_pose_global_loss_weight,
            extra_hand_joints_loss_weight=extra_hand_joints_loss_weight,
            orthonormality_loss_weight=orthonormality_loss_weight,
        )

        # ---- optimiser (scheduler set later via set_scheduler) ----
        self.lr    = lr
        self.optim = torch.optim.Adam(
            (p for p in self.parameters() if p.requires_grad), lr=lr
        )
        self.nbatch      = None
        self.lr_scheduler = None

        # ---- cached predictions (used by forward_pass) ----
        self._global_orient_6d_pred: Optional[torch.Tensor] = None
        self._body_pose_6d_pred:     Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Setup helpers (called after construction / before training)
    # ------------------------------------------------------------------

    def set_nbatch(self, batch: int):
        self.nbatch = batch

    def set_scheduler(self):
        assert self.nbatch and self.nbatch > 0, \
            f"nbatch must be set before calling set_scheduler; got {self.nbatch}."
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optim, max_lr=self.lr, epochs=400, steps_per_epoch=self.nbatch
        )

    def dataset_pass(self, dataset: Dataset, device, dtype):
        pass  # no dataset-dependent normalisation needed

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def reset(self):
        self.prev_rnn_states = [
            [None] * self.num_channels for _ in range(self.num_blocks)
        ]

    def forward(self, model_input: BaseModelInput) -> BaseModelOutput:
        batch_size = model_input.batch_size
        win_len    = model_input.win_len

        # ---- head-centred coordinate frame ----
        head_rot_inv = torch.linalg.inv(model_input.head_rot_global)
        hmd_posis, hmd_rots = self._compute_hmd_signals(model_input, head_rot_inv)

        # ---- prepare HMR joint features ----
        hmr_joints_local = (
            model_input.hmr_joints[:, :, 1:YoloJoints.NUM_JTS]
            - model_input.head_pos_global.unsqueeze(2)
        )
        chosen_joints    = hmr_joints_local[:, :, self.chosen_jts_local]
        chosen_body_pose = model_input.hmr_body_pose[:, :, self.chosen_jts_local]


        # ---- embeddings ----
        embeddings = self._embed_hmd(hmd_posis, hmd_rots)
        embeddings += self._embed_joints(chosen_joints, chosen_body_pose)
        feats = torch.stack(embeddings, dim=-2)  # (B, T, C, H)

        # ---- temporal + spatial blocks ----
        feats = self._encode(feats, batch_size, win_len)

        # ---- heads ----
        flat_feats  = feats.reshape(batch_size, win_len, -1)
        pose_pred   = self.pose_head(flat_feats).reshape(batch_size, win_len, SmplxJoints.NUM_JTS, 6)
        betas_pred  = self.shape_head(flat_feats)

        self._global_orient_6d_pred = pose_pred[:, :, 0]
        self._body_pose_6d_pred     = pose_pred[:, :, 1:]

        # ---- decode to 3×3 matrices ----
        global_orient_3x3 = two_axis_to_matrix(self._global_orient_6d_pred)
        body_pose_3x3     = two_axis_to_matrix(self._body_pose_6d_pred)

        # ---- body-model forward pass (joints in local frame) ----
        joints_pred, transl_pred = self._body_model_forward(
            model_input, global_orient_3x3, body_pose_3x3, betas_pred,
            batch_size, win_len,
        )

        return BaseModelOutput(
            betas=betas_pred,
            transl=transl_pred,
            global_orient=global_orient_3x3,
            body_pose=body_pose_3x3,
            joints=joints_pred,
            gender=model_input.gender,
        )

    def forward_pass(
        self,
        model_input: BaseModelInput,
        model_target: BaseModelOutput,
        optimise: bool = False,
    ) -> dict:
        model_output = self(model_input)

        loss_dict = compute_losses(
            self.loss_weights,
            model_output,
            model_target,
            self._global_orient_6d_pred,
            self._body_pose_6d_pred,
        )

        if optimise:
            self.optim.zero_grad()
            loss_dict['loss'].backward()
            self._batch_end()

        return loss_dict

    def epoch_end(self, epoch: int, train_losses: dict, val_losses: dict):
        pass  # scheduler steps happen per-batch in _batch_end

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_args(hidden_size, num_blocks, rnn_type, num_rnn_layers,
                       num_transformer_heads, num_transformer_layers,
                       loss_func, lr, num_betas):
        assert hidden_size % 4 == 0,    f"hidden_size ({hidden_size}) must be divisible by 4."
        assert num_blocks > 0
        assert rnn_type in ('lstm', 'gru')
        assert num_rnn_layers > 0
        assert num_transformer_heads > 0
        assert num_transformer_layers > 0
        assert loss_func in ('l1', 'mse')
        assert lr > 0
        assert num_betas > 0

    @staticmethod
    def _resolve_chosen_joints(chosen_jts, topology) -> list[int]:
        if chosen_jts is not None:
            for jt in chosen_jts:
                assert 0 < jt < topology.NUM_JTS, (
                    f"Joint index {jt} out of range 1–{topology.NUM_JTS - 1}."
                )
            return chosen_jts

        # Default: SEWHKA set
        return [
            topology.LEFT_SHOULDER,  topology.RIGHT_SHOULDER,
            topology.LEFT_ELBOW,     topology.RIGHT_ELBOW,
            topology.LEFT_WRIST,     topology.RIGHT_WRIST,
            topology.LEFT_HIP,       topology.RIGHT_HIP,
            topology.LEFT_KNEE,      topology.RIGHT_KNEE,
            topology.LEFT_ANKLE,     topology.RIGHT_ANKLE,
        ]

    @staticmethod
    def _compute_hmd_signals(model_input: BaseModelInput, head_rot_inv: torch.Tensor):
        """Return (hmd_posis, hmd_rots) for the five HMD channels."""
        def _local_pos(pos):
            return (head_rot_inv @ (pos - model_input.head_pos_global)[..., None])[..., 0]

        hmd_posis = [
            model_input.head_pos_global,
            model_input.lh_pos_global,
            model_input.rh_pos_global,
            _local_pos(model_input.lh_pos_global),
            _local_pos(model_input.rh_pos_global),
        ]
        hmd_rots = [
            model_input.head_rot_global,
            model_input.lh_rot_global,
            model_input.rh_rot_global,
            head_rot_inv @ model_input.lh_rot_global,
            head_rot_inv @ model_input.rh_rot_global,
        ]
        return hmd_posis, hmd_rots

    def _embed_hmd(self, hmd_posis, hmd_rots) -> list[torch.Tensor]:
        embeddings = []
        for c in range(self.num_hmd_channels):
            pos         = hmd_posis[c]
            rot_3x3     = hmd_rots[c]
            pos_delta   = self._pos_delta(pos)
            rot_delta   = matrix_to_two_axis(self._rot_delta(rot_3x3))
            rot_6d      = matrix_to_two_axis(rot_3x3)
            feats       = [rot_6d, rot_delta, pos, pos_delta]
            emb = torch.cat([self.hmd_embers[c][i](f) for i, f in enumerate(feats)], dim=-1)
            if self.use_rnn_layer_norm:
                emb = self.rnn_layer_norm(emb)
            embeddings.append(emb)
        return embeddings

    def _embed_joints(self, chosen_joints, chosen_body_pose) -> list[torch.Tensor]:
        embeddings = []
        for c in range(self.num_chosen_jts):
            pos   = chosen_joints[:, :, c]
            feats = [pos]
            if self.use_hmr_velocities:
                feats.append(self._pos_delta(pos))
            if self.use_hmr_body_pose:
                rot_3x3 = chosen_body_pose[:, :, c]
                feats.append(matrix_to_two_axis(rot_3x3))
                if self.use_hmr_velocities:
                    feats.append(matrix_to_two_axis(self._rot_delta(rot_3x3)))
            emb = torch.cat([self.joint_embers[c][i](f) for i, f in enumerate(feats)], dim=-1)
            if self.use_rnn_layer_norm:
                emb = self.rnn_layer_norm(emb)
            embeddings.append(emb)
        return embeddings

    def _encode(self, feats: torch.Tensor, batch_size: int, win_len: int) -> torch.Tensor:
        for b in range(self.num_blocks):
            feats = feats.reshape(batch_size, win_len, self.num_channels, -1)

            temporal_out = []
            for c in range(self.num_channels):
                rnn_out, state = self.temporal_encoder[b][c](
                    feats[:, :, c, :], self.prev_rnn_states[b][c]
                )
                self.prev_rnn_states[b][c] = state
                temporal_out.append(rnn_out)

            feats = torch.stack(temporal_out, dim=-2)                          # (B, T, C, H)
            feats = self.spatial_encoder[b](
                feats.reshape(batch_size * win_len, self.num_channels, -1)
            )
        return feats

    def _body_model_forward(
        self,
        model_input: BaseModelInput,
        global_orient_3x3: torch.Tensor,
        body_pose_3x3: torch.Tensor,
        betas_pred: torch.Tensor,
        batch_size: int,
        win_len: int,
    ):
        sq = batch_size * win_len
        global_orient_aa = matrix_to_angle_axis(global_orient_3x3, warn=False)
        body_pose_aa     = matrix_to_angle_axis(body_pose_3x3,     warn=False)

        bm = self.bm_male if model_input.gender == 'male' else self.bm_female
        body_parms = {
            'pose_body':   body_pose_aa.view(sq, (SmplxJoints.NUM_JTS - 1) * 3),
            'root_orient': global_orient_aa.view(sq, -1),
        }
        with torch.no_grad():
            body_local = bm(**body_parms)

        joints_local = body_local.Jtr[:, :SmplxJoints.NUM_JTS].reshape(batch_size, win_len, -1, 3)
        correction   = model_input.head_pos_global - joints_local[:, :, SmplxJoints.HEAD]
        joints_pred  = joints_local + correction[..., None, :]
        transl_pred  = correction
        return joints_pred, transl_pred

    def _batch_end(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0, norm_type=2.0)
        self.optim.step()
        self.lr_scheduler.step()

    # ------------------------------------------------------------------
    # Static delta utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _rot_delta(rot_3x3: torch.Tensor) -> torch.Tensor:
        """Frame-wise rotation delta; zero-acceleration first frame."""
        assert rot_3x3.shape[-2:] == (3, 3)
        delta = torch.empty_like(rot_3x3)
        delta[:, 1:] = torch.linalg.inv(rot_3x3[:, :-1]) @ rot_3x3[:, 1:]
        delta[:, 0]  = delta[:, 1] @ torch.linalg.inv(delta[:, 2]) @ delta[:, 1]
        return delta

    @staticmethod
    def _pos_delta(pos: torch.Tensor) -> torch.Tensor:
        """Frame-wise positional delta; zero-acceleration first frame."""
        delta = torch.empty_like(pos)
        delta[:, 1:] = pos[:, 1:] - pos[:, :-1]
        delta[:, 0]  = 2 * delta[:, 1] - delta[:, 2]
        return delta
