"""
Inspired from https://github.com/georgedf1/sfbpe/tree/main
"""

# External
#import smplx
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.parametrizations import weight_norm

# Internal
from . import base, BaseModelInput, BaseModelOutput
from anim.data.amass import SmplxJoints, YoloJoints
from utils.utils_transform import two_axis_to_matrix, matrix_to_two_axis, rotational_fk, matrix_to_angle_axis
from human_body_prior.body_model.body_model import BodyModel
import anim.bm_config as bm_C

class HMDPoserExt(base.BaseModel):
    """ Extension of HMD-Poser to gracefully exploit 'hmr_joints' and/or 'hmr_body_pose' """

    @staticmethod
    def model_str() -> str:
        return 'hmd-poser-ext'

    def __init__(self,
                 # Architecture
                 chosen_jts: list[int] = None,  # indices from SmplxJoints, None defaults to 'SEWHKA'
                 mode3d:str = None,
                 use_hmr_body_pose: bool = False,
                 use_hmr_velocities: bool = False,
                 use_rnn_layer_norm: bool = False,
                 num_betas: int = bm_C._NUM_BETAS_,
                 num_dmpls: int = bm_C._NUM_DMPLS_,
                 hidden_size: int = 256,
                 num_blocks: int = 2,
                 rnn_type: str = 'lstm',  # TODO validate
                 num_rnn_layers: int = 1,
                 num_transformer_heads: int = 8,
                 num_transformer_layers: int = 3,
                 # Loss
                 loss_func: str = 'l1',  # l1, mse  TODO validate
                 global_orient_loss_weight: float = 1.0,
                 body_pose_loss_weight: float = 5.0,
                 body_pose_global_loss_weight: float = 1.0,
                 joints_loss_weight: float = 1.0,
                 smooth_loss_weight: float = 0.5,
                 shape_loss_weight: float = 0.1,
                 extra_shape_loss_weight: float = 0.0,
                 # TODO validate hand weights
                 extra_hand_pose_loss_weight: float = 0.0,
                 extra_hand_pose_global_loss_weight: float = 0.0,
                 extra_hand_joints_loss_weight: float = 0.0,
                 # Optimizer
                 lr: float = 1e-3,
                 # Augmentation
                 augmentation_seed: int = 42,
                 use_rotational_augment: bool = False,
                 use_scale_augment: bool = False,
                 use_noise_augment: bool = False,
                 min_scale_augment: float = 0.9,
                 max_scale_augment: float = 1.1,
                 noise_augment_strength: float = 0.0,
                 ):
        super().__init__()

        if chosen_jts is None:
            if mode3d == 'gt':
                topology = SmplxJoints
            elif mode3d == 'external':
                topology = YoloJoints

            # default to SEWHKA
            chosen_jts = [
                topology.LEFT_SHOULDER, topology.RIGHT_SHOULDER,
                topology.LEFT_ELBOW, topology.RIGHT_ELBOW,
                topology.LEFT_WRIST, topology.RIGHT_WRIST,
                topology.LEFT_HIP, topology.RIGHT_HIP,
                topology.LEFT_KNEE, topology.RIGHT_KNEE,
                topology.LEFT_ANKLE, topology.RIGHT_ANKLE,
            ]

            for jt in chosen_jts:
                assert 0 < jt < topology.NUM_JTS, \
                    f"An element ({jt}) of 'chosen_jts' was outside acceptable range 1-{topology.NUM_JTS - 1}))"

        assert hidden_size % 4 == 0, f"hidden_size ({hidden_size}) must be a multiple of 4"
        assert num_blocks > 0
        assert rnn_type in ('lstm', 'gru')
        assert num_rnn_layers > 0
        assert num_transformer_heads > 0
        assert num_transformer_layers > 0
        assert loss_func in ('l1', 'mse')
        assert lr > 0
        assert num_betas > 0
        assert min_scale_augment > 0.0
        assert max_scale_augment >= min_scale_augment
        assert noise_augment_strength >= 0.0

        # --- Body models ---
        self.num_betas = num_betas
        self.num_dmpls = num_dmpls
        self.bm_male = BodyModel(bm_fname=bm_C._BM_FNAME_MALE_, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=bm_C._DMPL_FNAME_MALE_)
        self.bm_female = BodyModel(bm_fname=bm_C._BM_FNAME_MALE_, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=bm_C._DMPL_FNAME_FEMALE_)
        #self.smplx_layer = get_frozen_smplx_layer(gender='neutral', num_betas=num_betas)

        # --- HMD embeddings ---
        self.mode3d = mode3d
        # rot, rot_vel, pos, pos_vel; per hmd signal
        self.num_hmd_channels = 5  # head, lh, rh, lh_local, rh_local
        self.hmd_embers = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(nn.Linear(6, hidden_size // 4), nn.LeakyReLU()),
                nn.Sequential(nn.Linear(6, hidden_size // 4), nn.LeakyReLU()),
                nn.Sequential(nn.Linear(3, hidden_size // 4), nn.LeakyReLU()),
                nn.Sequential(nn.Linear(3, hidden_size // 4), nn.LeakyReLU())
            ]) for _ in range(self.num_hmd_channels)
        ])

        # --- Pose estimator (HMR) embeddings ---

        self.chosen_jts_local = torch.tensor(chosen_jts, dtype=torch.int32) - 1
        self.num_chosen_jts = len(chosen_jts)
        self.use_hmr_body_pose = use_hmr_body_pose
        self.use_hmr_velocities = use_hmr_velocities

        def create_hmr_embedding():
            out_factor = 1
            if use_hmr_velocities:
                out_factor *= 2
            if use_hmr_body_pose:
                out_factor *= 2
            # pos
            parts = [nn.Sequential(nn.Linear(3, hidden_size // out_factor), nn.LeakyReLU())]
            if use_hmr_velocities:
                # pos_vel
                parts.append(nn.Sequential(nn.Linear(3, hidden_size // out_factor), nn.LeakyReLU()))
            if use_hmr_body_pose:
                # rot
                parts.append(nn.Sequential(nn.Linear(6, hidden_size // out_factor), nn.LeakyReLU()))
                if use_hmr_velocities:
                    # rot_vel
                    parts.append(nn.Sequential(nn.Linear(6, hidden_size // out_factor), nn.LeakyReLU()))
            return nn.ModuleList(parts)

        # pos, [pos_vel], [rot, [rot_vel]]; per chosen joint
        self.joint_embers = nn.ModuleList([create_hmr_embedding() for _ in range(self.num_chosen_jts)])

        # ---
        self.use_rnn_layer_norm = use_rnn_layer_norm
        if self.use_rnn_layer_norm:
            self.rnn_layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        # --- Temporal encoding ---
        self.num_channels = self.num_hmd_channels + self.num_chosen_jts
        self.num_blocks = num_blocks
        rnn_module = torch.nn.LSTM if rnn_type == 'lstm' else torch.nn.GRU
        self.temporal_encoder = nn.ModuleList(
            [nn.ModuleList(
                [rnn_module(hidden_size, hidden_size, num_rnn_layers, batch_first=True)
                 for c in range(self.num_channels)]
            ) for b in range(self.num_blocks)]
        )
        self.prev_rnn_states = [[None for c in range(self.num_channels)] for b in range(self.num_blocks)]

        # --- Spatial encoding ---

        encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead=num_transformer_heads, batch_first=True)
        self.spatial_encoder = nn.ModuleList(
            # Note, this is not reuse of the encoder_layer, under the hood the layer is cloned
            [nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers) for _ in range(self.num_blocks)]
        )

        # ---

        # TODO Copied from HMD-Poser; is it necessary?
        # weight_norm and initialization
        for encoder_i in self.temporal_encoder:
            for model_i in encoder_i:
                for layer_idx in range(num_rnn_layers):
                    model_i = weight_norm(model_i, f"weight_ih_l{layer_idx}")
                    model_i = weight_norm(model_i, f"weight_hh_l{layer_idx}")
                for name, param in model_i.named_parameters():
                    if name.startswith("weight"):
                        torch.nn.init.orthogonal_(param)

        self.pose_head = nn.Sequential(
            nn.Linear(hidden_size * self.num_channels, 256),
            nn.LeakyReLU(),
            nn.Linear(256, SmplxJoints.NUM_JTS * 6)
        )
        self.shape_head = nn.Sequential(
            nn.Linear(hidden_size * self.num_channels, 256),
            nn.LeakyReLU(),
            nn.Linear(256, num_betas)
        )

        if loss_func =='l1':
            self.loss_func = nn.functional.l1_loss
        elif loss_func == 'mse':
            self.loss_func = nn.functional.mse_loss
        else:
            raise NotImplementedError

        self.global_orient_loss_weight = global_orient_loss_weight
        self.body_pose_loss_weight = body_pose_loss_weight
        self.body_pose_global_loss_weight = body_pose_global_loss_weight
        self.joints_loss_weight = joints_loss_weight
        self.smooth_loss_weight = smooth_loss_weight
        self.shape_loss_weight = shape_loss_weight
        self.extra_shape_loss_weight = extra_shape_loss_weight
        self.extra_hand_pose_loss_weight = extra_hand_pose_loss_weight
        self.extra_hand_pose_global_loss_weight = extra_hand_pose_global_loss_weight
        self.extra_hand_joints_loss_weight = extra_hand_joints_loss_weight

        # TODO Parameterise more Adam parameters?
        self.optim = torch.optim.Adam((p for p in self.parameters() if p.requires_grad), lr=lr)
        # TODO Save last_epoch for restarts?
        self.lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler([
            torch.optim.lr_scheduler.LinearLR(self.optim, start_factor=1/100, end_factor=1, total_iters=10),
            torch.optim.lr_scheduler.LinearLR(self.optim, start_factor=1, end_factor=1/100, total_iters=390)
        ])

        self.betas_pred = None

        self.augmentation_generator = torch.Generator()
        self.augmentation_generator.manual_seed(augmentation_seed)
        self.use_rotational_augment = use_rotational_augment
        self.use_scale_augment = use_scale_augment
        self.use_noise_augment = use_noise_augment
        self.min_scale_augment = min_scale_augment
        self.max_scale_augment = max_scale_augment
        self.noise_augment_strength = noise_augment_strength
        self.random_rotation = None
        self.random_scale = None
        self.new_augments_required = True


    def dataset_pass(self, dataset: Dataset, device, dtype):
        pass

    @staticmethod
    def compute_rot_delta(rot_3x3):
        # Expects shapes [batch_size, num_frames, ...]
        assert len(rot_3x3.shape) == 4
        assert rot_3x3.shape[2:] == (3, 3)
        # Delta is from previous frame (not to the next frame)
        rot_delta_3x3 = torch.empty_like(rot_3x3)
        rot_delta_3x3[:, 1:] = torch.linalg.inv(rot_3x3[:, :-1]) @ rot_3x3[:, 1:]
        # zero-acceleration approximation [rv0 := (rv2 @ rv1^{-1})^{-1} @ rv1] TODO reasonable?
        rot_delta_3x3[:, 0] = rot_delta_3x3[:, 1] @ torch.linalg.inv(rot_delta_3x3[:, 2]) @ rot_delta_3x3[:, 1]
        return rot_delta_3x3

    @staticmethod
    def compute_pos_delta(pos):
        # Expects shapes [batch_size, num_frames, ...]
        assert len(pos.shape) == 3
        assert pos.shape[2] == 3
        # Delta is from previous frame (not to the next frame)
        pos_delta = torch.empty_like(pos)
        pos_delta[:, 1:] = pos[:, 1:] - pos[:, :-1]
        # zero-acceleration approximation [v0 := v1 - (v2 - v1)] TODO reasonable?
        pos_delta[:, 0] = (2 * pos_delta[:, 1]) - pos_delta[:, 2]
        return pos_delta

    def reset(self):
        self.new_augments_required = True
        self.prev_rnn_states = [[None for c in range(self.num_channels)] for b in range(self.num_blocks)]

    def forward(self, model_input: BaseModelInput) -> BaseModelOutput:
        batch_size = model_input.batch_size
        win_len = model_input.win_len

        # --- Compute input features ---

        head_rot_3x3_global_inv = torch.linalg.inv(model_input.head_rot_global)
        hmd_posis = [
            # Head + hands
            model_input.head_pos_global,
            model_input.lh_pos_global,
            model_input.rh_pos_global,
            # Hands in head-local space
            (head_rot_3x3_global_inv @ (model_input.lh_pos_global - model_input.head_pos_global)[..., None])[..., 0],
            (head_rot_3x3_global_inv @ (model_input.rh_pos_global - model_input.head_pos_global)[..., None])[..., 0],
        ]
        hmd_rots = [
            # Head + hands
            model_input.head_rot_global,
            model_input.lh_rot_global,
            model_input.rh_rot_global,
            # Hands in head-local space
            head_rot_3x3_global_inv @ model_input.lh_rot_global,
            head_rot_3x3_global_inv @ model_input.rh_rot_global,
        ]

        # Model variation that only uses body_pose
        hmr_joints_local = model_input.hmr_joints[:, :, 1:SmplxJoints.NUM_JTS] - model_input.hmr_joints[:, :, :1]
        chosen_joints = hmr_joints_local[:, :, self.chosen_jts_local]
        chosen_body_pose = model_input.hmr_body_pose[:, :, self.chosen_jts_local]

        # At train time, hmr_joints are ground-truth (synthetic) world-space joints
        #   So we need to make robust to arbitrary rotation, scale, and noise...
        if self.training:
            if self.new_augments_required:
                # Make two "random" perpendicular unit vectors, not a uniform rotation sampler, but simple one
                random_vectors = torch.rand((batch_size, 2, 3), generator=self.augmentation_generator).to(
                    dtype=chosen_joints.dtype, device=chosen_joints.device)
                random_vectors[:, 1] = random_vectors[:, 0].cross(random_vectors[:, 1], dim=-1)
                random_vectors = random_vectors / random_vectors.square().sum(dim=-1, keepdim=True).sqrt()
                random_rotation_6d = random_vectors.reshape(batch_size, 1, 6).expand(batch_size, win_len, 6)
                self.random_rotation = two_axis_to_matrix(random_rotation_6d)

                self.random_scale = torch.rand((batch_size,), generator=self.augmentation_generator).to(
                    dtype=chosen_joints.dtype, device=chosen_joints.device)
                self.random_scale = self.min_scale_augment + (
                        self.random_scale * (self.max_scale_augment - self.min_scale_augment))

                self.new_augments_required = False

            # Rotational augmentation; robustness to fixed but various camera poses
            if self.use_rotational_augment:
                chosen_joints = (self.random_rotation[:, :, None] @ chosen_joints[..., None])[..., 0]

            # Scale augmentation; since chosen_joints is already PELVIS local, multiplication is valid scale augment
            if self.use_scale_augment:
                chosen_joints *= self.random_scale[:, None, None, None]

            # Additive noise augmentation; robustness to noisy pose estimation
            if self.use_noise_augment:
                # uniform noise in [-0.5, 0.5]
                noise = (2 * torch.rand(chosen_joints.shape, generator=self.augmentation_generator)) - 1
                noise = (self.noise_augment_strength * noise).to(dtype=chosen_joints.dtype, device=chosen_joints.device)
                chosen_joints += noise

            # TODO Add noise to chosen_body_pose?

            # TODO Perform more augmentation?

        # --- Embeddings ---

        embeddings = []

        # HMD embeddings
        for c in range(self.num_hmd_channels):
            pos = hmd_posis[c]
            rot_3x3 = hmd_rots[c]
            pos_delta = self.compute_pos_delta(pos)
            rot_delta_3x3 = self.compute_rot_delta(rot_3x3)
            rot_6d = matrix_to_two_axis(rot_3x3)
            rot_delta_6d = matrix_to_two_axis(rot_delta_3x3)
            feats = [rot_6d, rot_delta_6d, pos, pos_delta]
            emb = torch.cat([self.hmd_embers[c][i](feat) for i, feat in enumerate(feats)], dim=-1)
            if self.use_rnn_layer_norm:
                emb = self.rnn_layer_norm(emb)
            embeddings.append(emb)

        # Pose estimator embeddings
        for c in range(self.num_chosen_jts):
            pos = chosen_joints[:, :, c]
            feats = [pos]
            if self.use_hmr_velocities:
                pos_delta = self.compute_pos_delta(pos)
                feats.append(pos_delta)
            if self.use_hmr_body_pose:
                rot_3x3 = chosen_body_pose[:, :, c]
                rot_6d = matrix_to_two_axis(rot_3x3)
                feats.append(rot_6d)
                if self.use_hmr_velocities:
                    rot_delta_3x3 = self.compute_rot_delta(rot_3x3)
                    rot_delta_6d = matrix_to_two_axis(rot_delta_3x3)
                    feats.append(rot_delta_6d)
            emb = torch.cat([self.joint_embers[c][i](feat) for i, feat in enumerate(feats)], dim=-1)
            if self.use_rnn_layer_norm:
                emb = self.rnn_layer_norm(emb)
            embeddings.append(emb)

        # Pack into one tensor of shape (batch_size, win_len, self.num_channels, hidden_size)
        feats = torch.stack(embeddings, dim=-2)

        # --- Temporal (LSTM) + Spatial (TransformerEncoder) blocks ---
        for b in range(self.num_blocks):
            feats = feats.reshape(batch_size, win_len, self.num_channels, -1)
            # --- Temporal ---
            # Run temporal encoder per feature group
            feats_temporal = []
            for c in range(self.num_channels):
                prev_rnn_state = self.prev_rnn_states[b][c]
                rnn_output, rnn_state = self.temporal_encoder[b][c](feats[:, :, c, :], prev_rnn_state)
                self.prev_rnn_states[b][c] = rnn_state
                feats_temporal.append(rnn_output)
            feats_temporal = torch.stack(feats_temporal, dim=-2)
            # --- Spatial ---
            # Pack frames into the batch dimension for efficiency
            feats_temporal_tok = feats_temporal.reshape(batch_size * win_len, self.num_channels, -1)
            feats = self.spatial_encoder[b](feats_temporal_tok)

        # --- Prediction heads ---
        feats = feats.reshape(batch_size, win_len, -1)
        pose_pred = self.pose_head(feats)
        pose_pred = pose_pred.reshape(batch_size, win_len, SmplxJoints.NUM_JTS, 6)
        global_orient_6d_pred = pose_pred[:, :, 0]
        body_pose_6d_pred = pose_pred[:, :, 1:]
        betas_pred = self.shape_head(feats)
        self.betas_pred = betas_pred

        # --- Computing transl_pred and joints_pred ---
        global_orient_3x3_pred = two_axis_to_matrix(global_orient_6d_pred)
        body_pose_3x3_pred = two_axis_to_matrix(body_pose_6d_pred)
        global_orient_aa_pred = matrix_to_angle_axis(global_orient_3x3_pred)
        body_pose_aa_pred = matrix_to_angle_axis(body_pose_3x3_pred)

        sq_size = batch_size * win_len

        bm = self.bm_male if model_input.gender == 'male' else self.bm_female
        body_parms_pred = {
            'pose_body': body_pose_aa_pred.view(sq_size, (SmplxJoints.NUM_JTS-1)*3),
            'root_orient' : global_orient_aa_pred.view(sq_size, -1)
        }  
        # Log shapes of each tensor in the dict
        
        body_pose_local = bm(**{k:v for k, v in body_parms_pred.items() if k in ['pose_body', 'root_orient']})
        
        '''
        smplx_output_pred = self.smplx_layer(
            betas=betas_pred.view(sq_size, self.num_betas),
            global_orient=global_orient_3x3_pred.view(sq_size, 3, 3),
            body_pose=body_pose_3x3_pred.view(sq_size, SmplxJoints.NUM_JTS - 1, 3, 3))
        '''
        
        joints_local_pred = (body_pose_local.Jtr[:, :SmplxJoints.NUM_JTS]
                             .reshape(batch_size, win_len, -1, 3))

        #vertices_local_pred = smplx_output_pred.vertices.reshape(batch_size, win_len, -1, 3)
        head_pos_local_pred = joints_local_pred[:, :, SmplxJoints.HEAD]
        # Force predictions to agree with head ground truth since HMD 6DoF always available
        correction = model_input.head_pos_global - head_pos_local_pred
        #transl_pred = correction
        joints_pred = joints_local_pred + correction[..., None, :]
        #vertices_pred = vertices_local_pred + correction[..., None, :]

        return base.BaseModelOutput(
            betas=betas_pred,
            #transl=transl_pred,
            global_orient=global_orient_3x3_pred,
            body_pose=body_pose_3x3_pred,
            joints=joints_pred,
            gender = model_input.gender
            #vertices=vertices_pred,
            #faces=self.smplx_layer.faces,
        )

    def forward_pass(self, model_input: BaseModelInput, model_target: BaseModelOutput, optimise=False) -> dict:
        model_output = self(model_input)

        loss_dict = {}

        # "ori" loss in paper
        global_orient_3x3_loss = self.loss_func(model_output.global_orient, model_target.global_orient)
        loss = self.global_orient_loss_weight * global_orient_3x3_loss
        loss_dict['global_orient_3x3_loss'] = global_orient_3x3_loss

        # "lrot" loss in paper
        body_pose_3x3_loss = self.loss_func(model_output.body_pose, model_target.body_pose)
        loss += self.body_pose_loss_weight * body_pose_3x3_loss
        loss_dict['body_pose_3x3_loss'] = body_pose_3x3_loss
        # extra hand loss
        if self.extra_hand_pose_loss_weight > 0.0:
            extra_hand_pose_loss = self.loss_func(
                model_output.body_pose[:, :, [SmplxJoints.LEFT_WRIST - 1, SmplxJoints.RIGHT_WRIST - 1]],
                model_target.body_pose[:, :, [SmplxJoints.LEFT_WRIST - 1, SmplxJoints.RIGHT_WRIST - 1]])
            loss += self.extra_hand_pose_loss_weight * extra_hand_pose_loss
            loss_dict['extra_hand_pose_loss'] = extra_hand_pose_loss

        # "grot" loss in paper
        with torch.no_grad():
            body_pose_global_3x3 = rotational_fk(model_target.global_orient, model_target.body_pose)
        body_pose_global_3x3_pred = rotational_fk(model_output.global_orient, model_output.body_pose)
        body_pose_global_3x3_loss = self.loss_func(body_pose_global_3x3_pred, body_pose_global_3x3)
        loss += self.body_pose_global_loss_weight * body_pose_global_3x3_loss
        loss_dict['body_pose_global_3x3_loss'] = body_pose_global_3x3_loss
        # extra hand loss
        if self.extra_hand_pose_global_loss_weight > 0.0:
            extra_hand_pose_global_loss = self.loss_func(
                body_pose_global_3x3_pred[:, :, [SmplxJoints.LEFT_WRIST - 1, SmplxJoints.RIGHT_WRIST - 1]],
                body_pose_global_3x3[:, :, [SmplxJoints.LEFT_WRIST - 1, SmplxJoints.RIGHT_WRIST - 1]])
            loss += self.extra_hand_pose_global_loss_weight * extra_hand_pose_global_loss
            loss_dict['extra_hand_pose_global_loss'] = extra_hand_pose_global_loss

        # "joint" loss in paper
        joints_target = model_target.joints
        joints_pred = model_output.joints
        joints_loss = self.loss_func(joints_pred, joints_target)
        loss += self.joints_loss_weight * joints_loss
        loss_dict['joints_loss'] = joints_loss
        # extra hand loss
        if self.extra_hand_joints_loss_weight > 0.0:
            extra_hand_joints_loss = self.loss_func(
                joints_pred[:, :, [SmplxJoints.LEFT_WRIST, SmplxJoints.RIGHT_WRIST]],
                joints_target[:, :, [SmplxJoints.LEFT_WRIST, SmplxJoints.RIGHT_WRIST]])
            loss += self.extra_hand_joints_loss_weight * extra_hand_joints_loss
            loss_dict['extra_hand_joints_loss'] = extra_hand_joints_loss

        # "smooth" loss in paper
        accel_target = joints_target[:, :-2, :] - 2 * joints_target[:, 1:-1, :] + joints_target[:, 2:, :]
        accel_pred = joints_pred[:, :-2, :] - 2 * joints_pred[:, 1:-1, :] + joints_pred[:, 2:, :]
        smooth_loss = self.loss_func(accel_pred, accel_target)
        loss += self.smooth_loss_weight * smooth_loss
        loss_dict['smooth_loss'] = smooth_loss

        # mysterious shape loss in not in paper, but in paper code, presumably for regularisation
        betas_pred = model_output.betas
        betas_pred_temporal_mean = betas_pred.mean(dim=1, keepdim=True).repeat(1, betas_pred.shape[1], 1)
        shape_loss = self.loss_func(betas_pred, betas_pred_temporal_mean, reduction='mean')
        loss += self.shape_loss_weight * shape_loss
        loss_dict['shape_loss'] = shape_loss

        # my own additional regularisation; keep close to average shape (zero)
        if self.extra_shape_loss_weight > 0.0:
            extra_shape_loss = self.loss_func(betas_pred, torch.zeros_like(betas_pred))
            loss += self.extra_shape_loss_weight * extra_shape_loss
            loss_dict['extra_shape_loss'] = extra_shape_loss

        # Metrics
        with torch.no_grad():
            loss_dict['MPJPE(cm)'] = 100 * (joints_target - joints_pred).square().sum(dim=-1).sqrt().mean()

        loss_dict['loss'] = loss

        if optimise:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        return loss_dict

    def epoch_end(self, epoch: int, train_losses: dict, val_losses: dict):
        self.lr_scheduler.step()
