import einops
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork

from model.utils.position_encodings import RotaryPositionEncoding3D
from model.utils.layers import (
    RelativeCrossAttentionLayer,
    FeedforwardLayer,
    RelativeCrossAttentionModule,
    TaskSpecificRelativeCrossAttentionModule
)
from model.utils.utils import normalise_quat, sample_ghost_points_uniform_cube, sample_ghost_points_uniform_sphere
from model.utils.resnet import load_resnet50
from model.utils.clip import load_clip


class AnalogicalPredictionHead(nn.Module):
    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 num_ghost_point_cross_attn_layers=2,
                 num_attn_heads=4,
                 gripper_loc_bounds=None,
                 num_ghost_points=1000,
                 num_ghost_points_val=10000,
                 weight_tying=True,
                 gp_emb_tying=True,
                 num_sampling_level=3,
                 fine_sampling_ball_diameter=0.16,
                 regress_position_offset=False,
                 use_instruction=False,
                 task_specific_biases=False,
                 task_ids=[],
                 positional_features="none",
                 support_set="others",
                 global_correspondence=False,
                 num_matching_cross_attn_layers=2):
        super().__init__()
        assert backbone in ["resnet", "clip"]
        assert image_size in [(128, 128), (256, 256)]
        assert num_sampling_level in [1, 2, 3, 4]
        assert positional_features in ["xyz_concat", "z_concat", "xyz_add", "z_add", "none"]
        assert support_set in ["self", "others"]

        self.image_size = image_size
        self.num_ghost_points = num_ghost_points // num_sampling_level
        self.num_ghost_points_val = num_ghost_points_val // num_sampling_level
        self.num_sampling_level = num_sampling_level
        self.sampling_ball_diameter_pyramid = [
            None,
            fine_sampling_ball_diameter,
            fine_sampling_ball_diameter / 4.0,
            fine_sampling_ball_diameter / 16.0
        ]
        self.gripper_loc_bounds = np.array(gripper_loc_bounds)
        self.regress_position_offset = regress_position_offset
        self.weight_tying = weight_tying
        self.gp_emb_tying = gp_emb_tying
        self.positional_features = positional_features
        self.support_set = support_set
        self.global_correspondence = global_correspondence

        # Frozen backbone
        if backbone == "resnet":
            self.backbone, self.normalize = load_resnet50()
        elif backbone == "clip":
            self.backbone, self.normalize = load_clip()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Semantic visual features at different scales
        if self.positional_features in ["xyz_concat", "z_concat"]:
            self.feature_pyramid = FeaturePyramidNetwork(
                [64, 256, 512, 1024, 2048], embedding_dim - embedding_dim // 10)
        else:
            self.feature_pyramid = FeaturePyramidNetwork(
                [64, 256, 512, 1024, 2048], embedding_dim)
        if self.image_size == (128, 128):
            # Coarse RGB features are the 2nd layer of the feature pyramid at 1/4 resolution (32x32)
            # Fine RGB features are the 1st layer of the feature pyramid at 1/2 resolution (64x64)
            self.coarse_feature_map = ['res2', 'res1', 'res1', 'res1']
            self.downscaling_factor_pyramid = [4, 2, 2, 2]
        elif self.image_size == (256, 256):
            # Coarse RGB features are the 3rd layer of the feature pyramid at 1/8 resolution (32x32)
            # Fine RGB features are the 1st layer of the feature pyramid at 1/2 resolution (128x128)
            self.feature_map_pyramid = ['res3', 'res1', 'res1', 'res1']
            self.downscaling_factor_pyramid = [8, 2, 2, 2]

        # 3D relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # 3D absolute positional embeddings (only used for positional features, if any)
        if self.positional_features == "xyz_concat":
            self.absolute_pe_layer = LearnedAbsolutePositionEncoding3D(3, embedding_dim // 10)
        elif self.positional_features == "z_concat":
            self.absolute_pe_layer = LearnedAbsolutePositionEncoding3D(1, embedding_dim // 10)
        if self.positional_features == "xyz_add":
            self.absolute_pe_layer = LearnedAbsolutePositionEncoding3D(3, embedding_dim)
        elif self.positional_features == "z_add":
            self.absolute_pe_layer = LearnedAbsolutePositionEncoding3D(1, embedding_dim)

        # Ghost points learnable initial features
        self.ghost_points_embed_pyramid = nn.ModuleList()
        if self.gp_emb_tying:
            gp_emb = nn.Embedding(1, embedding_dim)
            for _ in range(self.num_sampling_level):
                self.ghost_points_embed_pyramid.append(gp_emb)
        else:
            for _ in range(self.num_sampling_level):
                self.ghost_points_embed_pyramid.append(nn.Embedding(1, embedding_dim))

        # Current gripper learnable features
        self.curr_gripper_embed = nn.Embedding(1, embedding_dim)

        # Query learnable features
        self.query_embed = nn.Embedding(1, embedding_dim)

        # Ghost point cross-attention to visual features and current gripper position
        self.task_specific_biases = task_specific_biases
        if self.task_specific_biases:
            self.ghost_point_cross_attn_pyramid = nn.ModuleList()
            if self.weight_tying:
                ghost_point_cross_attn = TaskSpecificRelativeCrossAttentionModule(
                    embedding_dim, num_attn_heads, num_ghost_point_cross_attn_layers, task_ids)
                for _ in range(self.num_sampling_level):
                    self.ghost_point_cross_attn_pyramid.append(ghost_point_cross_attn)
            else:
                for _ in range(self.num_sampling_level):
                    ghost_point_cross_attn = TaskSpecificRelativeCrossAttentionModule(
                        embedding_dim, num_attn_heads, num_ghost_point_cross_attn_layers, task_ids)
                    self.ghost_point_cross_attn_pyramid.append(ghost_point_cross_attn)
        else:
            self.ghost_point_cross_attn_pyramid = nn.ModuleList()
            if self.weight_tying:
                ghost_point_cross_attn = RelativeCrossAttentionModule(
                    embedding_dim, num_attn_heads, num_ghost_point_cross_attn_layers)
                for _ in range(self.num_sampling_level):
                    self.ghost_point_cross_attn_pyramid.append(ghost_point_cross_attn)
            else:
                for _ in range(self.num_sampling_level):
                    ghost_point_cross_attn = RelativeCrossAttentionModule(
                        embedding_dim, num_attn_heads, num_ghost_point_cross_attn_layers)
                    self.ghost_point_cross_attn_pyramid.append(ghost_point_cross_attn)

        # Ghost point matching self-attention and cross-attention with support set
        # TODO If task-specific biases help, use them here too
        self.matching_cross_attn_layers = nn.ModuleList()
        self.matching_self_attn_layers = nn.ModuleList()
        self.matching_ffw_layers = nn.ModuleList()
        for _ in range(num_matching_cross_attn_layers):
            self.matching_cross_attn_layers.append(RelativeCrossAttentionLayer(embedding_dim, num_attn_heads))
            self.matching_self_attn_layers.append(RelativeCrossAttentionLayer(embedding_dim, num_attn_heads))
            self.matching_ffw_layers.append(FeedforwardLayer(embedding_dim, embedding_dim))

        # Ghost point offset prediction
        if self.regress_position_offset:
            self.ghost_point_offset_predictor = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 3)
            )

        # Gripper rotation (quaternion) and binary opening prediction
        self.gripper_state_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 4 + 1)
        )

        # Instruction encoder
        self.use_instruction = use_instruction
        if self.use_instruction:
            self.instruction_encoder = nn.Linear(512, embedding_dim)

    def forward(self,
                visible_rgb, visible_pcd, curr_gripper, instruction, task_id,
                padding_mask, gt_action_for_support, gt_action_for_sampling=None):
        """
        Arguments:
            visible_rgb: (batch, 1 + support, history, num_cameras, 3, height, width) in [0, 1]
            visible_pcd: (batch, 1 + support, history, num_cameras, 3, height, width) in world coordinates
            curr_gripper: (batch, 1 + support, history, 3)
            instruction: (batch, 1 + support, history, max_instruction_length, 512)
            task_id: (batch, 1 + support, history)
            padding_mask: (batch, 1 + support, history)
            gt_action_for_support: ground-truth action used as the support set
             of shape (batch, 1 + support, history, 8) in world coordinates
            gt_action_for_sampling: ground-truth action used to guide ghost point sampling
             of shape (batch, 1 + support, history, 8) in world coordinates

        Use of (1 + support) dimension:
        - During training, all demos in the set come from the same train split, and we use
           each demo in this dimension for training with all other demos as the support set
        - During evaluation, only the first demo in this dimension comes from the val split while
           others come from the train split and act as the support set
        """
        batch_size, demos_per_task, history_size, num_cameras, _, height, width = visible_rgb.shape
        device = visible_rgb.device

        visible_rgb = visible_rgb[padding_mask]
        visible_pcd = visible_pcd[padding_mask]
        curr_gripper = curr_gripper[padding_mask]
        instruction = instruction[padding_mask]
        task_id = task_id[padding_mask]
        if gt_action_for_sampling is not None:
            gt_position_for_sampling = gt_action_for_sampling[padding_mask][:, :3].unsqueeze(-2).detach()
        else:
            gt_position_for_sampling = None
        gt_position_for_support = gt_action_for_support[:, :, :, :3].unsqueeze(-2).detach()
        total_timesteps = visible_rgb.shape[0]

        # Compute visual features at different scales and their positional embeddings
        visible_rgb_features_pyramid, visible_rgb_pos_pyramid, visible_pcd_pyramid = self._compute_visual_features(
            visible_rgb, visible_pcd, num_cameras)

        # Encode instruction
        if self.use_instruction:
            instruction_features = einops.rearrange(self.instruction_encoder(instruction), "bst l c -> l bst c")
            instruction_dummy_pos = torch.zeros(total_timesteps, instruction_features.shape[0], 3, device=device)
            instruction_dummy_pos = self.relative_pe_layer(instruction_dummy_pos)
        else:
            instruction_features = None
            instruction_dummy_pos = None

        # Compute current gripper position features and positional embeddings
        curr_gripper_pos = self.relative_pe_layer(curr_gripper.unsqueeze(1))
        curr_gripper_features = self.curr_gripper_embed.weight.repeat(total_timesteps, 1).unsqueeze(0)

        ghost_pcd_features_pyramid = []
        ghost_pcd_pyramid = []
        position_pyramid = []
        ghost_pcd_masks_pyramid = []

        for i in range(self.num_sampling_level):
            # Sample ghost points
            if i == 0:
                anchor = None
            else:
                anchor = gt_position_for_sampling if gt_position_for_sampling is not None else position_pyramid[-1]
            ghost_pcd_i = self._sample_ghost_points(total_timesteps, device, level=i, anchor=anchor)

            if i == 0:
                # Global coarse RGB features
                visible_rgb_features_i = visible_rgb_features_pyramid[i]
                visible_rgb_pos_i = visible_rgb_pos_pyramid[i]
                ghost_pcd_context_features_i = einops.rearrange(
                    visible_rgb_features_i, "bst ncam c h w -> (ncam h w) bst c")
            else:
                # Local fine RGB features
                l2_pred_pos = ((anchor - visible_pcd_pyramid[i]) ** 2).sum(-1).sqrt()
                indices = l2_pred_pos.topk(k=32 * 32 * num_cameras, dim=-1, largest=False).indices

                visible_rgb_features_i = einops.rearrange(
                    visible_rgb_features_pyramid[i], "bst ncam c h w -> bst (ncam h w) c")
                visible_rgb_features_i = torch.stack([
                    f[i] for (f, i) in zip(visible_rgb_features_i, indices)])
                visible_rgb_pos_i = torch.stack([
                    f[i] for (f, i) in zip(visible_rgb_pos_pyramid[i], indices)])
                ghost_pcd_context_features_i = einops.rearrange(
                    visible_rgb_features_i, "bst npts c -> npts bst c")

            # Compute ghost point features and their positional embeddings by attending to visual
            # features and current gripper position
            ghost_pcd_context_features_i = torch.cat(
                [ghost_pcd_context_features_i, curr_gripper_features], dim=0)
            ghost_pcd_context_pos_i = torch.cat([visible_rgb_pos_i, curr_gripper_pos], dim=1)
            if self.use_instruction:
                ghost_pcd_context_features_i = torch.cat(
                    [ghost_pcd_context_features_i, instruction_features], dim=0)
                ghost_pcd_context_pos_i = torch.cat(
                    [ghost_pcd_context_pos_i, instruction_dummy_pos], dim=1)
            (
                ghost_pcd_features_i,
                ghost_pcd_pos_i,
            ) = self._compute_ghost_point_features(
                ghost_pcd_i, ghost_pcd_context_features_i, ghost_pcd_context_pos_i,
                task_id, total_timesteps, level=i
            )

            # Compute ghost point similarity scores with the ground-truth ghost points
            # in the support set at the corresponding timestep
            ghost_pcd_masks_i = self._match_ghost_points(
                ghost_pcd_features_i, ghost_pcd_pos_i, ghost_pcd_i, gt_position_for_support,
                padding_mask, batch_size, demos_per_task, history_size, device
            )

            top_idx = torch.max(ghost_pcd_masks_i, dim=-1).indices
            ghost_pcd_i = einops.rearrange(ghost_pcd_i, "b npts c -> b c npts")
            position_i = ghost_pcd_i[torch.arange(total_timesteps), :, top_idx].unsqueeze(1)

            ghost_pcd_pyramid.append(ghost_pcd_i)
            ghost_pcd_features_pyramid.append(ghost_pcd_features_i)
            position_pyramid.append(position_i)
            ghost_pcd_masks_pyramid.append([ghost_pcd_masks_i])

        # Regress an offset from the ghost point's position to the predicted position
        if self.regress_position_offset:
            fine_ghost_pcd_offsets = self.ghost_point_offset_predictor(ghost_pcd_features_i)
            fine_ghost_pcd_offsets = einops.rearrange(fine_ghost_pcd_offsets, "npts b c -> b c npts")
        else:
            fine_ghost_pcd_offsets = None

        ghost_pcd = ghost_pcd_i
        ghost_pcd_masks = ghost_pcd_masks_i
        ghost_pcd_features = ghost_pcd_features_i

        # Predict the next gripper action (position, rotation, gripper opening)
        position, rotation, gripper = self._predict_action(
            ghost_pcd_masks[-1], ghost_pcd, ghost_pcd_features, total_timesteps,
            fine_ghost_pcd_offsets if self.regress_position_offset else None
        )

        return {
            # Action
            "position": position,
            "rotation": rotation,
            "gripper": gripper,
            # Auxiliary outputs used to compute the loss or for visualization
            "position_pyramid": position_pyramid,
            "ghost_pcd_masks_pyramid": ghost_pcd_masks_pyramid,
            "ghost_pcd_pyramid": ghost_pcd_pyramid,
            "fine_ghost_pcd_offsets": fine_ghost_pcd_offsets if self.regress_position_offset else None,
        }

    def _compute_visual_features(self, visible_rgb, visible_pcd, num_cameras):
        """Compute visual features at different scales and their positional embeddings."""
        ncam = visible_rgb.shape[1]

        # Pass each view independently through backbone
        visible_rgb = einops.rearrange(visible_rgb, "bst ncam c h w -> (bst ncam) c h w")
        visible_rgb = self.normalize(visible_rgb)
        visible_rgb_features = self.backbone(visible_rgb)

        # Pass visual features through feature pyramid network
        visible_rgb_features = self.feature_pyramid(visible_rgb_features)

        visible_pcd = einops.rearrange(visible_pcd, "bst ncam c h w -> (bst ncam) c h w")

        visible_rgb_features_pyramid = []
        visible_rgb_pos_pyramid = []
        visible_pcd_pyramid = []

        for i in range(self.num_sampling_level):
            visible_rgb_features_i = visible_rgb_features[self.feature_map_pyramid[i]]
            visible_pcd_i = F.interpolate(
                visible_pcd, scale_factor=1. / self.downscaling_factor_pyramid[i], mode='bilinear')
            h, w = visible_pcd_i.shape[-2:]
            visible_pcd_i = einops.rearrange(
                visible_pcd_i, "(bst ncam) c h w -> bst (ncam h w) c", ncam=num_cameras)
            visible_rgb_pos_i = self.relative_pe_layer(visible_pcd_i)
            visible_rgb_features_i = einops.rearrange(
                visible_rgb_features_i, "(bst ncam) c h w -> bst ncam c h w", ncam=num_cameras)

            if self.positional_features in ["xyz_concat", "xyz_add"]:
                visible_rgb_pos_features_i = self.absolute_pe_layer(visible_pcd_i)
                visible_rgb_pos_features_i = einops.rearrange(
                    visible_rgb_pos_features_i, "bst (ncam h w) c -> bst ncam c h w", ncam=ncam, h=h, w=w)
            elif self.positional_features in ["z_concat", "z_add"]:
                visible_rgb_pos_features_i = self.absolute_pe_layer(visible_pcd_i[:, :, 2:3])
                visible_rgb_pos_features_i = einops.rearrange(
                    visible_rgb_pos_features_i, "bst (ncam h w) c -> bst ncam c h w", ncam=ncam, h=h, w=w)

            if self.positional_features in ["xyz_concat", "z_concat"]:
                visible_rgb_features_i = torch.cat([visible_rgb_features_i, visible_rgb_pos_features_i], dim=2)
            elif self.positional_features in ["xyz_add", "z_add"]:
                visible_rgb_features_i = visible_rgb_features_i + visible_rgb_pos_features_i

            visible_rgb_features_pyramid.append(visible_rgb_features_i)
            visible_rgb_pos_pyramid.append(visible_rgb_pos_i)
            visible_pcd_pyramid.append(visible_pcd_i)

        return visible_rgb_features_pyramid, visible_rgb_pos_pyramid, visible_pcd_pyramid

    def _sample_ghost_points(self, total_timesteps, device, level, anchor=None):
        """Sample ghost points.

        If level==0, sample points uniformly within the workspace bounds.

        If level>0, sample points uniformly within a local sphere
        of the workspace bounds centered around the anchor.
        """
        if self.training:
            num_ghost_points = self.num_ghost_points
        else:
            num_ghost_points = self.num_ghost_points_val

        if level == 0:
            bounds = np.stack([self.gripper_loc_bounds for _ in range(total_timesteps)])
            uniform_pcd = np.stack([
                sample_ghost_points_uniform_cube(
                    bounds=bounds[i],
                    num_points=num_ghost_points
                )
                for i in range(total_timesteps)
            ])

        elif level >= 1:
            anchor_ = anchor[:, 0].cpu().numpy()
            bounds_min = np.clip(
                anchor_ - self.sampling_ball_diameter_pyramid[level] / 2,
                a_min=self.gripper_loc_bounds[0], a_max=self.gripper_loc_bounds[1]
            )
            bounds_max = np.clip(
                anchor_ + self.sampling_ball_diameter_pyramid[level] / 2,
                a_min=self.gripper_loc_bounds[0], a_max=self.gripper_loc_bounds[1]
            )
            bounds = np.stack([bounds_min, bounds_max], axis=1)
            uniform_pcd = np.stack([
                sample_ghost_points_uniform_sphere(
                    center=anchor_[i],
                    radius=self.sampling_ball_diameter_pyramid[level] / 2,
                    bounds=bounds[i],
                    num_points=num_ghost_points
                )
                for i in range(total_timesteps)
            ])

        uniform_pcd = torch.from_numpy(uniform_pcd).float().to(device)

        return uniform_pcd

    def _compute_ghost_point_features(self,
                                      ghost_pcd, context_features, context_pos,
                                      task_id, total_timesteps, level):
        """
        Ghost points cross-attend to context features (visual features and current
        gripper position).
        """
        embed = self.ghost_points_embed_pyramid[level]
        attn_layers = self.ghost_point_cross_attn_pyramid[level]

        # Initialize ghost point features and positional embeddings
        ghost_pcd_pos = self.relative_pe_layer(ghost_pcd)
        num_ghost_points = ghost_pcd.shape[1]
        ghost_pcd_features = embed.weight.unsqueeze(0).repeat(num_ghost_points, total_timesteps, 1)

        # Ghost points cross-attend to visual features and current gripper position
        if self.task_specific_biases:
            ghost_pcd_features = attn_layers(
                task_id=task_id,
                query=ghost_pcd_features, value=context_features,
                query_pos=ghost_pcd_pos, value_pos=context_pos
            )[-1]
        else:
            ghost_pcd_features = attn_layers(
                query=ghost_pcd_features, value=context_features,
                query_pos=ghost_pcd_pos, value_pos=context_pos
            )[-1]

        return ghost_pcd_features, ghost_pcd_pos

    def _match_ghost_points(self,
                            ghost_pcd_features, ghost_pcd_pos, ghost_pcd, gt_position_for_support,
                            padding_mask, batch_size, demos_per_task, history_size, device):
        """Compute ghost point similarity scores with the ground-truth ghost point
        in the support set at the corresponding timestep.
        """
        # TODO Matching a specific timestep breaks down for tasks with a variable number
        #  of timesteps

        # Until now, we processed each timestep in parallel across the batch, support, and history
        # Here we reshape the tensors to be of shape (batch, 1 + support, history, ...) for matching
        ghost_pcd_ = torch.zeros(
            batch_size, demos_per_task, history_size, *ghost_pcd.shape[1:], device=device)
        ghost_pcd_[padding_mask] = ghost_pcd
        ghost_pcd = ghost_pcd_
        ghost_pcd_features_ = torch.zeros(
            ghost_pcd_features.shape[0], batch_size, demos_per_task,
            history_size, ghost_pcd_features.shape[-1], device=device
        )
        ghost_pcd_features_[:, padding_mask] = ghost_pcd_features
        ghost_pcd_features = ghost_pcd_features_
        ghost_pcd_pos_ = torch.zeros(
            batch_size, demos_per_task, history_size, *ghost_pcd_pos.shape[1:], device=device)
        ghost_pcd_pos_[padding_mask] = ghost_pcd_pos
        ghost_pcd_pos = ghost_pcd_pos_
        num_points, _, _, _, channels = ghost_pcd_features.shape

        if self.global_correspondence:
            # TODO This assumes there is a single demo in the support set for now
            assert self.support_set == "others"
            assert ghost_pcd_features.shape[2] == 2

            ghost_pcd_features1 = ghost_pcd_features[:, :, 0][:, padding_mask[:, 0]]
            ghost_pcd_features2 = ghost_pcd_features[:, :, 1][:, padding_mask[:, 1]]
            ghost_pcd_pos1 = ghost_pcd_pos[:, 0][padding_mask[:, 0]]
            ghost_pcd_pos2 = ghost_pcd_pos[:, 1][padding_mask[:, 1]]

            for i in range(len(self.matching_cross_attn_layers)):
                ghost_pcd_features1_prev = ghost_pcd_features1
                ghost_pcd_features2_prev = ghost_pcd_features2

                ghost_pcd_features1, _ = self.matching_cross_attn_layers[i](
                    query=ghost_pcd_features1_prev, value=ghost_pcd_features2_prev,
                    query_pos=ghost_pcd_pos1, value_pos=ghost_pcd_pos2
                )
                ghost_pcd_features2, _ = self.matching_cross_attn_layers[i](
                    query=ghost_pcd_features2_prev, value=ghost_pcd_features1_prev,
                    query_pos=ghost_pcd_pos2, value_pos=ghost_pcd_pos1
                )
                ghost_pcd_features1, _ = self.matching_self_attn_layers[i](
                    query=ghost_pcd_features1, value=ghost_pcd_features1,
                    query_pos=ghost_pcd_pos1, value_pos=ghost_pcd_pos1
                )
                ghost_pcd_features2, _ = self.matching_self_attn_layers[i](
                    query=ghost_pcd_features2, value=ghost_pcd_features2,
                    query_pos=ghost_pcd_pos2, value_pos=ghost_pcd_pos2
                )
                ghost_pcd_features1 = self.matching_ffw_layers[i](ghost_pcd_features1)
                ghost_pcd_features2 = self.matching_ffw_layers[i](ghost_pcd_features2)

            ghost_pcd_features = torch.zeros(
                ghost_pcd_features.shape[0], batch_size, demos_per_task,
                history_size, ghost_pcd_features.shape[-1], device=device
            )
            ghost_pcd_features[:, :, 0][:, padding_mask[:, 0]] = ghost_pcd_features1
            ghost_pcd_features[:, :, 1][:, padding_mask[:, 1]] = ghost_pcd_features2

        # Select ground-truth ghost points in the support set
        # TODO Currently we select the sampled ghost point closest to the ground-truth
        #  in the support set, but this might not be precise enough - maybe we should
        #  ensure we sample the ground-truth itself as a ghost point in the support set?
        l2_gt = ((gt_position_for_support - ghost_pcd) ** 2).sum(-1).sqrt()
        gt_indices = l2_gt.min(dim=-1).indices
        gt_ghost_pcd_features = ghost_pcd_features.view(
            num_points, -1, channels)[gt_indices.view(-1), torch.arange(batch_size * demos_per_task * history_size), :]
        gt_ghost_pcd_features = gt_ghost_pcd_features.view(batch_size, demos_per_task, history_size, channels)

        # Compute similarity scores from ghost points in the current scene to ground-truth
        # ghost points in the support set
        similarity_scores = einops.einsum(
            ghost_pcd_features, gt_ghost_pcd_features, "npts b s1 t c, b s2 t c -> b s1 t npts s2")

        ghost_pcd_mask = torch.zeros(
            batch_size, demos_per_task, history_size, num_points, device=ghost_pcd_features.device)
        for i in range(demos_per_task):
            if self.support_set == "self":
                # Select scores from the demo itself as the support set (for debugging)
                ghost_pcd_mask[:, i] = similarity_scores[:, i, :, :, i]
            elif self.support_set == "others":
                # Average scores from the other demos from the same task as the support set
                ghost_pcd_mask[:, i] = similarity_scores[:, i, :, :, torch.arange(demos_per_task) != i].mean(dim=-1)

        return ghost_pcd_mask[padding_mask]

    def _predict_action(self,
                        ghost_pcd_mask, ghost_pcd, ghost_pcd_features, total_timesteps,
                        fine_ghost_pcd_offsets=None):
        """Compute the predicted action (position, rotation, opening) from the predicted mask."""
        # Select top-scoring ghost point
        top_idx = torch.max(ghost_pcd_mask, dim=-1).indices
        position = ghost_pcd[torch.arange(total_timesteps), :, top_idx]

        # Add an offset regressed from the ghost point's position to the predicted position
        if fine_ghost_pcd_offsets is not None:
            position = position + fine_ghost_pcd_offsets[torch.arange(total_timesteps), :, top_idx]

        # Predict rotation and gripper opening
        ghost_pcd_features = einops.rearrange(ghost_pcd_features, "npts bst c -> bst npts c")
        features = ghost_pcd_features[torch.arange(total_timesteps), top_idx]

        pred = self.gripper_state_predictor(features)
        rotation = normalise_quat(pred[:, :4])
        gripper = torch.sigmoid(pred[:, 4:])

        return position, rotation, gripper
