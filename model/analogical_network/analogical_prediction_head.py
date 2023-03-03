import einops
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork

from model.utils.position_encodings import RotaryPositionEncoding3D
from model.utils.layers import RelativeCrossAttentionLayer, FeedforwardLayer
from model.utils.utils import normalise_quat, sample_ghost_points_uniform_cube, sample_ghost_points_uniform_sphere
from model.utils.resnet import load_resnet50
from model.utils.clip import load_clip


class AnalogicalPredictionHead(nn.Module):
    def __init__(self,
                 backbone="resnet",
                 image_size=(128, 128),
                 embedding_dim=60,
                 num_attn_heads=4,
                 num_ghost_point_cross_attn_layers=2,
                 rotation_parametrization="quat_from_query",
                 gripper_loc_bounds=None,
                 num_ghost_points=1000,
                 coarse_to_fine_sampling=True,
                 fine_sampling_ball_diameter=0.08,
                 separate_coarse_and_fine_layers=True,
                 regress_position_offset=True,
                 support_set="others",
                 use_instruction=False,
                 global_correspondence=False,
                 num_matching_cross_attn_layers=2):
        super().__init__()
        assert backbone in ["resnet", "clip"]
        assert image_size in [(128, 128), (256, 256)]
        assert rotation_parametrization in ["quat_from_top_ghost", "quat_from_query"]
        assert support_set in ["self", "others"]
        self.image_size = image_size
        self.rotation_parametrization = rotation_parametrization
        self.num_ghost_points = (num_ghost_points // 2) if coarse_to_fine_sampling else num_ghost_points
        self.coarse_to_fine_sampling = coarse_to_fine_sampling
        self.fine_sampling_ball_diameter = fine_sampling_ball_diameter
        self.gripper_loc_bounds = np.array(gripper_loc_bounds)
        self.regress_position_offset = regress_position_offset
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
        self.feature_pyramid = FeaturePyramidNetwork([64, 256, 512, 1024, 2048], embedding_dim)
        if self.image_size == (128, 128):
            # Coarse RGB features are the 2nd layer of the feature pyramid at 1/4 resolution (32x32)
            # Fine RGB features are the 1st layer of the feature pyramid at 1/2 resolution (64x64)
            self.coarse_feature_map = "res2"
            self.coarse_downscaling_factor = 4
            self.fine_feature_map = "res1"
            self.fine_downscaling_factor = 2
        elif self.image_size == (256, 256):
            # Coarse RGB features are the 3rd layer of the feature pyramid at 1/8 resolution (32x32)
            # Fine RGB features are the 1st layer of the feature pyramid at 1/2 resolution (128x128)
            self.coarse_feature_map = "res3"
            self.coarse_downscaling_factor = 8
            self.fine_feature_map = "res1"
            self.fine_downscaling_factor = 2

        # 3D positional embeddings
        self.pcd_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # Ghost points learnable initial features
        self.fine_ghost_points_embed = nn.Embedding(1, embedding_dim)
        if separate_coarse_and_fine_layers:
            self.coarse_ghost_points_embed = nn.Embedding(1, embedding_dim)
        else:
            self.coarse_ghost_points_embed = self.fine_ghost_points_embed

        # Current gripper learnable features
        self.curr_gripper_embed = nn.Embedding(1, embedding_dim)

        # Query learnable features
        self.query_embed = nn.Embedding(1, embedding_dim)

        # Ghost point cross-attention to visual features and current gripper position
        self.coarse_ghost_point_cross_attn_layers = nn.ModuleList()
        self.coarse_ghost_point_ffw_layers = nn.ModuleList()
        for _ in range(num_ghost_point_cross_attn_layers):
            self.coarse_ghost_point_cross_attn_layers.append(RelativeCrossAttentionLayer(embedding_dim, num_attn_heads))
            self.coarse_ghost_point_ffw_layers.append(FeedforwardLayer(embedding_dim, embedding_dim))
        if coarse_to_fine_sampling and separate_coarse_and_fine_layers:
            self.fine_ghost_point_cross_attn_layers = nn.ModuleList()
            self.fine_ghost_point_ffw_layers = nn.ModuleList()
            for _ in range(num_ghost_point_cross_attn_layers):
                self.fine_ghost_point_cross_attn_layers.append(RelativeCrossAttentionLayer(embedding_dim, num_attn_heads))
                self.fine_ghost_point_ffw_layers.append(FeedforwardLayer(embedding_dim, embedding_dim))
        else:
            self.fine_ghost_point_cross_attn_layers = self.coarse_ghost_point_cross_attn_layers
            self.fine_ghost_point_ffw_layers = self.coarse_ghost_point_ffw_layers

        # Ghost point matching self-attention and cross-attention with support set
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

        # Gripper rotation prediction
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
                visible_rgb, visible_pcd, curr_gripper, instruction,
                padding_mask, gt_action_for_support, gt_action_for_sampling=None):
        """
        Arguments:
            visible_rgb: (batch, 1 + support, history, num_cameras, 3, height, width) in [0, 1]
            visible_pcd: (batch, 1 + support, history, num_cameras, 3, height, width) in world coordinates
            curr_gripper: (batch, 1 + support, history, 3)
            instruction: (batch, 1 + support, history, max_instruction_length, 512)
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
        if gt_action_for_sampling is not None:
            gt_position_for_sampling = gt_action_for_sampling[padding_mask][:, :3].unsqueeze(-2).detach()
        else:
            gt_position_for_sampling = None
        gt_position_for_support = gt_action_for_support[:, :, :, :3].unsqueeze(-2).detach()
        total_timesteps = visible_rgb.shape[0]

        # Compute visual features at different scales and their positional embeddings
        (
            coarse_visible_rgb_features, coarse_visible_rgb_pos,
            fine_visible_rgb_features, fine_visible_rgb_pos, fine_visible_pcd
        ) = self._compute_visual_features(visible_rgb, visible_pcd, num_cameras)

        # Encode instruction
        if self.use_instruction:
            instruction_features = einops.rearrange(self.instruction_encoder(instruction), "bt l c -> l bt c")
            instruction_dummy_pos = torch.zeros(batch_size, instruction_features.shape[0], 3, device=device)
            instruction_dummy_pos = self.pcd_pe_layer(instruction_dummy_pos)
        else:
            instruction_features = None
            instruction_dummy_pos = None

        # Compute current gripper position features and positional embeddings
        curr_gripper_pos = self.pcd_pe_layer(curr_gripper.unsqueeze(1))
        curr_gripper_features = self.curr_gripper_embed.weight.repeat(total_timesteps, 1).unsqueeze(0)

        # Sample ghost points coarsely across the entire workspace
        coarse_ghost_pcd = self._sample_ghost_points(
            total_timesteps, device, ghost_point_type="coarse", anchor=gt_position_for_sampling)

        # Compute coarse ghost point features by attending to coarse visual features and
        # current gripper position
        coarse_ghost_pcd_context_features = einops.rearrange(
            coarse_visible_rgb_features, "bst ncam c h w -> (ncam h w) bst c")
        coarse_ghost_pcd_context_features = torch.cat(
            [coarse_ghost_pcd_context_features, curr_gripper_features], dim=0)
        coarse_ghost_pcd_context_pos = torch.cat([coarse_visible_rgb_pos, curr_gripper_pos], dim=1)
        if self.use_instruction:
            coarse_ghost_pcd_context_features = torch.cat(
                [coarse_ghost_pcd_context_features, instruction_features], dim=0)
            coarse_ghost_pcd_context_pos = torch.cat(
                [coarse_ghost_pcd_context_pos, instruction_dummy_pos], dim=1)
        coarse_ghost_pcd_features, coarse_ghost_pcd_pos = self._compute_ghost_point_features(
            coarse_ghost_pcd, coarse_ghost_pcd_context_features, coarse_ghost_pcd_context_pos,
            total_timesteps, ghost_point_type="coarse"
        )

        # Compute coarse ghost point similarity scores with the ground-truth ghost points
        # in the support set at the corresponding timestep
        coarse_ghost_pcd_mask = self._match_ghost_points(
            coarse_ghost_pcd_features, coarse_ghost_pcd_pos, coarse_ghost_pcd, gt_position_for_support,
            padding_mask, batch_size, demos_per_task, history_size, device
        )

        coarse_ghost_pcd = einops.rearrange(coarse_ghost_pcd, "bst npts c -> bst c npts")
        ghost_pcd = coarse_ghost_pcd
        ghost_pcd_mask = coarse_ghost_pcd_mask
        ghost_pcd_features = coarse_ghost_pcd_features

        if self.coarse_to_fine_sampling:
            top_idx = torch.max(coarse_ghost_pcd_mask, dim=-1).indices
            coarse_position = ghost_pcd[torch.arange(total_timesteps), :, top_idx].unsqueeze(1)

            (
                fine_ghost_pcd_mask, fine_ghost_pcd, fine_ghost_pcd_features, fine_ghost_pcd_offsets,
            ) = self._coarse_to_fine(
                coarse_position, coarse_ghost_pcd, coarse_ghost_pcd_pos, coarse_ghost_pcd_features,
                fine_visible_rgb_features, fine_visible_pcd, fine_visible_rgb_pos,
                curr_gripper_features, curr_gripper_pos,
                instruction_features, instruction_dummy_pos, padding_mask,
                batch_size, demos_per_task, history_size, num_cameras, total_timesteps, device,
                gt_position_for_support, gt_position_for_sampling
            )
            fine_ghost_pcd = einops.rearrange(fine_ghost_pcd, "bst npts c -> bst c npts")
            ghost_pcd = fine_ghost_pcd
            ghost_pcd_mask = fine_ghost_pcd_mask
            ghost_pcd_features = fine_ghost_pcd_features

            top_idx = torch.max(fine_ghost_pcd_mask, dim=-1).indices
            fine_position = fine_ghost_pcd[torch.arange(total_timesteps), :, top_idx].unsqueeze(1)

        # Predict the next gripper action (position, rotation, gripper opening)
        position, rotation, gripper = self._predict_action(
            ghost_pcd_mask, ghost_pcd, ghost_pcd_features, total_timesteps,
            fine_ghost_pcd_offsets if (self.coarse_to_fine_sampling and self.regress_position_offset) else None
        )

        return {
            # Action
            "position": position,
            "rotation": rotation,
            "gripper": gripper,
            # Auxiliary outputs used to compute the loss or for visualization
            "coarse_position": coarse_position if self.coarse_to_fine_sampling else None,
            "coarse_ghost_pcd_masks":  [coarse_ghost_pcd_mask],
            "coarse_ghost_pcd": coarse_ghost_pcd,
            "fine_position": fine_position if self.coarse_to_fine_sampling else None,
            "fine_ghost_pcd_masks": [fine_ghost_pcd_mask] if self.coarse_to_fine_sampling else None,
            "fine_ghost_pcd": fine_ghost_pcd if self.coarse_to_fine_sampling else None,
            "fine_ghost_pcd_offsets": fine_ghost_pcd_offsets if self.coarse_to_fine_sampling else None,
        }

    def _compute_visual_features(self, visible_rgb, visible_pcd, num_cameras):
        """Compute visual features at different scales and their positional embeddings."""
        # Pass each view independently through ResNet50 backbone
        visible_rgb = einops.rearrange(visible_rgb, "bst ncam c h w -> (bst ncam) c h w")
        visible_rgb = self.normalize(visible_rgb)
        visible_rgb_features = self.backbone(visible_rgb)

        # Pass visual features through feature pyramid network
        visible_rgb_features = self.feature_pyramid(visible_rgb_features)

        visible_pcd = einops.rearrange(visible_pcd, "bst ncam c h w -> (bst ncam) c h w")

        coarse_visible_rgb_features = visible_rgb_features[self.coarse_feature_map]
        coarse_visible_pcd = F.interpolate(
            visible_pcd, scale_factor=1. / self.coarse_downscaling_factor, mode='bilinear')
        coarse_visible_pcd = einops.rearrange(
            coarse_visible_pcd, "(bst ncam) c h w -> bst (ncam h w) c", ncam=num_cameras)
        coarse_visible_rgb_pos = self.pcd_pe_layer(coarse_visible_pcd)
        coarse_visible_rgb_features = einops.rearrange(
            coarse_visible_rgb_features, "(bst ncam) c h w -> bst ncam c h w", ncam=num_cameras)

        fine_visible_rgb_features = visible_rgb_features[self.fine_feature_map]
        fine_visible_pcd = F.interpolate(
            visible_pcd, scale_factor=1. / self.fine_downscaling_factor, mode='bilinear')
        fine_visible_pcd = einops.rearrange(
            fine_visible_pcd, "(bst ncam) c h w -> bst (ncam h w) c", ncam=num_cameras)
        fine_visible_rgb_pos = self.pcd_pe_layer(fine_visible_pcd)
        fine_visible_rgb_features = einops.rearrange(
            fine_visible_rgb_features, "(bst ncam) c h w -> bst ncam c h w", ncam=num_cameras)

        return (
            coarse_visible_rgb_features, coarse_visible_rgb_pos,
            fine_visible_rgb_features, fine_visible_rgb_pos, fine_visible_pcd
        )

    def _sample_ghost_points(self, batch_size, device, ghost_point_type, anchor=None):
        """Sample ghost points.

        If ghost_point_type is "coarse", sample points uniformly within the workspace
        bounds and one near the anchor if it is specified.

        If ghost_point_type is "fine", sample points uniformly within a local sphere
        of the workspace bounds centered around the anchor.
        """
        if ghost_point_type == "coarse":
            bounds = np.stack([self.gripper_loc_bounds for _ in range(batch_size)])
            uniform_pcd = np.stack([
                sample_ghost_points_uniform_cube(
                    bounds=bounds[i],
                    num_points=self.num_ghost_points
                )
                for i in range(batch_size)
            ])

        elif ghost_point_type == "fine":
            anchor_ = anchor[:, 0].cpu().numpy()
            bounds_min = np.clip(
                anchor_ - self.fine_sampling_ball_diameter / 2,
                a_min=self.gripper_loc_bounds[0], a_max=self.gripper_loc_bounds[1]
            )
            bounds_max = np.clip(
                anchor_ + self.fine_sampling_ball_diameter / 2,
                a_min=self.gripper_loc_bounds[0], a_max=self.gripper_loc_bounds[1]
            )
            bounds = np.stack([bounds_min, bounds_max], axis=1)
            uniform_pcd = np.stack([
                sample_ghost_points_uniform_sphere(
                    center=anchor_[i],
                    radius=self.fine_sampling_ball_diameter / 2,
                    bounds=bounds[i],
                    num_points=self.num_ghost_points
                )
                for i in range(batch_size)
            ])

        uniform_pcd = torch.from_numpy(uniform_pcd).float().to(device)

        if anchor is not None:
            # Sample a point near the anchor position as an additional ghost point
            offset = (torch.rand(batch_size, 1, 3, device=device) - 1 / 2) * self.fine_sampling_ball_diameter / 2
            anchor_pcd = anchor + offset
            ghost_pcd = torch.cat([uniform_pcd, anchor_pcd], dim=1)
        else:
            ghost_pcd = uniform_pcd

        return ghost_pcd

    def _compute_ghost_point_features(self, ghost_pcd, context_features, context_pos, batch_size, ghost_point_type):
        """
        Ghost points cross-attend to context features (visual features and current
        gripper position).
        """
        if ghost_point_type == "fine":
            embed = self.fine_ghost_points_embed
            attn_layers = self.fine_ghost_point_cross_attn_layers
            ffw_layers = self.fine_ghost_point_ffw_layers
        elif ghost_point_type == "coarse":
            embed = self.coarse_ghost_points_embed
            attn_layers = self.coarse_ghost_point_cross_attn_layers
            ffw_layers = self.coarse_ghost_point_ffw_layers

        # Initialize ghost point features and positional embeddings
        ghost_pcd_pos = self.pcd_pe_layer(ghost_pcd)
        num_ghost_points = ghost_pcd.shape[1]
        ghost_pcd_features = embed.weight.unsqueeze(0).repeat(num_ghost_points, batch_size, 1)

        # Ghost points cross-attend to visual features and current gripper position
        for i in range(len(attn_layers)):
            ghost_pcd_features, _ = attn_layers[i](
                query=ghost_pcd_features, value=context_features,
                query_pos=ghost_pcd_pos, value_pos=context_pos
            )
            ghost_pcd_features = ffw_layers[i](ghost_pcd_features)

        return ghost_pcd_features, ghost_pcd_pos

    def _match_ghost_points(self,
                            ghost_pcd_features, ghost_pcd_pos, ghost_pcd, gt_position_for_support,
                            padding_mask, batch_size, demos_per_task, history_size, device):
        """Compute ghost point similarity scores with the ground-truth ghost points
        in the support set at the corresponding timestep.
        """
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
            # TODO We assume there is a single demo in the support set for now
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
        #  in the support set, but maybe we should ensure we sample the ground-truth itself
        #  as a ghost point in the support set?
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
                        ghost_pcd_mask, ghost_pcd, ghost_pcd_features, batch_size,
                        fine_ghost_pcd_offsets=None):
        """Compute the predicted action (position, rotation, opening) from the predicted mask."""
        # Select top-scoring ghost point
        top_idx = torch.max(ghost_pcd_mask, dim=-1).indices
        position = ghost_pcd[torch.arange(batch_size), :, top_idx]

        # Add an offset regressed from the ghost point's position to the predicted position
        if fine_ghost_pcd_offsets is not None:
            position = position + fine_ghost_pcd_offsets[torch.arange(batch_size), :, top_idx]

        # Predict rotation and gripper opening
        if self.rotation_parametrization == "quat_from_top_ghost":
            ghost_pcd_features = einops.rearrange(ghost_pcd_features, "npts bst c -> bst npts c")
            features = ghost_pcd_features[torch.arange(batch_size), top_idx]
        elif self.rotation_parametrization == "quat_from_query":
            raise NotImplementedError

        pred = self.gripper_state_predictor(features)
        rotation = normalise_quat(pred[:, :4])
        gripper = torch.sigmoid(pred[:, 4:])

        return position, rotation, gripper

    def _coarse_to_fine(self,
                        coarse_position, coarse_ghost_pcd, coarse_ghost_pcd_pos, coarse_ghost_pcd_features,
                        fine_visible_rgb_features, fine_visible_pcd, fine_visible_rgb_pos,
                        curr_gripper_features, curr_gripper_pos,
                        instruction_features, instruction_dummy_pos, padding_mask,
                        batch_size, demos_per_task, history_size, num_cameras, total_timesteps, device,
                        gt_position_for_support, gt_position_for_sampling):
        """
        Refine the predicted position by sampling fine ghost points that attend to local
        RGB features.
        """
        # Sample ghost points finely near the top scoring coarse point (or the ground truth
        # position at training time)
        fine_ghost_pcd = self._sample_ghost_points(
            total_timesteps, device, ghost_point_type="fine",
            anchor=gt_position_for_sampling if gt_position_for_sampling is not None else coarse_position)

        # Select local fine RGB features
        l2_pred_pos = ((coarse_position - fine_visible_pcd) ** 2).sum(-1).sqrt()
        indices = l2_pred_pos.topk(k=32 * 32 * num_cameras, dim=-1, largest=False).indices

        local_fine_visible_rgb_features = einops.rearrange(
            fine_visible_rgb_features, "bst ncam c h w -> bst (ncam h w) c")
        local_fine_visible_rgb_features = torch.stack([
            f[i] for (f, i) in zip(local_fine_visible_rgb_features, indices)])
        local_fine_visible_rgb_pos = torch.stack([
            f[i] for (f, i) in zip(fine_visible_rgb_pos, indices)])

        # Compute fine ghost point features by attending to the local fine RGB features
        fine_ghost_pcd_context_features = einops.rearrange(
            local_fine_visible_rgb_features, "bst npts c -> npts bst c")
        fine_ghost_pcd_context_features = torch.cat(
            [fine_ghost_pcd_context_features, curr_gripper_features], dim=0)
        fine_ghost_pcd_context_pos = torch.cat(
            [local_fine_visible_rgb_pos, curr_gripper_pos], dim=1)
        if self.use_instruction:
            fine_ghost_pcd_context_features = torch.cat(
                [fine_ghost_pcd_context_features, instruction_features], dim=0)
            fine_ghost_pcd_context_pos = torch.cat(
                [fine_ghost_pcd_context_pos, instruction_dummy_pos], dim=1)
        (
            fine_ghost_pcd_features,
            fine_ghost_pcd_pos
        ) = self._compute_ghost_point_features(
            fine_ghost_pcd, fine_ghost_pcd_context_features, fine_ghost_pcd_context_pos,
            total_timesteps, ghost_point_type="fine"
        )

        # Compute all ghost point (coarse + fine) similarity scores with the ground-truth ghost points
        # in the support set at the corresponding timestep
        fine_ghost_pcd = torch.cat(
            [einops.rearrange(coarse_ghost_pcd, "bst c npts -> bst npts c"), fine_ghost_pcd], dim=-2)
        fine_ghost_pcd_features = torch.cat([coarse_ghost_pcd_features, fine_ghost_pcd_features], dim=0)
        fine_ghost_pcd_pos = torch.cat([coarse_ghost_pcd_pos, fine_ghost_pcd_pos], dim=1)
        fine_ghost_pcd_mask = self._match_ghost_points(
            fine_ghost_pcd_features, fine_ghost_pcd_pos, fine_ghost_pcd, gt_position_for_support,
            padding_mask, batch_size, demos_per_task, history_size, device
        )

        # Regress an offset from the ghost point's position to the predicted position
        if self.regress_position_offset:
            fine_ghost_pcd_offsets = self.ghost_point_offset_predictor(fine_ghost_pcd_features)
            fine_ghost_pcd_offsets = einops.rearrange(fine_ghost_pcd_offsets, "npts bst c -> bst c npts")
        else:
            fine_ghost_pcd_offsets = None

        return fine_ghost_pcd_mask, fine_ghost_pcd, fine_ghost_pcd_features, fine_ghost_pcd_offsets
