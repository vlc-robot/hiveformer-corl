import einops
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.ops import FeaturePyramidNetwork

from model.utils.position_encodings import RotaryPositionEncoding3D
from model.utils.layers import RelativeCrossAttentionLayer, FeedforwardLayer
from model.utils.utils import normalise_quat, sample_ghost_points_uniform_cube, sample_ghost_points_uniform_sphere
from model.utils.resnet import resnet50


class PredictionHead(nn.Module):
    def __init__(self,
                 image_size=(128, 128),
                 embedding_dim=60,
                 num_attn_heads=4,
                 num_ghost_point_cross_attn_layers=2,
                 num_query_cross_attn_layers=2,
                 rotation_parametrization="quat_from_query",
                 gripper_loc_bounds=None,
                 num_ghost_points=1000,
                 coarse_to_fine_sampling=True,
                 fine_sampling_ball_diameter=0.08,
                 separate_coarse_and_fine_layers=True,
                 regress_position_offset=True,
                 visualize_rgb_attn=False):
        super().__init__()
        assert image_size in [(128, 128), (256, 256)]
        self.image_size = image_size
        assert rotation_parametrization in ["quat_from_top_ghost", "quat_from_query"]
        self.rotation_parametrization = rotation_parametrization
        self.num_ghost_points = (num_ghost_points // 2) if coarse_to_fine_sampling else num_ghost_points
        self.coarse_to_fine_sampling = coarse_to_fine_sampling
        self.fine_sampling_ball_diameter = fine_sampling_ball_diameter
        self.gripper_loc_bounds = np.array(gripper_loc_bounds)
        self.regress_position_offset = regress_position_offset
        self.visualize_rgb_attn = visualize_rgb_attn

        # Frozen ResNet50 backbone
        self.backbone = resnet50()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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

        # Query cross-attention to visual features, ghost points, and the current gripper position
        self.coarse_query_cross_attn_layers = nn.ModuleList()
        self.coarse_query_ffw_layers = nn.ModuleList()
        for _ in range(num_query_cross_attn_layers):
            self.coarse_query_cross_attn_layers.append(RelativeCrossAttentionLayer(embedding_dim, num_attn_heads))
            self.coarse_query_ffw_layers.append(FeedforwardLayer(embedding_dim, embedding_dim))
        if coarse_to_fine_sampling and separate_coarse_and_fine_layers:
            self.fine_query_cross_attn_layers = nn.ModuleList()
            self.fine_query_ffw_layers = nn.ModuleList()
            for _ in range(num_query_cross_attn_layers):
                self.fine_query_cross_attn_layers.append(RelativeCrossAttentionLayer(embedding_dim, num_attn_heads))
                self.fine_query_ffw_layers.append(FeedforwardLayer(embedding_dim, embedding_dim))
        else:
            self.fine_query_cross_attn_layers = self.coarse_query_cross_attn_layers
            self.fine_query_ffw_layers = self.coarse_query_ffw_layers

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

    def forward(self, visible_rgb, visible_pcd, curr_gripper, instruction, gt_action=None):
        """
        Arguments:
            visible_rgb: (batch x history, num_cameras, 3, height, width) in [0, 1]
            visible_pcd: (batch x history, num_cameras, 3, height, width) in world coordinates
            curr_gripper: (batch x history, 3)
            instruction: (batch x history, max_instruction_length, 512)
            gt_action: (batch x history, 8) in world coordinates
        """
        batch_size, num_cameras, _, height, width = visible_rgb.shape
        device = visible_rgb.device
        if gt_action is not None:
            gt_position = gt_action[:, :3].unsqueeze(1).detach()
        else:
            gt_position = None

        # Compute visual features at different scales and their positional embeddings
        (
            coarse_visible_rgb_features, coarse_visible_rgb_pos,
            fine_visible_rgb_features, fine_visible_rgb_pos, fine_visible_pcd
        ) = self._compute_visual_features(visible_rgb, visible_pcd, device, num_cameras)

        # Compute current gripper position features and positional embeddings
        curr_gripper_pos = self.pcd_pe_layer(curr_gripper.unsqueeze(1))
        curr_gripper_features = self.curr_gripper_embed.weight.repeat(batch_size, 1).unsqueeze(0)

        # Sample ghost points coarsely across the entire workspace
        coarse_ghost_pcd = self._sample_ghost_points(
            batch_size, device, ghost_point_type="coarse", anchor=gt_position)

        # Compute coarse ghost point features and their positional embeddings by attending to
        # coarse visual features and current gripper position
        coarse_ghost_pcd_context_features = einops.rearrange(
            coarse_visible_rgb_features, "b ncam c h w -> (ncam h w) b c")
        coarse_ghost_pcd_context_features = torch.cat(
            [coarse_ghost_pcd_context_features, curr_gripper_features], dim=0)
        coarse_ghost_pcd_context_pos = torch.cat([coarse_visible_rgb_pos, curr_gripper_pos], dim=1)
        (
            coarse_ghost_pcd_features,
            coarse_ghost_pcd_pos,
            coarse_ghost_pcd_to_visible_rgb_attn
        ) = self._compute_ghost_point_features(
            coarse_ghost_pcd, coarse_ghost_pcd_context_features, coarse_ghost_pcd_context_pos, batch_size,
            ghost_point_type="coarse"
        )

        # Initialize query features
        query_features = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)

        # Contextualize the query and predict masks over coarse ghost points
        # Given the query is not localized yet, we don't use positional embeddings
        coarse_query_context_features = torch.cat([coarse_ghost_pcd_context_features, coarse_ghost_pcd_features], dim=0)
        coarse_query_context_pos = torch.cat([coarse_ghost_pcd_context_pos, coarse_ghost_pcd_pos], dim=1)
        query_features, coarse_ghost_pcd_masks, coarse_visible_rgb_mask = self._decode_mask(
            coarse_ghost_pcd_features, coarse_ghost_pcd_to_visible_rgb_attn, height, width,
            query_features, coarse_query_context_features, query_pos=None, context_pos=None,
            ghost_point_type="coarse"
        )

        coarse_ghost_pcd = einops.rearrange(coarse_ghost_pcd, "b npts c -> b c npts")
        ghost_pcd = coarse_ghost_pcd
        ghost_pcd_masks = coarse_ghost_pcd_masks
        ghost_pcd_features = coarse_ghost_pcd_features

        if self.coarse_to_fine_sampling:
            top_idx = torch.max(coarse_ghost_pcd_masks[-1], dim=-1).indices
            coarse_position = coarse_ghost_pcd[torch.arange(batch_size), :, top_idx].unsqueeze(1)

            (
                fine_visible_rgb_mask, fine_ghost_pcd_masks, fine_ghost_pcd,
                fine_ghost_pcd_features, fine_ghost_pcd_offsets,
                query_features
            ) = self._coarse_to_fine(
                coarse_position, query_features, coarse_ghost_pcd_features,
                fine_visible_rgb_features, fine_visible_pcd, fine_visible_rgb_pos,
                curr_gripper_features, curr_gripper_pos,
                batch_size, num_cameras, height, width, device, gt_position
            )
            fine_ghost_pcd = torch.cat([
                coarse_ghost_pcd, einops.rearrange(fine_ghost_pcd, "b npts c -> b c npts")], dim=-1)
            ghost_pcd = fine_ghost_pcd
            ghost_pcd_masks = fine_ghost_pcd_masks
            ghost_pcd_features = fine_ghost_pcd_features

            top_idx = torch.max(fine_ghost_pcd_masks[-1], dim=-1).indices
            fine_position = fine_ghost_pcd[torch.arange(batch_size), :, top_idx].unsqueeze(1)

        # Predict the next gripper action (position, rotation, gripper opening)
        position, rotation, gripper = self._predict_action(
            ghost_pcd_masks[-1], ghost_pcd, ghost_pcd_features, query_features, batch_size,
            fine_ghost_pcd_offsets if (self.coarse_to_fine_sampling and self.regress_position_offset) else None
        )

        return {
            # Action
            "position": position,
            "rotation": rotation,
            "gripper": gripper,
            # Auxiliary outputs used to compute the loss or for visualization
            "coarse_position": coarse_position if self.coarse_to_fine_sampling else None,
            "coarse_visible_rgb_mask": coarse_visible_rgb_mask,
            "coarse_ghost_pcd_masks":  coarse_ghost_pcd_masks,
            "coarse_ghost_pcd": coarse_ghost_pcd,
            "coarse_ghost_pcd_features": coarse_ghost_pcd_features,
            "fine_position": fine_position if self.coarse_to_fine_sampling else None,
            "fine_visible_rgb_mask": fine_visible_rgb_mask if self.coarse_to_fine_sampling else None,
            "fine_ghost_pcd_masks": fine_ghost_pcd_masks if self.coarse_to_fine_sampling else None,
            "fine_ghost_pcd": fine_ghost_pcd if self.coarse_to_fine_sampling else None,
            "fine_ghost_pcd_features": fine_ghost_pcd_features if self.coarse_to_fine_sampling else None,
            "fine_ghost_pcd_offsets": fine_ghost_pcd_offsets if self.coarse_to_fine_sampling else None,
        }

    def _compute_visual_features(self, visible_rgb, visible_pcd, device, num_cameras):
        """Compute visual features at different scales and their positional embeddings."""
        # Pass each view independently through ResNet50 backbone
        visible_rgb = einops.rearrange(visible_rgb, "b ncam c h w -> (b ncam) c h w")
        visible_rgb = self.normalize(visible_rgb)
        visible_rgb_features = self.backbone(visible_rgb)

        # Pass visual features through feature pyramid network
        visible_rgb_features = self.feature_pyramid(visible_rgb_features)

        visible_pcd = einops.rearrange(visible_pcd, "b ncam c h w -> (b ncam) c h w")

        coarse_visible_rgb_features = visible_rgb_features[self.coarse_feature_map]
        coarse_visible_pcd = F.interpolate(
            visible_pcd, scale_factor=1. / self.coarse_downscaling_factor, mode='bilinear')
        coarse_visible_pcd = einops.rearrange(
            coarse_visible_pcd, "(b ncam) c h w -> b (ncam h w) c", ncam=num_cameras)
        coarse_visible_rgb_pos = self.pcd_pe_layer(coarse_visible_pcd)
        coarse_visible_rgb_features = einops.rearrange(
            coarse_visible_rgb_features, "(b ncam) c h w -> b ncam c h w", ncam=num_cameras)

        fine_visible_rgb_features = visible_rgb_features[self.fine_feature_map]
        fine_visible_pcd = F.interpolate(
            visible_pcd, scale_factor=1. / self.fine_downscaling_factor, mode='bilinear')
        fine_visible_pcd = einops.rearrange(
            fine_visible_pcd, "(b ncam) c h w -> b (ncam h w) c", ncam=num_cameras)
        fine_visible_rgb_pos = self.pcd_pe_layer(fine_visible_pcd)
        fine_visible_rgb_features = einops.rearrange(
            fine_visible_rgb_features, "(b ncam) c h w -> b ncam c h w", ncam=num_cameras)

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
            ghost_pcd_features, attn = attn_layers[i](
                query=ghost_pcd_features, value=context_features,
                query_pos=ghost_pcd_pos, value_pos=context_pos
            )
            ghost_pcd_features = ffw_layers[i](ghost_pcd_features)

        if self.visualize_rgb_attn:
            # Extract the attention weights over visual features for each ghost point for visualization
            ghost_pcd_to_visible_rgb_attn = attn[:, :, :-1]  # Last dim is gripper position
        else:
            ghost_pcd_to_visible_rgb_attn = None

        return ghost_pcd_features, ghost_pcd_pos, ghost_pcd_to_visible_rgb_attn

    def _decode_mask(self,
                     ghost_pcd_features, ghost_pcd_to_visible_rgb_attn,
                     rgb_height, rgb_width,
                     query_features, context_features,
                     query_pos, context_pos,
                     ghost_point_type):
        """
        The query cross-attends to context features (visual features, ghost points, and the
        current gripper position) then decodes a mask over ghost points (used to predict
        the gripper position) and over visual features (for visualization only).
        """
        if ghost_point_type == "fine":
            attn_layers = self.fine_query_cross_attn_layers
            ffw_layers = self.fine_query_ffw_layers
            h, w = rgb_height // self.fine_downscaling_factor, rgb_width // self.fine_downscaling_factor
        elif ghost_point_type == "coarse":
            attn_layers = self.coarse_query_cross_attn_layers
            ffw_layers = self.coarse_query_ffw_layers
            h, w = rgb_height // self.coarse_downscaling_factor, rgb_width // self.coarse_downscaling_factor

        ghost_pcd_masks = []

        for i in range(len(attn_layers)):
            query_features, _ = attn_layers[i](
                query=query_features, value=context_features,
                query_pos=query_pos, value_pos=context_pos
            )
            query_features = ffw_layers[i](query_features)

            ghost_pcd_masks.append(einops.einsum(
                query_features.squeeze(0), ghost_pcd_features, "b c, npts b c -> b npts"))

        # Extract attention from top ghost point to visual features for visualization
        if ghost_pcd_to_visible_rgb_attn is not None:
            top_idx = torch.max(ghost_pcd_masks[-1], dim=-1).indices
            visible_rgb_mask = ghost_pcd_to_visible_rgb_attn[torch.arange(len(top_idx)), top_idx]
            visible_rgb_mask = einops.rearrange(visible_rgb_mask, "b (ncam h w) -> b ncam h w", h=h, w=w)
            visible_rgb_mask = F.interpolate(visible_rgb_mask, size=(rgb_height, rgb_width), mode="nearest")
        else:
            visible_rgb_mask = None

        return query_features, ghost_pcd_masks, visible_rgb_mask

    def _predict_action(self,
                        ghost_pcd_mask, ghost_pcd, ghost_pcd_features, query_features, batch_size,
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
            ghost_pcd_features = einops.rearrange(ghost_pcd_features, "npts b c -> b npts c")
            features = ghost_pcd_features[torch.arange(batch_size), top_idx]
        elif self.rotation_parametrization == "quat_from_query":
            features = query_features.squeeze(0)

        pred = self.gripper_state_predictor(features)
        rotation = normalise_quat(pred[:, :4])
        gripper = torch.sigmoid(pred[:, 4:])

        return position, rotation, gripper

    def _coarse_to_fine(self,
                        coarse_position, query_features, coarse_ghost_pcd_features,
                        fine_visible_rgb_features, fine_visible_pcd, fine_visible_rgb_pos,
                        curr_gripper_features, curr_gripper_pos,
                        batch_size, num_cameras, height, width, device, gt_position):
        """
        Refine the predicted position by sampling fine ghost points that attend to local
        RGB features.
        """
        # Sample ghost points finely near the top scoring coarse point (or the ground truth
        # position at training time)
        fine_ghost_pcd = self._sample_ghost_points(
            batch_size, device, ghost_point_type="fine",
            anchor=gt_position if gt_position is not None else coarse_position)

        # Select local fine RGB features
        l2_pred_pos = ((coarse_position - fine_visible_pcd) ** 2).sum(-1).sqrt()
        indices = l2_pred_pos.topk(k=32 * 32 * num_cameras, dim=-1, largest=False).indices

        local_fine_visible_rgb_features = einops.rearrange(
            fine_visible_rgb_features, "b ncam c h w -> b (ncam h w) c")
        local_fine_visible_rgb_features = torch.stack([
            f[i] for (f, i) in zip(local_fine_visible_rgb_features, indices)])
        local_fine_visible_rgb_pos = torch.stack([
            f[i] for (f, i) in zip(fine_visible_rgb_pos, indices)])

        # Compute fine ghost point features by attending to the local fine RGB features
        fine_ghost_pcd_context_features = einops.rearrange(
            local_fine_visible_rgb_features, "b npts c -> npts b c")
        fine_ghost_pcd_context_features = torch.cat(
            [fine_ghost_pcd_context_features, curr_gripper_features], dim=0)
        fine_ghost_pcd_context_pos = torch.cat(
            [local_fine_visible_rgb_pos, curr_gripper_pos], dim=1)
        (
            fine_ghost_pcd_features,
            fine_ghost_pcd_pos,
            fine_ghost_pcd_to_visible_rgb_attn_
        ) = self._compute_ghost_point_features(
            fine_ghost_pcd, fine_ghost_pcd_context_features, fine_ghost_pcd_context_pos, batch_size,
            ghost_point_type="fine"
        )

        if self.visualize_rgb_attn:
            # TODO This currently has a lot of memory overhead for visualization, optimize it
            num_coarse_pts, num_fine_pts = coarse_ghost_pcd_features.shape[0], fine_ghost_pcd_features.shape[0]
            h, w = fine_visible_rgb_features.shape[-2:]
            fine_ghost_pcd_to_visible_rgb_attn = torch.zeros(
                batch_size, num_coarse_pts + num_fine_pts, num_cameras * h * w, device=device)
            for i, f in enumerate(indices):
                fine_ghost_pcd_to_visible_rgb_attn[i, num_coarse_pts:, f] = fine_ghost_pcd_to_visible_rgb_attn_[i]
        else:
            fine_ghost_pcd_to_visible_rgb_attn = None

        # Contextualize the query and predict masks over all (coarse + fine) ghost points
        # Now that the query is localized, we use positional embeddings
        query_pos = self.pcd_pe_layer(coarse_position)
        fine_query_context_features = torch.cat([fine_ghost_pcd_context_features, fine_ghost_pcd_features], dim=0)
        fine_query_context_pos = torch.cat([fine_ghost_pcd_context_pos, fine_ghost_pcd_pos], dim=1)
        fine_ghost_pcd_features = torch.cat([coarse_ghost_pcd_features, fine_ghost_pcd_features], dim=0)

        query_features, fine_ghost_pcd_masks, fine_visible_rgb_mask = self._decode_mask(
            fine_ghost_pcd_features, fine_ghost_pcd_to_visible_rgb_attn, height, width,
            query_features, fine_query_context_features, query_pos=query_pos, context_pos=fine_query_context_pos,
            ghost_point_type="fine"
        )

        # Regress an offset from the ghost point's position to the predicted position
        if self.regress_position_offset:
            fine_ghost_pcd_offsets = self.ghost_point_offset_predictor(fine_ghost_pcd_features)
            fine_ghost_pcd_offsets = einops.rearrange(fine_ghost_pcd_offsets, "npts b c -> b c npts")
        else:
            fine_ghost_pcd_offsets = None

        return (
            fine_visible_rgb_mask, fine_ghost_pcd_masks, fine_ghost_pcd,
            fine_ghost_pcd_features, fine_ghost_pcd_offsets,
            query_features
        )
