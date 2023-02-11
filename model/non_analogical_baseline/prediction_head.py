import einops
from pathlib import Path
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from torchvision.ops import FeaturePyramidNetwork

from .position_encodings import RotaryPositionEncoding3D
from .layers import RelativeCrossAttentionLayer, FeedforwardLayer
from .utils import normalise_quat, sample_ghost_points_randomly


class PredictionHead(nn.Module):
    def __init__(self,
                 image_size=(128, 128),
                 embedding_dim=60,
                 num_attn_heads=4,
                 num_ghost_point_cross_attn_layers=2,
                 num_query_cross_attn_layers=2,
                 loss="ce",
                 rotation_pooling_gaussian_spread=0.01,
                 use_ground_truth_position_for_sampling=False,
                 gripper_loc_bounds=None,
                 num_ghost_points=1000,
                 coarse_to_fine_sampling=True,
                 fine_sampling_cube_size=0.05):
        super().__init__()
        assert image_size in [(128, 128), (256, 256)]
        self.image_size = image_size
        self.loss = loss
        self.rotation_pooling_gaussian_spread = rotation_pooling_gaussian_spread
        self.use_ground_truth_position_for_sampling = use_ground_truth_position_for_sampling
        self.num_ghost_points = (num_ghost_points // 2) if coarse_to_fine_sampling else num_ghost_points
        self.coarse_to_fine_sampling = coarse_to_fine_sampling
        self.fine_sampling_cube_size = fine_sampling_cube_size
        self.gripper_loc_bounds = np.array(gripper_loc_bounds)

        # Frozen ResNet50 backbone
        cfg = get_cfg()
        cfg.merge_from_file(str(Path(__file__).resolve().parent / "resnet50.yaml"))
        model = build_model(cfg)
        self.backbone = model.backbone
        self.pixel_mean = model.pixel_mean.to()
        self.pixel_std = model.pixel_std
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Semantic visual features at different scales
        self.feature_pyramid = FeaturePyramidNetwork([256, 512, 1024, 2048], embedding_dim)

        # 3D positional embeddings
        self.pcd_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # Ghost points learnable initial features
        self.ghost_points_embed = nn.Embedding(1, embedding_dim)

        # Current gripper learnable features
        self.curr_gripper_embed = nn.Embedding(1, embedding_dim)

        # Query learnable features
        self.query_embed = nn.Embedding(1, embedding_dim)

        # Ghost point cross-attention to visual features and current gripper position
        self.ghost_point_cross_attn_layers = nn.ModuleList()
        self.ghost_point_ffw_layers = nn.ModuleList()
        for _ in range(num_ghost_point_cross_attn_layers):
            self.ghost_point_cross_attn_layers.append(RelativeCrossAttentionLayer(embedding_dim, num_attn_heads))
            self.ghost_point_ffw_layers.append(FeedforwardLayer(embedding_dim, embedding_dim))

        # Query cross-attention to visual features, ghost points, and the current gripper position
        self.query_cross_attn_layers = nn.ModuleList()
        self.query_ffw_layers = nn.ModuleList()
        for _ in range(num_query_cross_attn_layers):
            self.query_cross_attn_layers.append(RelativeCrossAttentionLayer(embedding_dim, num_attn_heads))
            self.query_ffw_layers.append(FeedforwardLayer(embedding_dim, embedding_dim))

        # Gripper rotation prediction
        self.gripper_state_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 4 + 1)
        )

    def forward(self, visible_rgb, visible_pcd, curr_gripper, gt_action=None):
        """
        Arguments:
            visible_rgb: (batch x history, num_cameras, 3, height, width) in [0, 1]
            visible_pcd: (batch x history, num_cameras, 3, height, width) in world coordinates
            curr_gripper: (batch x history, 3)
            gt_action: (batch x history, 8) in world coordinates

        Returns:
            next_gripper: (batch x history, 3)
        """
        batch_size, num_cameras, _, height, width = visible_rgb.shape
        device = visible_rgb.device

        # Compute visual features at different scales and their positional embeddings
        (
            coarse_visible_rgb_features, coarse_visible_rgb_pos,
            fine_visible_rgb_features, fine_visible_rgb_pos
        ) = self._compute_visual_features(visible_rgb, visible_pcd, device, num_cameras)

        # Compute current gripper position features and positional embeddings
        curr_gripper_pos = self.pcd_pe_layer(curr_gripper.unsqueeze(1))
        curr_gripper_features = self.curr_gripper_embed.weight.repeat(batch_size, 1).unsqueeze(0)

        # Sample ghost points coarsely across the entire workspace
        coarse_ghost_pcd = self._sample_ghost_points(
            np.stack([self.gripper_loc_bounds for _ in range(batch_size)]),
            batch_size, device, gt_action
        )

        # Compute coarse ghost point features and their positional embeddings by attending to
        # coarse visual features and current gripper position
        coarse_ghost_pcd_context_features = einops.rearrange(
            coarse_visible_rgb_features, "b ncam c h w -> (ncam h w) b c")
        coarse_ghost_pcd_context_features = torch.cat(
            [coarse_ghost_pcd_context_features, curr_gripper_features], dim=0)
        coarse_ghost_pcd_context_pos = torch.cat([coarse_visible_rgb_pos, curr_gripper_pos], dim=1)
        coarse_ghost_pcd_features, coarse_ghost_pcd_pos = self._compute_ghost_point_features(
            coarse_ghost_pcd, coarse_ghost_pcd_context_features, coarse_ghost_pcd_context_pos, batch_size)

        # Initialize query features
        query_features = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)

        # Contextualize the query and predict masks over coarse ghost points
        # Given the query is not localized yet, we don't use positional embeddings
        coarse_query_context_features = torch.cat([coarse_ghost_pcd_context_features, coarse_ghost_pcd_features], dim=0)
        coarse_query_context_pos = torch.cat([coarse_ghost_pcd_context_pos, coarse_ghost_pcd_pos], dim=1)
        query_features, coarse_ghost_pcd_masks, coarse_visible_rgb_mask = self._decode_mask(
            coarse_visible_rgb_features, coarse_ghost_pcd_features, height, width,
            query_features, coarse_query_context_features, query_pos=None, context_pos=None
        )

        ghost_pcd = einops.rearrange(coarse_ghost_pcd, "b npts c -> b c npts")
        ghost_pcd_masks = coarse_ghost_pcd_masks
        ghost_pcd_features = coarse_ghost_pcd_features
        visible_rgb_mask = coarse_visible_rgb_mask

        # if self.coarse_to_fine_sampling:
        #     # Sample ghost points finely near the top scoring point
        #     top_idx = torch.max(all_masks[-1], dim=-1).indices
        #     position = all_pcd[torch.arange(batch_size), :, top_idx]
        #     position_ = position.cpu().numpy()
        #     bounds_min = np.clip(
        #         position_ - self.fine_sampling_cube_size / 2,
        #         a_min=self.gripper_loc_bounds[0], a_max=self.gripper_loc_bounds[1]
        #     )
        #     bounds_max = np.clip(
        #         position_ + self.fine_sampling_cube_size / 2,
        #         a_min=self.gripper_loc_bounds[0], a_max=self.gripper_loc_bounds[1]
        #     )
        #     bounds = np.stack([bounds_min, bounds_max], axis=1)
        #     fine_ghost_pcd = self._sample_ghost_points(bounds, batch_size, device, gt_action)
        #
        #     # Compute fine ghost point features
        #     # TODO fine_ghost_pcd_context_features
        #     fine_ghost_pcd_features, fine_ghost_pcd_pos = self._compute_ghost_point_features(
        #         fine_ghost_pcd, coarse_ghost_pcd_context_features, coarse_ghost_pcd_context_pos, batch_size)
        #
        #     # Contextualize the query and predict masks over fine visual features and all ghost points
        #     # Now that the query is localized, we use positional embeddings
        #     query_pos = self.pcd_pe_layer(position.unsqueeze(1))
        #     # TODO fine_query_context_features, fine_query_context_pos
        #     fine_query_context_features = torch.cat([coarse_query_context_features, fine_ghost_pcd_features], dim=0)
        #     fine_query_context_pos = torch.cat([coarse_query_context_pos, fine_ghost_pcd_pos], dim=1)
        #     fine_ghost_pcd_features = torch.cat([coarse_ghost_pcd_features, fine_ghost_pcd_features], dim=0)
        #     query_features, visible_rgb_masks, ghost_pcd_masks, all_masks = self._decode_mask(
        #         fine_visible_rgb_features, fine_ghost_pcd_features, height, width,
        #         query_features, fine_query_context_features, query_pos=query_pos, context_pos=fine_query_context_pos
        #     )
        #
        #     all_pcd = torch.cat([all_pcd, einops.rearrange(fine_ghost_pcd, "b npts c -> b c npts")], dim=-1)

        # Predict the next gripper action (position, rotation, gripper opening)
        position, rotation, gripper = self._predict_action(
            ghost_pcd_masks[-1], ghost_pcd, ghost_pcd_features, batch_size)

        return {
            # Action
            "position": position,
            "rotation": rotation,
            "gripper": gripper,
            # Auxiliary outputs used to compute the loss or for visualization
            "visible_rgb_mask": visible_rgb_mask,
            "ghost_pcd_masks":  ghost_pcd_masks,
            "ghost_pcd": ghost_pcd,
            "ghost_pcd_features": ghost_pcd_features,
        }

    def _compute_visual_features(self, visible_rgb, visible_pcd, device, num_cameras):
        """Compute visual features at different scales and their positional embeddings."""
        # Pass each view independently through ResNet50 backbone
        visible_rgb = einops.rearrange(visible_rgb, "b ncam c h w -> (b ncam) c h w")
        self.pixel_mean = self.pixel_mean.to(device)
        self.pixel_std = self.pixel_std.to(device)
        visible_rgb = (visible_rgb - self.pixel_mean) / self.pixel_std
        visible_rgb_features = self.backbone(visible_rgb)

        # Pass visual features through feature pyramid network
        visible_rgb_features = self.feature_pyramid(visible_rgb_features)

        visible_pcd = einops.rearrange(visible_pcd, "b ncam c h w -> (b ncam) c h w")

        if self.image_size == (128, 128):
            # Both coarse and fine RGB features are the first layer of the feature pyramid
            # at 1/4 resolution (32x32)
            coarse_visible_rgb_features = fine_visible_rgb_features = visible_rgb_features["res2"]
            visible_pcd = F.interpolate(visible_pcd, scale_factor=1. / 4, mode='bilinear')
            visible_pcd = einops.rearrange(visible_pcd, "(b ncam) c h w -> b (ncam h w) c", ncam=num_cameras)
            coarse_visible_rgb_pos = fine_visible_rgb_pos = self.pcd_pe_layer(visible_pcd)

        elif self.image_size == (256, 256):
            # Coarse RGB features are the second layer of the feature pyramid at 1/8 resolution (32x32)
            coarse_visible_rgb_features = visible_rgb_features["res3"]
            coarse_visible_pcd = F.interpolate(visible_pcd, scale_factor=1. / 8, mode='bilinear')
            coarse_visible_pcd = einops.rearrange(
                coarse_visible_pcd, "(b ncam) c h w -> b (ncam h w) c", ncam=num_cameras)
            coarse_visible_rgb_pos = self.pcd_pe_layer(coarse_visible_pcd)

            # Fine RGB features are the first layer of the feature pyramid at 1/4 resolution (64x64)
            fine_visible_rgb_features = visible_rgb_features["res2"]
            fine_visible_pcd = F.interpolate(visible_pcd, scale_factor=1. / 4, mode='bilinear')
            fine_visible_pcd = einops.rearrange(
                fine_visible_pcd, "(b ncam) c h w -> b (ncam h w) c", ncam=num_cameras)
            fine_visible_rgb_pos = self.pcd_pe_layer(fine_visible_pcd)

        coarse_visible_rgb_features = einops.rearrange(
            coarse_visible_rgb_features, "(b ncam) c h w -> b ncam c h w", ncam=num_cameras)
        fine_visible_rgb_features = einops.rearrange(
            fine_visible_rgb_features, "(b ncam) c h w -> b ncam c h w", ncam=num_cameras)

        return (
            coarse_visible_rgb_features, coarse_visible_rgb_pos,
            fine_visible_rgb_features, fine_visible_rgb_pos
        )

    def _sample_ghost_points(self, bounds, batch_size, device, gt_action=None):
        """Sample ghost points uniformly within the bounds."""
        uniform_pcd = np.stack([
            sample_ghost_points_randomly(bounds[i], num_points=self.num_ghost_points)
            for i in range(batch_size)
        ])
        uniform_pcd = torch.from_numpy(uniform_pcd).float().to(device)

        if self.use_ground_truth_position_for_sampling and gt_action is not None:
            # Sample the ground-truth position as an additional ghost point
            ground_truth_pcd = einops.rearrange(gt_action, "b t c -> (b t) c")[:, :3].unsqueeze(1).detach()
            ghost_pcd = torch.cat([uniform_pcd, ground_truth_pcd], dim=1)
        else:
            ghost_pcd = uniform_pcd

        return ghost_pcd

    def _compute_ghost_point_features(self, ghost_pcd, context_features, context_pos, batch_size):
        """
        Ghost points cross-attend to context features (visual features and current
        gripper position).
        """
        # Initialize ghost point features and positional embeddings
        ghost_pcd_pos = self.pcd_pe_layer(ghost_pcd)
        ghost_pcd_features = self.ghost_points_embed.weight.unsqueeze(0).repeat(
            self.num_ghost_points, batch_size, 1)

        # Ghost points cross-attend to visual features and current gripper position
        for i in range(len(self.ghost_point_cross_attn_layers)):
            ghost_pcd_features = self.ghost_point_cross_attn_layers[i](
                query=ghost_pcd_features, value=context_features,
                query_pos=ghost_pcd_pos, value_pos=context_pos
            )
            ghost_pcd_features = self.ghost_point_ffw_layers[i](ghost_pcd_features)

        return ghost_pcd_features, ghost_pcd_pos

    def _decode_mask(self,
                     visible_rgb_features, ghost_pcd_features,
                     rgb_height, rgb_width,
                     query_features, context_features,
                     query_pos, context_pos):
        """
        The query cross-attends to context features (visual features, ghost points, and the
        current gripper position) then decodes a mask over ghost points (used to predict
        the gripper position) and over visual features (for visualization only).
        """
        ghost_pcd_masks = []

        for i in range(len(self.query_cross_attn_layers)):
            query_features = self.query_cross_attn_layers[i](
                query=query_features, value=context_features,
                query_pos=query_pos, value_pos=context_pos
            )
            query_features = self.query_ffw_layers[i](query_features)

            ghost_pcd_masks.append(einops.einsum(
                query_features.squeeze(0), ghost_pcd_features, "b c, npts b c -> b npts"))

        visible_rgb_mask = einops.einsum(
            query_features.squeeze(0), visible_rgb_features, "b c, b ncam c h w -> b ncam h w")
        visible_rgb_mask = F.interpolate(
            visible_rgb_mask, size=(rgb_height, rgb_width), mode="bilinear", align_corners=False)

        return query_features, ghost_pcd_masks, visible_rgb_mask

    def _predict_action(self, ghost_pcd_mask, ghost_pcd, ghost_pcd_features, batch_size):
        """Compute the predicted action (position, rotation, opening) from the predicted mask."""
        # Predict position differently depending on the loss
        if self.loss == "mse":
            # Weighted sum of all points
            ghost_pcd_mask = torch.softmax(ghost_pcd_mask, dim=-1)
            position = einops.einsum(ghost_pcd, ghost_pcd_mask, "b c npts, b npts -> b c")
        elif self.loss in ["ce", "bce"]:
            # Top point
            top_idx = torch.max(ghost_pcd_mask, dim=-1).indices
            position = ghost_pcd[torch.arange(batch_size), :, top_idx]

        # Predict rotation and gripper opening from features pooled near the predicted position
        if self.loss in ["ce", "bce"]:
            ghost_pcd_features = einops.rearrange(ghost_pcd_features, "npts b c -> b npts c")

            if self.rotation_pooling_gaussian_spread == 0:
                features = ghost_pcd_features[torch.arange(batch_size), top_idx]
            else:
                # Pool features around predicted position
                l2_pred_pos = ((position.unsqueeze(-1) - ghost_pcd) ** 2).sum(1).sqrt()
                weights = torch.softmax(-l2_pred_pos / self.rotation_pooling_gaussian_spread, dim=-1).detach()
                features = einops.einsum(ghost_pcd_features, weights, "b npts c, b npts -> b c")

            pred = self.gripper_state_predictor(features)
            rotation = normalise_quat(pred[:, :4])
            gripper = torch.sigmoid(pred[:, 4:])

        else:
            raise NotImplementedError

        return position, rotation, gripper
