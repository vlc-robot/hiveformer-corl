import einops
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from torchvision.ops import FeaturePyramidNetwork

from .multihead_custom_attention import MultiheadCustomAttention
from .position_encodings import PositionEmbeddingLearned, RotaryPositionEncoding3D


def normalise_quat(x: torch.Tensor):
    return x / x.square().sum(dim=-1).sqrt().unsqueeze(-1)


class CrossAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query, value, query_pos=None, value_pos=None):
        attn = self.multihead_attn(
            query=(query + query_pos) if query_pos is not None else query,
            key=(value + value_pos) if value_pos is not None else value,
            value=value
        )[0]
        output = query + self.dropout(attn)
        output = self.norm(output)
        return output


class RelativeCrossAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.0):
        super().__init__()
        self.multihead_attn = MultiheadCustomAttention(embedding_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, value, query_pos=None, value_pos=None):
        attn = self.multihead_attn(
            query=query,
            key=value,
            value=value,
            rotary_pe=(query_pos, value_pos) if query_pos is not None else None
        )[0]
        output = query + self.dropout(attn)
        output = self.norm(output)
        return output


class FeedforwardLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.activation = F.relu
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        output = x + self.dropout(output)
        output = self.norm(output)
        return output


class PositionPrediction(nn.Module):
    def __init__(self,
                 embedding_dim=128,
                 num_attn_heads=4,
                 num_ghost_point_cross_attn_layers=4,
                 num_query_cross_attn_layers=4,
                 loss="ce"):
        super().__init__()
        self.loss = loss

        # Frozen ResNet50 backbone
        cfg = get_cfg()
        cfg.merge_from_file(str(Path(__file__).resolve().parent / "resnet50.yaml"))
        model = build_model(cfg)
        self.backbone = model.backbone
        self.pixel_mean = model.pixel_mean.to()
        self.pixel_std = model.pixel_std
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Fine-grained semantic visual features
        self.feature_pyramid = FeaturePyramidNetwork([256, 512, 1024, 2048], embedding_dim)

        # 3D positional embeddings
        self.pcd_pe_layer = PositionEmbeddingLearned(3, embedding_dim)

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
            self.ghost_point_cross_attn_layers.append(CrossAttentionLayer(embedding_dim, num_attn_heads))
            self.ghost_point_ffw_layers.append(FeedforwardLayer(embedding_dim, embedding_dim))

        # Query cross-attention to visual features, ghost points, and the current gripper position
        self.query_cross_attn_layers = nn.ModuleList()
        self.query_ffw_layers = nn.ModuleList()
        for _ in range(num_query_cross_attn_layers):
            self.query_cross_attn_layers.append(CrossAttentionLayer(embedding_dim, num_attn_heads))
            self.query_ffw_layers.append(FeedforwardLayer(embedding_dim, embedding_dim))

    def forward(self, visible_rgb, visible_pcd, curr_gripper, ghost_pcd):
        """
        Arguments:
            visible_rgb: (batch x history, num_cameras, 3, height, width) in [0, 1]
            visible_pcd: (batch x history, num_cameras, 3, height, width) in world coordinates
            curr_gripper: (batch x history, 3)
            ghost_pcd: (batch x history, num_points, 3) in world coordinates

        Returns:
            next_gripper: (batch x history, 3)
        """
        batch_size, num_cameras, _, height, width = visible_rgb.shape
        num_ghost_points = ghost_pcd.shape[1]
        device = visible_rgb.device

        # Pass each view independently through ResNet50 backbone
        visible_rgb = einops.rearrange(visible_rgb, "b ncam c h w -> (b ncam) c h w")
        self.pixel_mean = self.pixel_mean.to(device)
        self.pixel_std = self.pixel_std.to(device)
        visible_rgb = (visible_rgb - self.pixel_mean) / self.pixel_std
        visible_rgb_features = self.backbone(visible_rgb)

        # Compute fine-grained semantic visual features and their positional embeddings
        visible_rgb_features = self.feature_pyramid(visible_rgb_features)
        visible_rgb_features = visible_rgb_features["res2"]
        visible_rgb_pos = self.pcd_pe_layer(
            visible_rgb_features, einops.rearrange(visible_pcd, "b ncam c h w -> (b ncam) c h w"))
        visible_rgb_features = einops.rearrange(
            visible_rgb_features, "(b ncam) c h w -> b ncam c h w", ncam=num_cameras)
        visible_rgb_pos = einops.rearrange(
            visible_rgb_pos, "(b ncam) c h w -> b ncam c h w", ncam=num_cameras)

        # Compute current gripper position features and positional embeddings
        curr_gripper_pos = self.pcd_pe_layer.position_embedding_head(curr_gripper.unsqueeze(-1))
        curr_gripper_pos = einops.rearrange(curr_gripper_pos, "b c x -> x b c", x=1)
        curr_gripper_features = self.curr_gripper_embed.weight.repeat(batch_size, 1).unsqueeze(0)

        # Initialize ghost point features and positional embeddings
        ghost_pcd = einops.rearrange(ghost_pcd, "b npts c -> b c npts")
        ghost_pcd_pos = self.pcd_pe_layer.position_embedding_head(ghost_pcd)
        ghost_pcd_pos = einops.rearrange(ghost_pcd_pos, "b c npts -> npts b c")
        ghost_pcd_features = self.ghost_points_embed.weight.unsqueeze(0).repeat(
            num_ghost_points, batch_size, 1)

        # Ghost points cross-attend to visual features and current gripper position
        context_features = einops.rearrange(visible_rgb_features, "b ncam c h w -> (ncam h w) b c")
        context_pos = einops.rearrange(visible_rgb_pos, "b ncam c h w -> (ncam h w) b c")
        context_features = torch.cat([context_features, curr_gripper_features], dim=0)
        context_pos = torch.cat([context_pos, curr_gripper_pos], dim=0)

        for i in range(len(self.ghost_point_cross_attn_layers)):
            ghost_pcd_features = self.ghost_point_cross_attn_layers[i](
                query=ghost_pcd_features, value=context_features,
                query_pos=ghost_pcd_pos, value_pos=context_pos
            )
            ghost_pcd_features = self.ghost_point_ffw_layers[i](ghost_pcd_features)

        # Intialize query features
        query_features = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)

        # The query cross-attends to visual features, ghost points, and the current gripper position
        # then decodes a mask over visual features (which we up-sample) and ghost points
        context_features = torch.cat([context_features, ghost_pcd_features], dim=0)
        context_pos = torch.cat([context_pos, ghost_pcd_pos], dim=0)

        visible_rgb_masks = []
        ghost_pcd_masks = []
        all_masks = []

        for i in range(len(self.query_cross_attn_layers)):
            query_features = self.query_cross_attn_layers[i](
                query=query_features, value=context_features,
                query_pos=None, value_pos=context_pos
            )
            query_features = self.query_ffw_layers[i](query_features)

            visible_rgb_mask = einops.einsum(
                query_features.squeeze(0), visible_rgb_features, "b c, b ncam c h w -> b ncam h w")
            visible_rgb_mask = F.interpolate(
                visible_rgb_mask, size=(height, width), mode="bilinear", align_corners=False)

            ghost_pcd_mask = einops.einsum(
                query_features.squeeze(0), ghost_pcd_features, "b c, npts b c -> b npts")

            all_mask = einops.rearrange(visible_rgb_mask, "b ncam h w -> b (ncam h w)")
            all_mask = torch.cat([all_mask, ghost_pcd_mask], dim=-1)

            visible_rgb_masks.append(visible_rgb_mask)
            ghost_pcd_masks.append(ghost_pcd_mask)
            all_masks.append(all_mask)

        # Predict position differently depending on the loss we use
        all_pcd = einops.rearrange(visible_pcd, "b ncam c h w -> b c (ncam h w)")
        all_pcd = torch.cat([all_pcd, ghost_pcd], dim=-1)

        if self.loss == "mse":
            # Weighted sum of all points
            all_mask = torch.softmax(all_masks[-1], dim=-1)
            position = einops.einsum(all_pcd, all_mask, "b c npts, b npts -> b c")

        elif self.loss in ["ce", "bce"]:
            # Top point
            top_idx = torch.max(all_masks[-1], dim=-1).indices
            position = all_pcd[torch.arange(batch_size), :, top_idx]

        return {
            "position": position,
            "visible_rgb_masks": visible_rgb_masks,
            "ghost_pcd_masks":  ghost_pcd_masks,
            "all_masks": all_masks,
            "all_pcd": all_pcd
        }


class RelativePositionPrediction(nn.Module):
    def __init__(self,
                 embedding_dim=128,
                 num_attn_heads=4,
                 num_ghost_point_cross_attn_layers=4,
                 num_query_cross_attn_layers=4,
                 loss="ce",
                 rotation_pooling_gaussian_spread=0.01):
        super().__init__()
        self.loss = loss
        self.rotation_pooling_gaussian_spread = rotation_pooling_gaussian_spread

        # Frozen ResNet50 backbone
        cfg = get_cfg()
        cfg.merge_from_file(str(Path(__file__).resolve().parent / "resnet50.yaml"))
        model = build_model(cfg)
        self.backbone = model.backbone
        self.pixel_mean = model.pixel_mean.to()
        self.pixel_std = model.pixel_std
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Fine-grained semantic visual features
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

    def forward(self, visible_rgb, visible_pcd, curr_gripper, ghost_pcd):
        """
        Arguments:
            visible_rgb: (batch x history, num_cameras, 3, height, width) in [0, 1]
            visible_pcd: (batch x history, num_cameras, 3, height, width) in world coordinates
            curr_gripper: (batch x history, 3)
            ghost_pcd: (batch x history, num_points, 3) in world coordinates

        Returns:
            next_gripper: (batch x history, 3)
        """
        batch_size, num_cameras, _, height, width = visible_rgb.shape
        num_ghost_points = ghost_pcd.shape[1]
        device = visible_rgb.device

        # Pass each view independently through ResNet50 backbone
        visible_rgb = einops.rearrange(visible_rgb, "b ncam c h w -> (b ncam) c h w")
        self.pixel_mean = self.pixel_mean.to(device)
        self.pixel_std = self.pixel_std.to(device)
        visible_rgb = (visible_rgb - self.pixel_mean) / self.pixel_std
        visible_rgb_features = self.backbone(visible_rgb)

        # Compute fine-grained semantic visual features and their positional embeddings
        visible_rgb_features = self.feature_pyramid(visible_rgb_features)
        visible_rgb_features = visible_rgb_features["res2"]
        visible_rgb_features = einops.rearrange(
            visible_rgb_features, "(b ncam) c h w -> b ncam c h w", ncam=num_cameras)
        visible_pcd_ = einops.rearrange(visible_pcd, "b ncam c h w -> (b ncam) c h w")
        visible_pcd_ = F.interpolate(visible_pcd_, scale_factor=1. / 4, mode='bilinear')
        visible_pcd_ = einops.rearrange(visible_pcd_, "(b ncam) c h w -> b (ncam h w) c", ncam=num_cameras)
        visible_rgb_pos = self.pcd_pe_layer(visible_pcd_)

        # Compute current gripper position features and positional embeddings
        curr_gripper_pos = self.pcd_pe_layer(curr_gripper.unsqueeze(1))
        curr_gripper_features = self.curr_gripper_embed.weight.repeat(batch_size, 1).unsqueeze(0)

        # Initialize ghost point features and positional embeddings
        ghost_pcd_pos = self.pcd_pe_layer(ghost_pcd)
        ghost_pcd_features = self.ghost_points_embed.weight.unsqueeze(0).repeat(
            num_ghost_points, batch_size, 1)

        # Ghost points cross-attend to visual features and current gripper position
        context_features = einops.rearrange(visible_rgb_features, "b ncam c h w -> (ncam h w) b c")
        context_features = torch.cat([context_features, curr_gripper_features], dim=0)
        context_pos = torch.cat([visible_rgb_pos, curr_gripper_pos], dim=1)

        for i in range(len(self.ghost_point_cross_attn_layers)):
            ghost_pcd_features = self.ghost_point_cross_attn_layers[i](
                query=ghost_pcd_features, value=context_features,
                query_pos=ghost_pcd_pos, value_pos=context_pos
            )
            ghost_pcd_features = self.ghost_point_ffw_layers[i](ghost_pcd_features)

        # Intialize query features
        query_features = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)

        # The query cross-attends to visual features, ghost points, and the current gripper position
        # then decodes a mask over visual features (which we up-sample) and ghost points
        context_features = torch.cat([context_features, ghost_pcd_features], dim=0)
        context_pos = torch.cat([context_pos, ghost_pcd_pos], dim=1)

        visible_rgb_masks = []
        ghost_pcd_masks = []
        all_masks = []

        for i in range(len(self.query_cross_attn_layers)):
            query_features = self.query_cross_attn_layers[i](
                query=query_features, value=context_features,
                query_pos=None, value_pos=None  # TODO Do we want positional embeddings here?
            )
            query_features = self.query_ffw_layers[i](query_features)

            visible_rgb_mask = einops.einsum(
                query_features.squeeze(0), visible_rgb_features, "b c, b ncam c h w -> b ncam h w")
            visible_rgb_mask = F.interpolate(
                visible_rgb_mask, size=(height, width), mode="bilinear", align_corners=False)

            ghost_pcd_mask = einops.einsum(
                query_features.squeeze(0), ghost_pcd_features, "b c, npts b c -> b npts")

            all_mask = einops.rearrange(visible_rgb_mask, "b ncam h w -> b (ncam h w)")
            all_mask = torch.cat([all_mask, ghost_pcd_mask], dim=-1)

            visible_rgb_masks.append(visible_rgb_mask)
            ghost_pcd_masks.append(ghost_pcd_mask)
            all_masks.append(all_mask)

        # Predict position differently depending on the loss we use
        all_pcd = torch.cat([
            einops.rearrange(visible_pcd, "b ncam c h w -> b c (ncam h w)"),
            einops.rearrange(ghost_pcd, "b npts c -> b c npts")],
            dim=-1
        )

        if self.loss == "mse":
            # Weighted sum of all points
            all_mask = torch.softmax(all_masks[-1], dim=-1)
            position = einops.einsum(all_pcd, all_mask, "b c npts, b npts -> b c")

        elif self.loss in ["ce", "bce"]:
            # Top point
            top_idx = torch.max(all_masks[-1], dim=-1).indices
            position = all_pcd[torch.arange(batch_size), :, top_idx]

        # Predict rotation and gripper opening from features at the predicted position
        if self.loss in ["ce", "bce"]:
            visible_rgb_features = einops.rearrange(visible_rgb_features, "b ncam c h w -> (b ncam) c h w")
            visible_rgb_features = F.interpolate(
                visible_rgb_features, size=(height, width), mode="bilinear", align_corners=False)
            visible_rgb_features = einops.rearrange(
                visible_rgb_features, "(b ncam) c h w -> b (ncam h w) c", ncam=num_cameras)
            ghost_pcd_features = einops.rearrange(ghost_pcd_features, "npts b c -> b npts c")
            all_features = torch.cat([visible_rgb_features, ghost_pcd_features], dim=1)

            if self.rotation_pooling_gaussian_spread == 0:
                features = all_features[torch.arange(batch_size), top_idx]
            else:
                # Pool features around predicted position
                l2_pred_pos = ((position.unsqueeze(-1) - all_pcd) ** 2).sum(1).sqrt()
                weights = torch.softmax(-l2_pred_pos / self.rotation_pooling_gaussian_spread, dim=-1).detach()
                features = einops.einsum(all_features, weights, "b npts c, b npts -> b c")

            pred = self.gripper_state_predictor(features)
            rotation = normalise_quat(pred[:, :4])
            gripper = torch.sigmoid(pred[:, 4:])

        else:
            raise NotImplementedError

        return {
            # Action
            "position": position,
            "rotation": rotation,
            "gripper": gripper,
            # Auxiliary outputs used to compute the loss
            "visible_rgb_masks": visible_rgb_masks,
            "ghost_pcd_masks":  ghost_pcd_masks,
            "all_masks": all_masks,
            "all_features": all_features,
            "all_pcd": all_pcd,
        }
