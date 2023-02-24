import einops
import torch
import torch.nn as nn

from .prediction_head import PredictionHead
from model.utils.utils import norm_tensor


class Baseline(nn.Module):
    def __init__(self,
                 image_size=(128, 128),
                 embedding_dim=60,
                 num_ghost_point_cross_attn_layers=2,
                 num_query_cross_attn_layers=2,
                 rotation_parametrization="quat_from_query",
                 gripper_loc_bounds=None,
                 num_ghost_points=1000,
                 coarse_to_fine_sampling=True,
                 fine_sampling_cube_size=0.05,
                 separate_coarse_and_fine_layers=False,
                 regress_position_offset=False):
        super().__init__()

        self.prediction_head = PredictionHead(
            image_size=image_size,
            embedding_dim=embedding_dim,
            num_ghost_point_cross_attn_layers=num_ghost_point_cross_attn_layers,
            num_query_cross_attn_layers=num_query_cross_attn_layers,
            rotation_parametrization=rotation_parametrization,
            num_ghost_points=num_ghost_points,
            coarse_to_fine_sampling=coarse_to_fine_sampling,
            fine_sampling_cube_size=fine_sampling_cube_size,
            gripper_loc_bounds=gripper_loc_bounds,
            separate_coarse_and_fine_layers=separate_coarse_and_fine_layers,
            regress_position_offset=regress_position_offset,
        )

    def compute_action(self, pred) -> torch.Tensor:
        rotation = norm_tensor(pred["rotation"])
        return torch.cat(
            [pred["position"], rotation, pred["gripper"]],
            dim=1,
        )

    def forward(self,
                rgb_obs,
                pcd_obs,
                padding_mask,
                instruction,
                gripper,
                gt_action=None):
        visible_pcd = pcd_obs[padding_mask]

        # Undo pre-processing to feed RGB to pre-trained ResNet (from [-1, 1] to [0, 1])
        visible_rgb = einops.rearrange(rgb_obs, "b t n d h w -> (b t) n d h w")
        visible_rgb = (visible_rgb / 2 + 0.5)
        visible_rgb = visible_rgb[:, :, :3, :, :]

        curr_gripper = einops.rearrange(gripper, "b t c -> (b t) c")[:, :3]

        pred = self.prediction_head(
            visible_rgb=visible_rgb,
            visible_pcd=visible_pcd,
            curr_gripper=curr_gripper,
            gt_action=gt_action,
        )
        pred["task"] = None
        return pred
