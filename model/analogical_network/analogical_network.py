import torch
import torch.nn as nn

from .analogical_prediction_head import AnalogicalPredictionHead
from model.utils.utils import norm_tensor


class AnalogicalNetwork(nn.Module):
    def __init__(self,
                 image_size=(128, 128),
                 embedding_dim=60,
                 num_ghost_point_cross_attn_layers=2,
                 num_query_cross_attn_layers=2,
                 rotation_parametrization="quat_from_query",
                 gripper_loc_bounds=None,
                 num_ghost_points=1000,
                 coarse_to_fine_sampling=True,
                 fine_sampling_ball_diameter=0.08,
                 separate_coarse_and_fine_layers=False,
                 regress_position_offset=False,
                 support_set="rest_of_batch",
                 use_instruction=False):
        super().__init__()

        self.prediction_head = AnalogicalPredictionHead(
            image_size=image_size,
            embedding_dim=embedding_dim,
            num_ghost_point_cross_attn_layers=num_ghost_point_cross_attn_layers,
            num_query_cross_attn_layers=num_query_cross_attn_layers,
            rotation_parametrization=rotation_parametrization,
            num_ghost_points=num_ghost_points,
            coarse_to_fine_sampling=coarse_to_fine_sampling,
            fine_sampling_ball_diameter=fine_sampling_ball_diameter,
            gripper_loc_bounds=gripper_loc_bounds,
            separate_coarse_and_fine_layers=separate_coarse_and_fine_layers,
            regress_position_offset=regress_position_offset,
            support_set=support_set,
            use_instruction=use_instruction,
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
                gt_action_for_support,
                gt_action_for_sampling=None):

        history_length = rgb_obs.shape[2]
        instruction = instruction.unsqueeze(2).repeat(1, 1, history_length, 1, 1)

        visible_pcd = pcd_obs

        # Undo pre-processing to feed RGB to pre-trained backbone (from [-1, 1] to [0, 1])
        visible_rgb = (rgb_obs / 2 + 0.5)
        visible_rgb = visible_rgb[:, :, :, :, :3, :, :]

        curr_gripper = gripper[:, :, :, :3]

        pred = self.prediction_head(
            visible_rgb=visible_rgb,
            visible_pcd=visible_pcd,
            curr_gripper=curr_gripper,
            instruction=instruction,
            padding_mask=padding_mask,
            gt_action_for_support=gt_action_for_support,
            gt_action_for_sampling=gt_action_for_sampling,
        )
        pred["task"] = None
        return pred
