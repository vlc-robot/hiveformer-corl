import torch
import torch.nn as nn

from .analogical_prediction_head import AnalogicalPredictionHead


class AnalogicalNetwork(nn.Module):
    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 num_ghost_point_cross_attn_layers=2,
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

        self.prediction_head = AnalogicalPredictionHead(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            num_ghost_point_cross_attn_layers=num_ghost_point_cross_attn_layers,
            gripper_loc_bounds=gripper_loc_bounds,
            num_ghost_points=num_ghost_points,
            num_ghost_points_val=num_ghost_points_val,
            weight_tying=weight_tying,
            gp_emb_tying=gp_emb_tying,
            num_sampling_level=num_sampling_level,
            fine_sampling_ball_diameter=fine_sampling_ball_diameter,
            regress_position_offset=regress_position_offset,
            use_instruction=use_instruction,
            task_specific_biases=task_specific_biases,
            task_ids=task_ids,
            positional_features=positional_features,
            support_set=support_set,
            global_correspondence=global_correspondence,
            num_matching_cross_attn_layers=num_matching_cross_attn_layers
        )

    def compute_action(self, pred) -> torch.Tensor:
        return torch.cat(
            [pred["position"], pred["rotation"], pred["gripper"]],
            dim=1,
        )

    def forward(self,
                rgb_obs,
                pcd_obs,
                padding_mask,
                instruction,
                gripper,
                task_id,
                gt_action_for_support,
                gt_action_for_sampling=None):

        history_length = rgb_obs.shape[2]
        instruction = instruction.unsqueeze(2).repeat(1, 1, history_length, 1, 1)
        task_id = task_id.unsqueeze(2).repeat(1, 1, history_length)

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
            task_id=task_id,
            padding_mask=padding_mask,
            gt_action_for_support=gt_action_for_support,
            gt_action_for_sampling=gt_action_for_sampling,
        )

        pred["task"] = None
        return pred
