import torch
import torch.nn as nn

from .prediction_head import PredictionHead
try:
    from pytorch3d import transforms as torch3d_tf
except:
    pass


class Baseline(nn.Module):
    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 num_ghost_point_cross_attn_layers=2,
                 num_query_cross_attn_layers=2,
                 num_vis_ins_attn_layers=2,
                 rotation_parametrization="quat_from_query",
                 gripper_loc_bounds=None,
                 num_ghost_points=1000,
                 num_ghost_points_val=10000,
                 weight_tying=True,
                 ins_pos_emb=False,
                 gp_emb_tying=True,
                 num_sampling_level=3,
                 fine_sampling_ball_diameter=0.16,
                 regress_position_offset=False,
                 visualize_rgb_attn=False,
                 use_instruction=False,
                 task_specific_biases=False,
                 positional_features="none",
                 task_ids=[]):
        super().__init__()

        self.prediction_head = PredictionHead(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            num_ghost_point_cross_attn_layers=num_ghost_point_cross_attn_layers,
            num_query_cross_attn_layers=num_query_cross_attn_layers,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            rotation_parametrization=rotation_parametrization,
            num_ghost_points=num_ghost_points,
            num_ghost_points_val=num_ghost_points_val,
            weight_tying=weight_tying,
            ins_pos_emb=ins_pos_emb,
            gp_emb_tying=gp_emb_tying,
            num_sampling_level=num_sampling_level,
            fine_sampling_ball_diameter=fine_sampling_ball_diameter,
            gripper_loc_bounds=gripper_loc_bounds,
            regress_position_offset=regress_position_offset,
            visualize_rgb_attn=visualize_rgb_attn,
            use_instruction=use_instruction,
            task_specific_biases=task_specific_biases,
            positional_features=positional_features,
            task_ids=task_ids,
        )

    def compute_action(self, pred) -> torch.Tensor:
        if "quat" in self.prediction_head.rotation_parametrization:
            rotation = pred["rotation"]
        elif "6D" in self.prediction_head.rotation_parametrization:
            rotation = torch3d_tf.matrix_to_quaternion(pred["rotation"])
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
                task_id,
                gt_action=None):

        history_length = rgb_obs.shape[1]
        instruction = instruction.unsqueeze(1).repeat(1, history_length, 1, 1)[padding_mask]
        task_id = task_id.unsqueeze(1).repeat(1, history_length)[padding_mask]
        visible_pcd = pcd_obs[padding_mask]
        visible_rgb = rgb_obs[padding_mask]
        curr_gripper = gripper[padding_mask][:, :3]
        if gt_action is not None:
            gt_action = gt_action[padding_mask]

        # Undo pre-processing to feed RGB to pre-trained backbone (from [-1, 1] to [0, 1])
        visible_rgb = (visible_rgb / 2 + 0.5)
        visible_rgb = visible_rgb[:, :, :3, :, :]

        pred = self.prediction_head(
            visible_rgb=visible_rgb,
            visible_pcd=visible_pcd,
            curr_gripper=curr_gripper,
            instruction=instruction,
            task_id=task_id,
            gt_action=gt_action,
        )

        pred["task"] = None
        return pred
