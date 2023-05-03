import itertools
import pickle
from typing import List, Literal, Dict, Optional, Tuple, TypedDict, Union, Any, Sequence
from pathlib import Path
import json
import torch
import torch.nn.functional as F
import numpy as np
import einops
try:
    from pytorch3d import transforms as torch3d_tf
except:
    pass


Camera = Literal["wrist", "left_shoulder", "right_shoulder", "overhead", "front"]
Instructions = Dict[str, Dict[int, torch.Tensor]]


class Sample(TypedDict):
    frame_id: torch.Tensor
    task: Union[List[str], str]
    task_id: int
    variation: Union[List[int], int]
    rgbs: torch.Tensor
    pcds: torch.Tensor
    action: torch.Tensor
    padding_mask: torch.Tensor
    instr: torch.Tensor
    gripper: torch.Tensor


def round_floats(o):
    if isinstance(o, float): return round(o, 2)
    if isinstance(o, dict): return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [round_floats(x) for x in o]
    return o


def normalise_quat(x: torch.Tensor):
    return x / x.square().sum(dim=-1).sqrt().unsqueeze(-1)


def load_episodes() -> Dict[str, Any]:
    with open(Path(__file__).parent.parent / "data_preprocessing/episodes.json") as fid:
        return json.load(fid)


def get_max_episode_length(tasks: Tuple[str, ...], variations: Tuple[int, ...]) -> int:
    max_episode_length = 0
    max_eps_dict = load_episodes()["max_episode_length"]

    for task, var in itertools.product(tasks, variations):
        if max_eps_dict[task] > max_episode_length:
            max_episode_length = max_eps_dict[task]

    return max_episode_length


def get_gripper_loc_bounds(path: str, buffer: float = 0.0, task: Optional[str] = None):
    gripper_loc_bounds = json.load(open(path, "r"))
    if task is not None:
        gripper_loc_bounds = gripper_loc_bounds[task]
        gripper_loc_bounds_min = np.array(gripper_loc_bounds[0]) - buffer
        gripper_loc_bounds_max = np.array(gripper_loc_bounds[1]) + buffer
        gripper_loc_bounds = np.stack([gripper_loc_bounds_min, gripper_loc_bounds_max])
    else:
        # Gripper workspace is the union of workspaces for all tasks
        gripper_loc_bounds = json.load(open(path, "r"))
        gripper_loc_bounds_min = np.min(np.stack([bounds[0] for bounds in gripper_loc_bounds.values()]), axis=0) - buffer
        gripper_loc_bounds_max = np.max(np.stack([bounds[1] for bounds in gripper_loc_bounds.values()]), axis=0) + buffer
        gripper_loc_bounds = np.stack([gripper_loc_bounds_min, gripper_loc_bounds_max])
    print("Gripper workspace size:", gripper_loc_bounds_max - gripper_loc_bounds_min)
    return gripper_loc_bounds


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def norm_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / torch.linalg.norm(tensor, ord=2, dim=-1, keepdim=True)


def compute_geodesic_distance_from_two_matrices(m1, m2):
    # From https://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tools.py
    device = m1.device
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).to(device)))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).to(device)) * -1)
    theta = torch.acos(cos)
    return theta


def load_instructions(
    instructions: Optional[Path],
    tasks: Optional[Sequence[str]] = None,
    variations: Optional[Sequence[int]] = None,
) -> Optional[Instructions]:
    if instructions is not None:
        with open(instructions, "rb") as fid:
            data: Instructions = pickle.load(fid)
        if tasks is not None:
            data = {task: var_instr for task, var_instr in data.items() if task in tasks}
        if variations is not None:
            data = {
                task: {
                    var: instr for var, instr in var_instr.items() if var in variations
                }
                for task, var_instr in data.items()
            }
        return data
    return None



class LossAndMetrics:
    def __init__(
        self,
        position_loss,
        rotation_parametrization,
        ground_truth_gaussian_spread,
        position_prediction_only=False,
        compute_loss_at_all_layers=False,
        label_smoothing=0.0,
        position_loss_coeff=1.0,
        position_offset_loss_coeff=10000.0,
        rotation_loss_coeff=10.0,
        gripper_loss_coeff=1.0,
        regress_position_offset=False,
        symmetric_rotation_loss=False,
    ):
        assert position_loss in ["mse", "ce"]
        assert rotation_parametrization in [
            "quat_from_top_ghost", "quat_from_query", "6D_from_top_ghost", "6D_from_query"]
        self.position_loss = position_loss
        self.rotation_parametrization = rotation_parametrization
        self.position_prediction_only = position_prediction_only
        self.compute_loss_at_all_layers = compute_loss_at_all_layers
        self.ground_truth_gaussian_spread = ground_truth_gaussian_spread
        self.label_smoothing = label_smoothing
        self.position_loss_coeff = position_loss_coeff
        self.position_offset_loss_coeff = position_offset_loss_coeff
        self.rotation_loss_coeff = rotation_loss_coeff
        self.gripper_loss_coeff = gripper_loss_coeff
        self.regress_position_offset = regress_position_offset
        self.symmetric_rotation_loss = symmetric_rotation_loss
        task_file = Path(__file__).parent.parent / "tasks/all_82_tasks.csv"
        with open(task_file) as fid:
            self.tasks = [t.strip() for t in fid.readlines()]

    def compute_loss(
        self, pred: Dict[str, torch.Tensor], sample: Sample,
    ) -> Dict[str, torch.Tensor]:
        device = pred["position"].device
        padding_mask = sample["padding_mask"].to(device)
        gt_action = sample["action"].to(device)[padding_mask]

        losses = {}

        self._compute_position_loss(pred, gt_action[:, :3], losses)

        if not self.position_prediction_only:
            self._compute_rotation_loss(pred, gt_action[:, 3:7], losses)

            losses["gripper"] = F.mse_loss(pred["gripper"], gt_action[:, 7:8])
            losses["gripper"] *= self.gripper_loss_coeff

            if pred["task"] is not None:
                task = torch.Tensor([self.tasks.index(t) for t in sample["task"]])
                task = task.to(device).long()
                losses["task"] = F.cross_entropy(pred["task"], task)

        return losses

    def _compute_rotation_loss(self, pred, gt_quat, losses):
        if "quat" in self.rotation_parametrization:
            if self.symmetric_rotation_loss:
                gt_quat_ = -gt_quat.clone()
                quat_loss = F.mse_loss(pred["rotation"], gt_quat, reduction='none').mean(1)
                quat_loss_ = F.mse_loss(pred["rotation"], gt_quat_, reduction='none').mean(1)
                select_mask = (quat_loss < quat_loss_).float()
                losses['rotation'] = (select_mask * quat_loss + (1 - select_mask) * quat_loss_).mean()
            else:
                losses["rotation"] = F.mse_loss(pred["rotation"], gt_quat)

        elif "6D" in self.rotation_parametrization:
            gt_rot3x3 = torch3d_tf.quaternion_to_matrix(gt_quat)
            # losses["rotation"] = F.mse_loss(pred["rotation"], gt_rot3x3)
            losses["rotation"] = compute_geodesic_distance_from_two_matrices(pred["rotation"], gt_rot3x3).mean()

        losses["rotation"] *= self.rotation_loss_coeff

    def _compute_position_loss(self, pred, gt_position, losses):
        if self.position_loss == "mse":
            # Only used for original HiveFormer
            losses["position_mse"] = F.mse_loss(pred["position"], gt_position) * self.position_loss_coeff

        elif self.position_loss == "ce":
            # Select a normalized Gaussian ball around the ground-truth as a proxy label
            # for a soft cross-entropy loss
            l2_pyramid = []
            label_pyramid = []
            for ghost_pcd_i in pred['ghost_pcd_pyramid']:
                l2_i = ((ghost_pcd_i - gt_position.unsqueeze(-1)) ** 2).sum(1).sqrt()
                label_i = torch.softmax(-l2_i / self.ground_truth_gaussian_spread, dim=-1).detach()
                l2_pyramid.append(l2_i)
                label_pyramid.append(label_i)

            loss_layers = range(len(pred['ghost_pcd_masks_pyramid'][0])) if self.compute_loss_at_all_layers else [-1]

            for j in loss_layers:
                for i, ghost_pcd_masks_i in enumerate(pred["ghost_pcd_masks_pyramid"]):
                    losses[f"position_ce_level{i}"] = F.cross_entropy(
                        ghost_pcd_masks_i[j], label_pyramid[i],
                        label_smoothing=self.label_smoothing
                    ).mean() * self.position_loss_coeff / len(pred["ghost_pcd_masks_pyramid"])

            # Supervise offset from the ghost point's position to the predicted position
            num_sampling_level = len(pred['ghost_pcd_masks_pyramid'])
            if pred.get("fine_ghost_pcd_offsets") is not None:
                if pred["ghost_pcd_pyramid"][-1].shape[-1] != pred["ghost_pcd_pyramid"][0].shape[-1]:
                    npts = pred["ghost_pcd_pyramid"][-1].shape[-1] // num_sampling_level
                    pred_with_offset = (pred["ghost_pcd_pyramid"][-1] + pred["fine_ghost_pcd_offsets"])[:, :, -npts:]
                else:
                    pred_with_offset = (pred["ghost_pcd_pyramid"][-1] + pred["fine_ghost_pcd_offsets"])
                losses["position_offset"] = F.mse_loss(
                    pred_with_offset,
                    gt_position.unsqueeze(-1).repeat(1, 1, pred_with_offset.shape[-1])
                )
                losses["position_offset"] *= (self.position_offset_loss_coeff * self.position_loss_coeff)

            # Clear gradient on pred["position"] to avoid a memory leak since we don't
            # use it in the loss
            pred["position"] = pred["position"].detach()

    def compute_metrics(
        self, pred: Dict[str, torch.Tensor], sample: Sample
    ) -> Dict[str, torch.Tensor]:
        device = pred["position"].device
        dtype = pred["position"].dtype
        padding_mask = sample["padding_mask"].to(device)
        outputs = sample["action"].to(device)[padding_mask]

        metrics = {}

        tasks = np.array(sample["task"])
        if len(tasks.shape) == 2:
            tasks = einops.rearrange(tasks, "s b -> b s")[:, :, np.newaxis]
        else:
            tasks = tasks[:, np.newaxis]
        tasks = np.repeat(tasks, padding_mask.shape[-1], axis=-1)[padding_mask.cpu()]

        final_pos_l2 = ((pred["position"] - outputs[:, :3]) ** 2).sum(1).sqrt()
        metrics["mean/pos_l2_final"] = final_pos_l2.to(dtype).mean()
        metrics["mean/pos_l2_final<0.01"] = (final_pos_l2 < 0.01).to(dtype).mean()

        for i in range(len(pred["position_pyramid"])):
            pos_l2_i = ((pred["position_pyramid"][i].squeeze(1) - outputs[:, :3]) ** 2).sum(1).sqrt()
            metrics[f"mean/pos_l2_level{i}"] = pos_l2_i.to(dtype).mean()

        for task in np.unique(tasks):
            task_l2 = final_pos_l2[tasks == task]
            metrics[f"{task}/pos_l2_final"] = task_l2.to(dtype).mean()
            metrics[f"{task}/pos_l2_final<0.01"] = (task_l2 < 0.01).to(dtype).mean()

        if not self.position_prediction_only:
            # Gripper accuracy
            pred_gripper = (pred["gripper"] > 0.5).squeeze(-1)
            true_gripper = outputs[:, 7].bool()
            acc = pred_gripper == true_gripper
            metrics["gripper"] = acc.to(dtype).mean()

            # Rotation accuracy
            gt_quat = outputs[:, 3:7]
            if "quat" in self.rotation_parametrization:
                if self.symmetric_rotation_loss:
                    gt_quat_ = -gt_quat.clone()
                    l1 = (pred["rotation"] - gt_quat).abs().sum(1)
                    l1_ = (pred["rotation"] - gt_quat_).abs().sum(1)
                    select_mask = (l1 < l1_).float()
                    l1 = (select_mask * l1 + (1 - select_mask) * l1_)
                else:
                    l1 = ((pred["rotation"] - gt_quat).abs().sum(1))
            elif "6D" in self.rotation_parametrization:
                pred_quat = torch3d_tf.matrix_to_quaternion(pred["rotation"])
                l1 = ((pred_quat - gt_quat).abs().sum(1))

            metrics["mean/rot_l1"] = l1.to(dtype).mean()
            metrics["mean/rot_l1<0.05"] = (l1 < 0.05).to(dtype).mean()
            metrics["mean/rot_l1<0.025"] = (l1 < 0.025).to(dtype).mean()

            for task in np.unique(tasks):
                task_l1 = l1[tasks == task]
                metrics[f"{task}/rot_l1"] = task_l1.to(dtype).mean()
                metrics[f"{task}/rot_l1<0.05"] = (task_l1 < 0.05).to(dtype).mean()
                metrics[f"{task}/rot_l1<0.025"] = (task_l1 < 0.025).to(dtype).mean()

            # Task prediction (not used by our models)
            if pred["task"] is not None:
                task = torch.Tensor([self.tasks.index(t) for t in sample["task"]])
                task = task.to(device).long()
                acc = task == pred["task"].argmax(1)
                metrics["task"] = acc.to(dtype).mean()

        return metrics


ALL_TASKS = [
    'basketball_in_hoop', 'beat_the_buzz', 'change_channel', 'change_clock', 'close_box',
    'close_door', 'close_drawer', 'close_fridge', 'close_grill', 'close_jar', 'close_laptop_lid',
    'close_microwave', 'hang_frame_on_hanger', 'insert_onto_square_peg', 'insert_usb_in_computer',
    'lamp_off', 'lamp_on', 'lift_numbered_block', 'light_bulb_in', 'meat_off_grill', 'meat_on_grill',
    'move_hanger', 'open_box', 'open_door', 'open_drawer', 'open_fridge', 'open_grill',
    'open_microwave', 'open_oven', 'open_window', 'open_wine_bottle', 'phone_on_base',
    'pick_and_lift', 'pick_and_lift_small', 'pick_up_cup', 'place_cups', 'place_hanger_on_rack',
    'place_shape_in_shape_sorter', 'place_wine_at_rack_location', 'play_jenga',
    'plug_charger_in_power_supply', 'press_switch', 'push_button', 'push_buttons', 'put_books_on_bookshelf',
    'put_groceries_in_cupboard', 'put_item_in_drawer', 'put_knife_on_chopping_board', 'put_money_in_safe',
    'put_rubbish_in_bin', 'put_umbrella_in_umbrella_stand', 'reach_and_drag', 'reach_target',
    'scoop_with_spatula', 'screw_nail', 'setup_checkers', 'slide_block_to_color_target',
    'slide_block_to_target', 'slide_cabinet_open_and_place_cups', 'stack_blocks', 'stack_cups',
    'stack_wine', 'straighten_rope', 'sweep_to_dustpan', 'sweep_to_dustpan_of_size', 'take_frame_off_hanger',
    'take_lid_off_saucepan', 'take_money_out_safe', 'take_plate_off_colored_dish_rack', 'take_shoes_out_of_box',
    'take_toilet_roll_off_stand', 'take_umbrella_out_of_umbrella_stand', 'take_usb_out_of_computer',
    'toilet_seat_down', 'toilet_seat_up', 'tower3', 'turn_oven_on', 'turn_tap', 'tv_on', 'unplug_charger',
    'water_plants', 'wipe_desk'
]
TASK_TO_ID = {task: i for i, task in enumerate(ALL_TASKS)}
