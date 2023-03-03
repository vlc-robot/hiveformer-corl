import itertools
import pickle
from typing import List, Literal, Dict, Optional, Tuple, TypedDict, Union, Any, Sequence
from pathlib import Path
import json
import torch
import torch.nn.functional as F
import numpy as np
import einops


Camera = Literal["wrist", "left_shoulder", "right_shoulder", "overhead", "front"]
Instructions = Dict[str, Dict[int, torch.Tensor]]


class Sample(TypedDict):
    frame_id: torch.Tensor
    task: Union[List[str], str]
    variation: Union[List[int], int]
    rgbs: torch.Tensor
    pcds: torch.Tensor
    action: torch.Tensor
    padding_mask: torch.Tensor
    instr: torch.Tensor
    gripper: torch.Tensor


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


def get_gripper_loc_bounds(path: str, buffer: float = 0.0):
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
        ground_truth_gaussian_spread,
        position_prediction_only=False,
        compute_loss_at_all_layers=True,
        label_smoothing=0.0,
        position_loss_coeff=1.0,
        position_offset_loss_coeff=1.0,
        rotation_loss_coeff=1.0,
        gripper_loss_coeff=1.0,
        rotation_parametrization="quat_from_query",
        regress_position_offset=False,
        points_supervised_for_offset="fine"
    ):
        assert position_loss in ["mse", "ce"]
        self.position_loss = position_loss
        self.position_prediction_only = position_prediction_only
        self.compute_loss_at_all_layers = compute_loss_at_all_layers
        self.ground_truth_gaussian_spread = ground_truth_gaussian_spread
        self.label_smoothing = label_smoothing
        self.position_loss_coeff = position_loss_coeff
        self.position_offset_loss_coeff = position_offset_loss_coeff
        self.rotation_loss_coeff = rotation_loss_coeff
        self.gripper_loss_coeff = gripper_loss_coeff
        self.rotation_parametrization = rotation_parametrization
        self.regress_position_offset = regress_position_offset
        self.points_supervised_for_offset = points_supervised_for_offset
        task_file = Path(__file__).parent.parent / "tasks/106_tasks.csv"
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
            losses["rotation"] = F.mse_loss(pred["rotation"], gt_action[:, 3:7])
            losses["gripper"] = F.mse_loss(pred["gripper"], gt_action[:, 7:8])

            if pred["task"] is not None:
                task = torch.Tensor([self.tasks.index(t) for t in sample["task"]])
                task = task.to(device).long()
                losses["task"] = F.cross_entropy(pred["task"], task)

            losses["rotation"] *= self.rotation_loss_coeff
            losses["gripper"] *= self.gripper_loss_coeff

        return losses

    def _compute_position_loss(self, pred, gt_position, losses):
        if self.position_loss == "mse":
            # Only used for original HiveFormer
            losses["position_mse"] = F.mse_loss(pred["position"], gt_position) * self.position_loss_coeff

        elif self.position_loss == "ce":
            losses["position_ce"] = []

            # Select a normalized Gaussian ball around the ground-truth as a proxy label
            # for a soft cross-entropy loss
            coarse_l2 = ((pred["coarse_ghost_pcd"] - gt_position.unsqueeze(-1)) ** 2).sum(1).sqrt()
            coarse_label = torch.softmax(-coarse_l2 / self.ground_truth_gaussian_spread, dim=-1).detach()
            if pred.get("fine_ghost_pcd") is not None:
                fine_l2 = ((pred["fine_ghost_pcd"] - gt_position.unsqueeze(-1)) ** 2).sum(1).sqrt()
                fine_label = torch.softmax(-fine_l2 / self.ground_truth_gaussian_spread, dim=-1).detach()

            loss_layers = range(
                len(pred["coarse_ghost_pcd_masks"])) if self.compute_loss_at_all_layers else [-1]

            for i in loss_layers:
                losses["position_ce"].append(F.cross_entropy(
                    pred["coarse_ghost_pcd_masks"][i], coarse_label, label_smoothing=self.label_smoothing))
                if pred.get("fine_ghost_pcd") is not None:
                    losses["position_ce"].append(F.cross_entropy(
                        pred["fine_ghost_pcd_masks"][i], fine_label, label_smoothing=self.label_smoothing))

            losses["position_ce"] = torch.stack(losses["position_ce"]).mean() * self.position_loss_coeff

            # Supervise offset from the ghost point's position to the predicted position
            if pred.get("fine_ghost_pcd_offsets") is not None:
                if self.points_supervised_for_offset == "fine":
                    npts = pred["fine_ghost_pcd"].shape[-1] // 2
                    pred_with_offset = (pred["fine_ghost_pcd"] + pred["fine_ghost_pcd_offsets"])[:, :, npts:]
                    losses["position_offset"] = F.mse_loss(
                        pred_with_offset,
                        gt_position.unsqueeze(-1).repeat(1, 1, pred_with_offset.shape[-1])
                    )
                elif self.points_supervised_for_offset == "closest":
                    b = pred["fine_ghost_pcd"].shape[0]
                    losses["position_offset"] = F.mse_loss(
                        (pred["fine_ghost_pcd"] + pred["fine_ghost_pcd_offsets"])[
                            torch.arange(b), :, torch.min(fine_l2, dim=-1).indices],
                        gt_position
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

        l2 = ((pred["position"] - outputs[:, :3]) ** 2).sum(1).sqrt()

        metrics["mean/position_l2"] = l2.to(dtype).mean()
        metrics["mean/position_l2<0.01"] = (l2 < 0.01).to(dtype).mean()
        metrics["mean/position_l2<0.02"] = (l2 < 0.02).to(dtype).mean()
        metrics["mean/position_l2<0.08"] = (l2 < 0.08).to(dtype).mean()

        for task in np.unique(tasks):
            task_l2 = l2[tasks == task]
            metrics[f"{task}/position_l2"] = task_l2.to(dtype).mean()
            metrics[f"{task}/position_l2<0.01"] = (task_l2 < 0.01).to(dtype).mean()

        if not self.position_prediction_only:
            pred_gripper = (pred["gripper"] > 0.5).squeeze(-1)
            true_gripper = outputs[:, 7].bool()
            acc = pred_gripper == true_gripper
            metrics["gripper"] = acc.to(dtype).mean()

            l1 = ((pred["rotation"] - outputs[:, 3:7]).abs().sum(1))
            metrics["mean/rotation_l1"] = l1.to(dtype).mean()
            metrics["mean/rotation_l1<0.025"] = (l1 < 0.025).to(dtype).mean()
            metrics["mean/rotation_l1<0.05"] = (l1 < 0.05).to(dtype).mean()

            for task in np.unique(tasks):
                task_l1 = l1[tasks == task]
                metrics[f"{task}/rotation_l1"] = task_l1.to(dtype).mean()
                metrics[f"{task}/rotation_l1<0.05"] = (task_l1 < 0.05).to(dtype).mean()

            if pred["task"] is not None:
                task = torch.Tensor([self.tasks.index(t) for t in sample["task"]])
                task = task.to(device).long()
                acc = task == pred["task"].argmax(1)
                metrics["task"] = acc.to(dtype).mean()

        return metrics
