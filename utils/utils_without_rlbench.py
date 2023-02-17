import itertools
import pickle
from typing import List, Literal, Dict, Optional, Tuple, TypedDict, Union, Any, Sequence
from pathlib import Path
import json
import torch
from torch import nn
import torch.nn.functional as F
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def norm_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / torch.linalg.norm(tensor, ord=2, dim=-1, keepdim=True)


def compute_rotation_metrics(
    pred: torch.Tensor,
    true: torch.Tensor,
    reduction: str = "mean",
) -> Dict[str, torch.Tensor]:
    pred = norm_tensor(pred)
    acc = (pred - true).abs().max(1).values < 0.05
    acc = acc.to(pred.dtype)
    if reduction == "mean":
        acc = acc.mean()
    return {"rotation": acc}


def compute_rotation_loss(logit: torch.Tensor, rot: torch.Tensor):
    dtype = logit.dtype
    loss = F.mse_loss(logit, rot, reduction="none").to(dtype)
    loss = loss.mean(1)
    return {"rotation": loss.mean()}


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
        rotation_pooling_gaussian_spread=0.01,
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
        self.rotation_pooling_gaussian_spread = rotation_pooling_gaussian_spread
        self.regress_position_offset = regress_position_offset
        self.points_supervised_for_offset = points_supervised_for_offset
        task_file = Path(__file__).parent.parent / "tasks/106_tasks.csv"
        with open(task_file) as fid:
            self.tasks = [t.strip() for t in fid.readlines()]

    def compute_loss(
        self, pred: Dict[str, torch.Tensor], sample: Sample, model=None
    ) -> Dict[str, torch.Tensor]:
        device = pred["position"].device
        padding_mask = sample["padding_mask"].to(device)
        gt_action = sample["action"].to(device)[padding_mask]

        losses = {}

        self._compute_position_loss(pred, gt_action[:, :3], losses)

        # TODO Clean this up when predicting rotation
        # if not self.position_prediction_only:
        #     # TODO Find a way not to require passing model to loss computation
        #     #  It's easy if we don't pool local features but not straightforward otherwise
        #
        #     if self.position_loss in ["ce", "bce"] and model is not None:
        #         ghost_pcd_features = einops.rearrange(pred["ghost_pcd_features"], "npts b c -> b npts c")
        #
        #         if self.rotation_pooling_gaussian_spread == 0:
        #             gt_indices = l2_gt.min(dim=-1).indices
        #             features = ghost_pcd_features[torch.arange(len(gt_indices)), gt_indices]
        #         else:
        #             weights = torch.softmax(-l2_gt / self.rotation_pooling_gaussian_spread, dim=-1).detach()
        #             features = einops.einsum(ghost_pcd_features, weights, "b npts c, b npts -> b c")
        #
        #         if type(model) == nn.DataParallel:
        #             gripper_state = model.module.prediction_head.gripper_state_predictor(features)
        #         else:
        #             gripper_state = model.position_prediction.gripper_state_predictor(features)
        #         rotation = normalise_quat(gripper_state[:, :4])
        #         gripper = torch.sigmoid(gripper_state[:, 4:])
        #
        #         losses.update(compute_rotation_loss(rotation, outputs[:, 3:7]))
        #         losses["gripper"] = F.mse_loss(gripper, outputs[:, 7:8])
        #
        #     else:
        #         losses.update(compute_rotation_loss(pred["rotation"], outputs[:, 3:7]))
        #         losses["gripper"] = F.mse_loss(pred["gripper"], outputs[:, 7:8])
        #
        #     if pred["task"] is not None:
        #         task = torch.Tensor([self.tasks.index(t) for t in sample["task"]])
        #         task = task.to(device).long()
        #         losses["task"] = F.cross_entropy(pred["task"], task)
        #
        #     losses["rotation"] *= self.rotation_loss_coeff
        #     losses["gripper"] *= self.gripper_loss_coeff

        return losses

    def _compute_position_loss(self, pred, gt_position, losses):
        if self.position_loss == "mse":
            # Only used for original HiveFormer
            losses["position"] = F.mse_loss(pred["position"], gt_position)

        elif self.position_loss == "ce":
            losses["position"] = []

            # Select a normalized Gaussian ball around the ground-truth as a proxy label
            coarse_l2 = ((pred["coarse_ghost_pcd"] - gt_position.unsqueeze(-1)) ** 2).sum(1).sqrt()
            coarse_label = torch.softmax(-coarse_l2 / self.ground_truth_gaussian_spread, dim=-1).detach()
            if pred.get("fine_ghost_pcd") is not None:
                fine_l2 = ((pred["fine_ghost_pcd"] - gt_position.unsqueeze(-1)) ** 2).sum(1).sqrt()
                fine_label = torch.softmax(-fine_l2 / self.ground_truth_gaussian_spread, dim=-1).detach()

            loss_layers = range(
                len(pred["coarse_ghost_pcd_masks"])) if self.compute_loss_at_all_layers else [-1]

            for i in loss_layers:
                losses["position"].append(F.cross_entropy(
                    pred["coarse_ghost_pcd_masks"][i], coarse_label, label_smoothing=self.label_smoothing))
                if pred.get("fine_ghost_pcd") is not None:
                    losses["position"].append(F.cross_entropy(
                        pred["fine_ghost_pcd_masks"][i], fine_label, label_smoothing=self.label_smoothing))

            losses["position"] = torch.stack(losses["position"]).mean()

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
                losses["position_offset"] *= self.position_offset_loss_coeff
                losses["position"] += losses["position_offset"]

            # Clear gradient on pred["position"] to avoid a memory leak since we don't
            # use it in the loss
            pred["position"] = pred["position"].detach()

        losses["position"] *= self.position_loss_coeff

    def compute_metrics(
        self, pred: Dict[str, torch.Tensor], sample: Sample
    ) -> Dict[str, torch.Tensor]:
        device = pred["position"].device
        dtype = pred["position"].dtype
        padding_mask = sample["padding_mask"].to(device)
        outputs = sample["action"].to(device)[padding_mask]

        metrics = {}

        l2 = ((pred["position"] - outputs[:, :3]) ** 2).sum(1).sqrt()
        metrics["position_l2"] = l2.to(dtype).mean()
        metrics["position_l2<0.01"] = (l2 < 0.01).to(dtype).mean()
        metrics["position_l2<0.02"] = (l2 < 0.02).to(dtype).mean()
        metrics["position_l2<0.04"] = (l2 < 0.04).to(dtype).mean()
        metrics["position_l2<0.08"] = (l2 < 0.08).to(dtype).mean()
        metrics["position_l2<0.16"] = (l2 < 0.16).to(dtype).mean()

        if not self.position_prediction_only:
            pred_gripper = (pred["gripper"] > 0.5).squeeze(-1)
            true_gripper = outputs[:, 7].bool()
            acc = pred_gripper == true_gripper
            metrics["gripper"] = acc.to(dtype).mean()

            metrics.update(compute_rotation_metrics(pred["rotation"], outputs[:, 3:7]))

            if pred["task"] is not None:
                task = torch.Tensor([self.tasks.index(t) for t in sample["task"]])
                task = task.to(device).long()
                acc = task == pred["task"].argmax(1)
                metrics["task"] = acc.to(dtype).mean()

        return metrics
