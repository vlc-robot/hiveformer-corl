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
        rotation_loss_coeff=1.0,
        gripper_loss_coeff=1.0,
        rotation_pooling_gaussian_spread=0.01,
    ):
        assert position_loss in ["mse", "ce", "bce"]
        self.position_loss = position_loss
        self.position_prediction_only = position_prediction_only
        self.compute_loss_at_all_layers = compute_loss_at_all_layers
        self.ground_truth_gaussian_spread = ground_truth_gaussian_spread
        self.label_smoothing = label_smoothing
        self.position_loss_coeff = position_loss_coeff
        self.rotation_loss_coeff = rotation_loss_coeff
        self.gripper_loss_coeff = gripper_loss_coeff
        self.rotation_pooling_gaussian_spread = rotation_pooling_gaussian_spread
        task_file = Path(__file__).parent.parent / "tasks/106_tasks.csv"
        with open(task_file) as fid:
            self.tasks = [t.strip() for t in fid.readlines()]

    def compute_loss(
        self, pred: Dict[str, torch.Tensor], sample: Sample, model=None
    ) -> Dict[str, torch.Tensor]:
        device = pred["position"].device
        padding_mask = sample["padding_mask"].to(device)
        outputs = sample["action"].to(device)[padding_mask]

        losses = {}

        if self.position_loss in ["ce", "bce"]:
            # Select a normalized Gaussian ball around the ground-truth as a proxy label
            gt = outputs[:, :3]

            l2_gt = ((pred["all_pcd"] - gt.unsqueeze(-1)) ** 2).sum(1).sqrt()
            label = torch.softmax(-l2_gt / self.ground_truth_gaussian_spread, dim=-1).detach()

            pred_masks = pred["all_masks"][-1]

            if self.position_loss == "ce":
                losses["position"] = F.cross_entropy(
                    pred_masks, label, label_smoothing=self.label_smoothing)

            elif self.position_loss == "bce":
                pos_weight = label.numel() / label.sum()
                losses["position"] = F.binary_cross_entropy_with_logits(
                    pred_masks, label, pos_weight=pos_weight)

            # Compute loss at intermediate layers
            # TODO Doesn't seem to help, try this again once get model working
            if self.compute_loss_at_all_layers:
                raise NotImplementedError

            # Clear gradient on pred["position"] to avoid a memory leak since we don't
            # use it in the loss
            pred["position"] = pred["position"].detach()

        elif self.position_loss == "mse":
            losses["position"] = F.mse_loss(pred["position"], outputs[:, :3])

        losses["position"] *= self.position_loss_coeff

        if not self.position_prediction_only:
            # TODO Find a way not to require passing model to loss computation
            #  It's easy if we don't pool local features but not straightforward otherwise

            if self.position_loss in ["ce", "bce"] and model is not None:
                if self.rotation_pooling_gaussian_spread == 0:
                    gt_indices = l2_gt.min(dim=-1).indices
                    features = pred["all_features"][torch.arange(len(gt_indices)), gt_indices]
                else:
                    weights = torch.softmax(-l2_gt / self.rotation_pooling_gaussian_spread, dim=-1).detach()
                    features = einops.einsum(pred["all_features"], weights, "b npts c, b npts -> b c")

                if type(model) == nn.DataParallel:
                    gripper_state = model.module.prediction_head.gripper_state_predictor(features)
                else:
                    gripper_state = model.position_prediction.gripper_state_predictor(features)
                rotation = normalise_quat(gripper_state[:, :4])
                gripper = torch.sigmoid(gripper_state[:, 4:])

                losses.update(compute_rotation_loss(rotation, outputs[:, 3:7]))
                losses["gripper"] = F.mse_loss(gripper, outputs[:, 7:8])

            else:
                losses.update(compute_rotation_loss(pred["rotation"], outputs[:, 3:7]))
                losses["gripper"] = F.mse_loss(pred["gripper"], outputs[:, 7:8])

            if pred["task"] is not None:
                task = torch.Tensor([self.tasks.index(t) for t in sample["task"]])
                task = task.to(device).long()
                losses["task"] = F.cross_entropy(pred["task"], task)

            losses["rotation"] *= self.rotation_loss_coeff
            losses["gripper"] *= self.gripper_loss_coeff

        return losses

    def compute_metrics(
        self, pred: Dict[str, torch.Tensor], sample: Sample
    ) -> Dict[str, torch.Tensor]:
        device = pred["position"].device
        dtype = pred["position"].dtype
        padding_mask = sample["padding_mask"].to(device)
        outputs = sample["action"].to(device)[padding_mask]

        metrics = {}

        l2 = ((pred["position"] - outputs[:, :3]) ** 2).sum(1).sqrt()
        acc = l2 < 0.01
        metrics["position_l2"] = l2.to(dtype).mean()
        metrics["position_l2<0.01"] = acc.to(dtype).mean()

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
