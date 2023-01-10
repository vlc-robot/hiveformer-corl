import torch
import torch.nn as nn
from torch import Tensor

from utils import Output


def norm_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / torch.linalg.norm(tensor, ord=2, dim=-1, keepdim=True)


class Baseline(nn.Module):
    def __init__(
        self,
        num_cams: int = 3,
        num_tasks: int = 106,
    ):
        super(Baseline, self).__init__()
        self._num_cams = num_cams
        self._num_tasks = num_tasks
        self.dummy = nn.Linear(1, 1)

    def compute_action(self, pred: Output) -> torch.Tensor:
        rotation = norm_tensor(pred["rotation"])
        return torch.cat([pred["position"], rotation, pred["gripper"]], dim=1)

    def forward(
            self,
            rgb_obs: Tensor,
            pc_obs: Tensor,
            padding_mask: Tensor,
            instruction: Tensor,
            gripper: Tensor,
    ) -> Output:
        bs, t, n_cam, c, h, w = rgb_obs.shape[:3]

        # TODO Implement baseline

        return {
            "position": torch.zeros(t, 3),
            "rotation": torch.zeros(t, 4),
            "gripper": torch.zeros(t, 1),
            "attention": torch.zeros(t, n_cam, 1, h, w),
            "task": torch.zeros(bs, self._num_tasks),
        }
