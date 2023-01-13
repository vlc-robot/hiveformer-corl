import os
import cv2
import open3d
import shutil
import einops
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from utils import Output
from video_utils import get_point_cloud_images


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
            pcd_obs: Tensor,
            padding_mask: Tensor,
            instruction: Tensor,
            gripper: Tensor,
    ) -> Output:
        bs, ts, n_cam, c, h, w = rgb_obs.shape

        # DEBUG
        vis = open3d.visualization.Visualizer()
        vis.create_window()
        dir_path = "images"
        shutil.rmtree(dir_path, ignore_errors=True)
        os.makedirs(dir_path)
        for t in range(ts):
            pcd_imgs = get_point_cloud_images(
                vis,
                rgb_obs[0, t].cpu().numpy(),
                pcd_obs[0, t].cpu().numpy(),
            )
            for c, img in enumerate(pcd_imgs):
                cv2.imwrite(f"{dir_path}/pcd_t{t}_view{c}.png", img[:, :, ::-1])
        vis.destroy_window()

        return {
            "position": torch.zeros(ts, 3),
            "rotation": torch.zeros(ts, 4),
            "gripper": torch.zeros(ts, 1),
            "attention": torch.zeros(ts, n_cam, 1, h, w),
            "task": torch.zeros(bs, self._num_tasks),
        }
