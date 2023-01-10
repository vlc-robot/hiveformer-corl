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


def norm_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / torch.linalg.norm(tensor, ord=2, dim=-1, keepdim=True)


def save_point_cloud_images(rgb_obs: np.array, pcd_obs: np.array, dir_path: str = "images"):
    shutil.rmtree(dir_path, ignore_errors=True)
    os.makedirs(dir_path)

    def save_point_cloud_image(opcds, vis, path):
        for opcd in opcds:
            vis.add_geometry(opcd)
            vis.update_geometry(opcd)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(path)
        for opcd in opcds:
            vis.remove_geometry(opcd)

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    for t in range(rgb_obs.shape[1]):
        opcds = []

        for cam in range(rgb_obs.shape[2]):
            rgb = einops.rearrange(rgb_obs[0, t, cam, :3], "c h w -> (h w) c")
            pcd = einops.rearrange(pcd_obs[0, t, cam], "c h w -> (h w) c")

            opcd = open3d.geometry.PointCloud()
            opcd.points = open3d.utility.Vector3dVector(pcd)
            opcd.colors = open3d.utility.Vector3dVector(rgb)
            opcds.append(opcd)

            save_point_cloud_image([opcd], vis, f"images/pcd_t{t}_cam{cam}.png")
            cv2.imwrite(
                f"images/rgb_t{t}_cam{cam}.png",
                einops.rearrange(rgb_obs[0, t, cam, :3][::-1] * 255.0, "c h w -> h w c")
            )

        save_point_cloud_image(opcds, vis, f"images/pcd_t{t}_allcams.png")

    vis.destroy_window()


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
        bs, t, n_cam, c, h, w = rgb_obs.shape

        # DEBUG
        save_point_cloud_images(rgb_obs.cpu().numpy(), pcd_obs.cpu().numpy())

        return {
            "position": torch.zeros(t, 3),
            "rotation": torch.zeros(t, 4),
            "gripper": torch.zeros(t, 1),
            "attention": torch.zeros(t, n_cam, 1, h, w),
            "task": torch.zeros(bs, self._num_tasks),
        }
