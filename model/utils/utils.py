import numpy as np
import einops
import torch


def normalise_quat(x: torch.Tensor):
    return x / x.square().sum(dim=-1).sqrt().unsqueeze(-1)


def norm_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / torch.linalg.norm(tensor, ord=2, dim=-1, keepdim=True)


def sample_ghost_points_grid(bounds, num_points_per_dim=10):
    x_ = np.linspace(bounds[0][0], bounds[1][0], num_points_per_dim)
    y_ = np.linspace(bounds[0][1], bounds[1][1], num_points_per_dim)
    z_ = np.linspace(bounds[0][2], bounds[1][2], num_points_per_dim)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    ghost_points = einops.rearrange(np.stack([x, y, z]), "n x y z -> (x y z) n")
    return ghost_points


def sample_ghost_points_uniform_cube(bounds, num_points=1000):
    x = np.random.uniform(bounds[0][0], bounds[1][0], num_points)
    y = np.random.uniform(bounds[0][1], bounds[1][1], num_points)
    z = np.random.uniform(bounds[0][2], bounds[1][2], num_points)
    ghost_points = np.stack([x, y, z], axis=1)
    return ghost_points


def sample_ghost_points_uniform_sphere(center, radius, bounds, num_points=1000):
    """Sample points uniformly within a sphere through rejection sampling."""
    ghost_points = np.empty((0, 3))
    while ghost_points.shape[0] < num_points:
        points = sample_ghost_points_uniform_cube(bounds, num_points)
        l2 = np.linalg.norm(points - center, axis=1)
        ghost_points = np.concatenate([ghost_points, points[l2 < radius]])
    ghost_points = ghost_points[:num_points]
    return ghost_points
