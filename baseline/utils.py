import numpy as np
import einops


def sample_ghost_points(gripper_loc_bounds, num_points_per_dim=10):
    x_ = np.linspace(gripper_loc_bounds[0][0], gripper_loc_bounds[1][0], num_points_per_dim)
    y_ = np.linspace(gripper_loc_bounds[0][1], gripper_loc_bounds[1][1], num_points_per_dim)
    z_ = np.linspace(gripper_loc_bounds[0][2], gripper_loc_bounds[1][2], num_points_per_dim)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    ghost_points = einops.rearrange(np.stack([x, y, z]), "n x y z -> (x y z) n")
    return ghost_points