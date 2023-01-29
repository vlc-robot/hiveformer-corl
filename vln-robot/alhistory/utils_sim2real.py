from abc import abstractmethod, ABC
import random
import sys
import itertools
import pickle
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple, Literal, Union, Any
from typing_extensions import TypedDict
from pathlib import Path
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import gym
from scipy.spatial.transform import Rotation as R
import einops
from rotations import compute_rotation_matrix_from_quaternions
from utils_instructions import MicToText, TextToToken
from structures import (
    Observation,
    Misc,
    Demo,
    Output,
    APPLY_CAMERAS,
    PointCloudToken,
    InstructionMode,
    GripperPose,
    Instructions,
    RotType,
    RotMode,
)


def get_gripper_state(state: Union[bool, List[bool]]):
    """
    Fix gripper state
    """
    if isinstance(state, list):
        state = state[0]
    state = not state
    return state


def load_episodes() -> Dict[str, Any]:
    with open(Path(__file__).parent / "episodes.json") as fid:
        return json.load(fid)


def get_max_episode_length(tasks: Tuple[str, ...], variations: Tuple[int, ...]) -> int:
    max_episode_length = 0
    max_eps_dict = load_episodes()["max_episode_length"]

    for task, var in itertools.product(tasks, variations):
        if max_eps_dict[task] > max_episode_length:
            max_episode_length = max_eps_dict[task]

    return max_episode_length


def task_file_to_task_class(task_file):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class


class MotionPlannerError(Exception):
    """When the motion planner is not able to execute an action"""


class Model(TypedDict):
    model: nn.Module
    t: Dict[str, torch.Tensor]
    z: Dict[str, torch.Tensor]


def plot_attention(
    attentions: torch.Tensor, rgbs: torch.Tensor, pcds: torch.Tensor, dest: Path
) -> plt.Figure:
    attentions = attentions.detach().cpu()
    rgbs = rgbs.detach().cpu()
    pcds = pcds.detach().cpu()

    ep_dir = dest.parent
    ep_dir.mkdir(exist_ok=True, parents=True)
    name = dest.stem
    ext = dest.suffix

    # plt.figure(figsize=(10, 8))
    num_cameras = len(attentions)
    for i, (a, rgb, pcd) in enumerate(zip(attentions, rgbs, pcds)):
        # plt.subplot(num_cameras, 4, i * 4 + 1)
        plt.imshow(a.permute(1, 2, 0).log())
        plt.axis("off")
        # plt.colorbar()
        plt.savefig(ep_dir / f"{name}-{i}-attn{ext}", bbox_inches="tight")
        plt.tight_layout()
        plt.clf()

        # plt.subplot(num_cameras, 4, i * 4 + 2)
        # plt.imshow(a.permute(1, 2, 0))
        # plt.axis('off')
        # plt.colorbar()
        # plt.tight_layout()
        # plt.clf()

        # plt.subplot(num_cameras, 4, i * 4 + 3)
        plt.imshow(((rgb + 1) / 2).permute(1, 2, 0))
        plt.axis("off")
        plt.savefig(ep_dir / f"{name}-{i}-rgb{ext}", bbox_inches="tight")
        plt.tight_layout()
        plt.clf()

        pcd_norm = (pcd - pcd.min(0).values) / (pcd.max(0).values - pcd.min(0).values)
        # plt.subplot(num_cameras, 4, i * 4 + 4)
        plt.imshow(pcd_norm.permute(1, 2, 0))
        plt.axis("off")
        plt.savefig(ep_dir / f"{name}-{i}-pcd{ext}", bbox_inches="tight")
        plt.tight_layout()
        plt.clf()

    return plt.gcf()


def obs_to_attn(obs, camera: str) -> Tuple[int, int]:
    extrinsics_44 = torch.from_numpy(obs.misc[f"{camera}_camera_extrinsics"]).float()
    extrinsics_44 = torch.linalg.inv(extrinsics_44)
    intrinsics_33 = torch.from_numpy(obs.misc[f"{camera}_camera_intrinsics"]).float()
    intrinsics_34 = F.pad(intrinsics_33, (0, 1, 0, 0))
    gripper_pos_3 = torch.from_numpy(obs.gripper_pose[:3]).float()
    gripper_pos_41 = F.pad(gripper_pos_3, (0, 1), value=1).unsqueeze(1)
    points_cam_41 = extrinsics_44 @ gripper_pos_41

    proj_31 = intrinsics_34 @ points_cam_41
    proj_3 = proj_31.float().squeeze(1)
    u = int((proj_3[0] / proj_3[2]).round())
    v = int((proj_3[1] / proj_3[2]).round())

    return u, v


# --------------------------------------------------------------------------------
# Sim2Real Environment & Related Functions
# --------------------------------------------------------------------------------
def get_observation(
    name: str, obs_mujoco: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fov = float(obs_mujoco[f"info_{name}_camera"]["fovy"])
    euler = torch.Tensor(obs_mujoco[f"info_{name}_camera"]["euler"])
    pos = torch.Tensor(obs_mujoco[f"info_{name}_camera"]["pos"])
    h, w = 128, 128
    extrinsics = get_extrinsics(pos, euler)
    intrinsics2 = get_intrinsics(fov, h, w)

    rgb = obs_mujoco[f"rgb_{name}_camera"]
    rgb = square(rgb)
    rgb2 = rgb_resize(rgb, h, w)

    depth = obs_mujoco[f"depth_{name}_camera"]
    depth2 = depth_resize(depth, extrinsics, fov, h, w)
    pcd2 = depth_to_pcd(depth2, extrinsics, intrinsics2)

    return rgb2, pcd2, extrinsics, intrinsics2


def get_rgb_pcd_gripper_from_obs(
    obs_mujoco: Dict,
    gripper_pose: GripperPose,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return rgb, pcd, and gripper from a given observation
    :param obs: an Observation from the env
    :return: rgb, pcd, gripper
    """
    pose = np.concatenate([obs_mujoco["gripper_pos"], obs_mujoco["gripper_quat"]])
    joint_velocities = obs_mujoco.get("arms_joint_vel", None)

    # FIXME add wrist camera parameters
    (
        wrist_rgb,
        wrist_pcd,
        wrist_extrinsics,
        wrist_intrinsics,
    ) = get_observation("charlie", obs_mujoco)
    ls_rgb, ls_pcd, ls_extrinsics, ls_intrinsics = get_observation("charlie", obs_mujoco)
    rs_rgb, rs_pcd, rs_extrinsics, rs_intrinsics = get_observation("bravo", obs_mujoco)

    gripper_open = get_gripper_state(obs_mujoco["gripper_state"])
    # # DEBUG
    # gripper_open = obs_mujoco["gripper_state"] == -2

    misc: Misc = {
        "wrist_camera_extrinsics": wrist_extrinsics,
        "wrist_camera_intrinsics": wrist_intrinsics[:, :3],
        "left_shoulder_camera_extrinsics": ls_extrinsics,
        "left_shoulder_camera_intrinsics": ls_intrinsics[:, :3],
        "right_shoulder_camera_extrinsics": rs_extrinsics,
        "right_shoulder_camera_intrinsics": rs_intrinsics[:, :3],
    }

    obs = Observation(
        gripper_open=gripper_open,
        gripper_pose=pose,
        joint_velocities=joint_velocities,
        wrist_rgb=wrist_rgb,
        wrist_pcd=wrist_pcd,
        left_shoulder_rgb=ls_rgb,
        left_shoulder_pcd=ls_pcd,
        right_shoulder_rgb=rs_rgb,
        right_shoulder_pcd=rs_pcd,
        misc=misc,
        loading=False,
    )

    state_dict, gripper = get_obs_action(obs)
    state = transform(state_dict)
    state = einops.rearrange(
        state,
        "(m n ch) h w -> n m ch h w",
        ch=3,
        n=len(APPLY_CAMERAS),
        m=2,
    )
    rgb = state[:, 0].unsqueeze(0)  # 1, N, C, H, W
    pcd = state[:, 1].unsqueeze(0)  # 1, N, C, H, W
    gripper = gripper.unsqueeze(0)  # 1, D

    if "attn" in gripper_pose:
        attns = torch.Tensor([])
        for cam in APPLY_CAMERAS:
            u, v = obs_to_attn(obs, cam)
            attn = torch.zeros((1, 1, 1, 128, 128))
            if not (u < 0 or u > 127 or v < 0 or v > 127):
                attn[0, 0, 0, v, u] = 1
            attns = torch.cat([attns, attn], 1)
        rgb = torch.cat([rgb, attns], 2)

    return rgb, pcd, gripper


def get_obs_action(obs):
    """
    Fetch the desired state and action based on the provided demo.
        :param obs: incoming obs
        :return: required observation and action list
    """
    # fetch state
    state_dict = {"rgb": [], "pc": []}
    for cam in APPLY_CAMERAS:
        rgb = getattr(obs, "{}_rgb".format(cam))
        state_dict["rgb"].append(rgb)

        pcd = getattr(obs, "{}_pcd".format(cam))
        state_dict["pc"].append(pcd)

    # fetch action
    action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
    return state_dict, torch.from_numpy(action).float()


def get_obs_action_from_demo(demo: Demo):
    """
    Fetch the desired state and action based on the provided demo.
        :param demo: fetch each demo and save key-point observations
        :param normalise_rgb: normalise rgb to (-1, 1)
        :return: a list of obs and action
    """
    key_frame = keypoint_discovery(demo)  # type: ignore
    key_frame.insert(0, 0)
    state_ls = []
    action_ls = []
    for f in key_frame:
        state, action = get_obs_action(demo[f])
        state = transform(state)
        state_ls.append(state.unsqueeze(0))
        action_ls.append(action.unsqueeze(0))
    return state_ls, action_ls


def load_instructions(instructions: Optional[Path]) -> Optional[Instructions]:
    if instructions is not None:
        with open(instructions, "rb") as fid:
            print(instructions)
            return pickle.load(fid)
    return None


class Sim2RealEnv:
    def __init__(
        self,
        model: Model,
        instructions: Optional[Path],
        gripper_pose: GripperPose = "none",
        instr_mode: InstructionMode = "precompute",
    ):
        self.gripper_pose: GripperPose = gripper_pose
        self._model = model
        self._instr_mode = instr_mode

        if self._instr_mode == "precompute":
            if instructions is None:
                raise ValueError("Please provide path to precomputed instructions")
            self._instructions: Optional[Instructions] = load_instructions(instructions)
        elif self._instr_mode == "mic":
            self._instructions = None
            self._mic_to_text = MicToText()
            self._text_to_token = TextToToken()
        elif self._instr_mode == "text":
            self._instructions = None
            self._text_to_token = TextToToken()
        else:
            raise ValueError(f"Unexpected instr mode {self._instr_mode}.")

        self._env_name = ""
        self._env = None

    @property
    def env(self) -> gym.Env:
        if self._env is None:
            raise RuntimeError("Please set the environment first")
        return self._env

    def set_env(self, env_name: str) -> None:
        if self._env_name == env_name and self._env is not None:
            return
        self._env_name = env_name
        self._env = gym.make(self._env_name)

    def evaluate(
        self,
        task_str: str,
        env_str: str,
        max_episodes: int,
        variation: int,
        num_demos: int,
        device,
        log_dir: Optional[Path],
        offset: int = 0,
        save_attn: bool = False,
        seed: int = 0,
    ):
        """
        Evaluate the policy network on the desired demo or test environments
            :param task_type: type of task to evaluate
            :param max_episodes: maximum episodes to finish a task
            :param num_demos: number of test demos for evaluation
            :param model: the policy network
            :param demos: whether to use the saved demos
            :return: success rate
        """
        success_rate = 0.0
        num_trials = 0.0

        self.set_env(env_str)

        with torch.no_grad():
            for demo_id in range(offset, num_demos):

                rgbs = torch.Tensor([]).to(device)
                pcds = torch.Tensor([]).to(device)
                grippers = torch.Tensor([]).to(device)

                # reset a new demo or a defined demo in the demo list
                self.env.seed(seed + demo_id)
                obs = self.env.reset()

                reward = None
                instr_embed = self._get_instruction(task_str, variation).to(device)

                for step_id in range(max_episodes):
                    images = {k: v for k, v in obs.items() if "rgb_" in k}
                    if log_dir is not None:
                        ep_dir = log_dir / f"episode{demo_id}"
                        ep_dir.mkdir(exist_ok=True, parents=True)
                        for cam, im in images.items():
                            cam_dir = ep_dir / cam
                            cam_dir.mkdir(exist_ok=True, parents=True)
                            Image.fromarray(im).save(cam_dir / f"{step_id}.png")

                    # fetch the current observation, and predict one action
                    rgb, pcd, gripper = get_rgb_pcd_gripper_from_obs(
                        obs, self.gripper_pose
                    )

                    rgb = rgb.to(device)
                    pcd = pcd.to(device)
                    gripper = gripper.to(device)

                    rgbs = torch.cat([rgbs, rgb.unsqueeze(1)], dim=1)
                    pcds = torch.cat([pcds, pcd.unsqueeze(1)], dim=1)
                    grippers = torch.cat([grippers, gripper.unsqueeze(1)], dim=1)

                    padding_mask = torch.ones_like(rgbs[:, :, 0, 0, 0, 0]).bool()
                    output: Output = self._model["model"](
                        rgbs, pcds, padding_mask, None, None, instr_embed, grippers, None
                    )

                    # update the observation based on the predicted action
                    gripper_open = bool((output["gripper"] > 0)[-1])
                    # DEBUG
                    # gripper_open = step_id % 2 == 1

                    position = output["position"].detach().cpu().numpy()[-1]
                    rotation = output["rotation"].detach().cpu().numpy()[-1]
                    # print(f"Position: {position} - Gripper: {gripper_open}")
                    if (
                        log_dir is not None
                        and save_attn
                        and output["attention"] is not None
                    ):
                        ep_dir = log_dir / f"episode{demo_id}"
                        plot_attention(
                            output["attention"][-1],
                            rgbs[0][-1, :, :3],
                            pcds[0][-1].view(3, 3, 128, 128),
                            ep_dir / f"attn_{step_id}.png",
                        )

                    # import ipdb
                    # ipdb.set_trace()

                    obs, reward, terminate, other_obs = self.move(
                        position,
                        rotation,
                        gripper_open,
                    )
                    # if reward == 1:
                    #     success_rate += 1 / num_demos
                    #     break

                    if terminate:
                        print("The episode has terminated!")

                ask = input("S=Success. F=Failure. I=Ignore. Q=Quit")
                if ask == "S":
                    num_trials += 1
                    success_rate += 1 / num_trials
                elif ask == "F":
                    num_trials += 1
                elif ask == "Q":
                    sys.exit(0)

                print(
                    task_str,
                    "Reward",
                    reward,
                    "Variation",
                    variation,
                    "Step",
                    demo_id,
                    "SR: %.2f" % (success_rate * 100),
                )

        return success_rate

    def move(self, position: np.ndarray, rotation: np.ndarray, gripper_open: np.ndarray):
        obs, reward, terminate, other_obs = self.env.move(
            position,
            rotation,
            gripper_open,
        )
        return obs, reward, terminate, other_obs

    def _get_instruction(self, task_str: str, variation: int) -> torch.Tensor:
        if self._instr_mode == "precompute":
            if self._instructions is None:
                raise RuntimeError()
            instrs = self._instructions[task_str][variation]
            instr = random.choice(instrs)
            instr = instr.unsqueeze(0)
            print(instr.shape)
            return instr
        if self._instr_mode == "text":
            text = input("Provide instructions (blank for exit)\n")
            if text == "":
                sys.exit(0)
            instr = self._text_to_token(text)
            # import pdb

            # pdb.set_trace()
            return instr
        if self._instr_mode == "mic":
            text = self._mic_to_text()
            print("Decoded instructions:")
            print(text)
            instr = self._text_to_token(text)
            return instr
        raise RuntimeError(f"Unexpected {self._instr_mode}")


# --------------------------------------------------------------------------------
# General Functions
# --------------------------------------------------------------------------------
def transform(obs_dict, scale_size=(0.75, 1.25)):
    apply_depth = len(obs_dict.get("depth", [])) > 0
    apply_pc = len(obs_dict["pc"]) > 0
    num_cams = len(obs_dict["rgb"])

    obs_rgb = []
    obs_depth = []
    obs_pc = []
    for i in range(num_cams):
        rgb = torch.tensor(obs_dict["rgb"][i]).float().permute(2, 0, 1)
        depth = (
            torch.tensor(obs_dict["depth"][i]).float().permute(2, 0, 1)
            if apply_depth
            else None
        )
        pc = (
            torch.tensor(obs_dict["pc"][i]).float().permute(2, 0, 1) if apply_pc else None
        )

        # normalise to [-1, 1]
        rgb = rgb / 255.0
        rgb = 2 * (rgb - 0.5)

        obs_rgb += [rgb.float()]
        if depth is not None:
            obs_depth += [depth.float()]
        if pc is not None:
            obs_pc += [pc.float()]
    obs = obs_rgb + obs_depth + obs_pc
    return torch.cat(obs, dim=0)


def get_intrinsics(fov, height, width):
    focal_scaling = (1.0 / np.tan(np.deg2rad(fov) / 2)) * height / 2.0
    focal = np.diag([focal_scaling, focal_scaling, 1.0, 0])[0:3, :]

    # Image matrix (3x3).
    image = np.eye(3)
    image[0, 2] = (width - 1) / 2.0
    image[1, 2] = (height - 1) / 2.0
    intrinsics = image @ focal
    return intrinsics


def get_extrinsics(xyz, euler):
    quat = R.from_euler("xyz", euler, degrees=False).as_quat()
    quat = torch.from_numpy(quat)
    rot = compute_rotation_matrix_from_quaternions(quat.unsqueeze(0)).squeeze(0)
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rot.numpy()
    extrinsics[:3, 3] = xyz
    return extrinsics


def world_coord_to_pixel(xyz, extrinsics, intrinsics, width):
    N = xyz.shape[0]
    xyzw = np.concatenate([xyz, np.ones((N, 1))], 1)
    xyzw = np.transpose(xyzw, (1, 0))
    pos_2d = intrinsics @ np.linalg.inv(extrinsics) @ xyzw
    pos_2d = np.transpose(pos_2d, (1, 0))
    depth = pos_2d[:, 2]
    u = width - pos_2d[:, 0] / depth
    v = pos_2d[:, 1] / depth
    return u, v, depth


def world_coord_to_cam_coord(position, extrinsics):
    N = position.shape[0]
    pos_2d_homo = np.concatenate([position, np.ones((N, 1))], 1)
    pos_2d_homo = np.transpose(pos_2d_homo, (1, 0))
    pos_2d = np.linalg.inv(extrinsics) @ pos_2d_homo
    pos_2d = np.transpose(pos_2d, (1, 0))
    return pos_2d


def pixel_to_world_coord(pos_ijd, extrinsics, intrinsics):
    N = pos_ijd.shape[0]
    # N, 3 --> N, 4
    pos_ijdk = np.concatenate([pos_ijd, np.ones((N, 1))], 1)
    # N, 4 --> 4, N
    pos_ijdk = np.transpose(pos_ijdk, (1, 0))

    cam_proj_mat = intrinsics @ np.linalg.inv(extrinsics)
    cam_proj_mat_homo = np.concatenate([cam_proj_mat, [np.array([0, 0, 0, 1])]])
    cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]

    transformed_coords_vector = cam_proj_mat_inv @ pos_ijdk
    transformed_coords_vector = np.transpose(transformed_coords_vector, (1, 0))
    return transformed_coords_vector


def depth_to_pcd(depth, extrinsics, intrinsics):
    h, w = depth.shape
    v, u = np.unravel_index(np.arange(h * w), (h, w))
    depth_flat = -depth.ravel()
    U = (w - u) * depth_flat
    V = v * depth_flat
    pos_2d = np.stack([U, V, depth_flat], 1)
    world_coords = pixel_to_world_coord(pos_2d, extrinsics, intrinsics)
    return world_coords.reshape(h, w, 3)


def square(im: np.ndarray) -> np.ndarray:
    h, w = im.shape[:2]
    if h == w:
        return im
    elif h > w:
        dy = (h - w) // 2
        return im[dy:-dy, :]
    else:
        dx = (w - h) // 2
        return im[:, dx:-dx]


def rgb_resize(im: np.ndarray, h: int, w: int) -> np.ndarray:
    pil_im = Image.fromarray(im)
    pil_im = pil_im.resize((w, h), Image.BILINEAR)
    return np.asarray(pil_im)


def depth_resize(
    depth: np.ndarray, extrinsics: np.ndarray, fov: float, h: int, w: int
) -> np.ndarray:
    depth = square(depth)
    orig_h, orig_w = depth.shape

    # get a pcd
    intrinsics = get_intrinsics(fov, orig_h, orig_w)
    pcd = depth_to_pcd(depth, extrinsics, intrinsics)

    # and then get back to a depth with desired resolution
    intrinsics2 = get_intrinsics(fov, h, w)

    u, v, depth2 = world_coord_to_pixel(pcd.reshape(-1, 3), extrinsics, intrinsics2, w)
    U = np.int32(np.floor(u))
    V = np.int32(np.floor(v))
    mask = (U >= 0) & (V >= 0) & (U < h) & (V < w)
    depth3 = np.zeros((h, w))
    depth3[V[mask], U[mask]] = -depth2[mask]  # type: ignore
    return depth3


def norm_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / torch.linalg.norm(tensor, ord=2, dim=-1, keepdim=True)


class Rotation(ABC):
    def __init__(self, mode: RotMode):
        self.mode = mode
        self.num_dims = -1
        self.resolution: int = 90
        self.default_quat: torch.Tensor = torch.Tensor([0, 1, 0, 0])

    def compute_action(self, logit: torch.Tensor) -> torch.Tensor:
        """We force the rotation being normalized"""
        if self.mode == "mse":
            assert logit.shape[-1] == self.num_dims
            return self._cont_to_quat_cont(logit)
        elif self.mode == "ce":
            return self._disc_to_quat_cont(logit.argmax(2))
        elif self.mode == "none":
            default = self.default_quat.type_as(logit)
            N = logit.dim()
            default = default.view(*([1] * (N - 1)), 4)
            b = logit.shape[0]
            return default.repeat(b, *([1] * (N - 1)))
        else:
            raise RuntimeError(f"Unexpected mode {self.mode}")

    def compute_metrics(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
        reduction: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        pred = self.compute_action(pred)
        acc = (pred - true).abs().max(1).values < 0.05
        acc = acc.to(pred.dtype)

        if reduction == "mean":
            acc = acc.mean()
        return {"rotation": acc}

    def compute_loss(
        self, logit: torch.Tensor, rot: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        rot_ = -rot.clone()

        if self.mode == "mse":
            rot_loss = self._compute_mse_loss(logit, rot)
            rot_loss_ = self._compute_mse_loss(logit, rot_)
        elif self.mode == "ce":
            rot_loss = self._compute_ce_loss(logit, rot)
            rot_loss_ = self._compute_ce_loss(logit, rot_)
        elif self.mode == "none":
            return {}
        else:
            raise ValueError(f"Unexpected mode {self.mode}")

        losses = {}
        for (key, loss), (_, loss_) in zip(rot_loss.items(), rot_loss_.items()):
            select_mask = (loss < loss_).float()
            losses[key] = 4 * (select_mask * loss + (1 - select_mask) * loss_)

        return losses

    @abstractmethod
    def _disc_to_quat_cont(self, rotation: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def _cont_to_quat_cont(self, rotation: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def _quat_cont_to_disc(self, rotation: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def _quat_cont_to_cont(self, quat: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def _compute_ce_loss(
        self, logit: torch.Tensor, true_cont: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        true_disc = self._quat_cont_to_disc(true_cont)
        loss = torch.tensor(0.0).type_as(logit)
        for i in range(self.num_dims):
            loss += F.cross_entropy(logit[:, i], true_disc[:, i])
        loss /= self.num_dims
        return {f"rotation": loss}

    def _compute_mse_loss(
        self, logit: torch.Tensor, true: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        dtype = logit.dtype
        true = self._quat_cont_to_cont(true).to(dtype)
        loss = F.mse_loss(logit, true).to(dtype)
        return {f"rotation": loss}


class QuatRotation(Rotation):
    """
    Helper function when using quaternion for rotation prediction
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.num_dims: int = 4

    def _disc_to_quat_cont(self, rotation: torch.Tensor):
        rot = rotation / self.resolution * 2 - 1
        norm = torch.linalg.norm(rot, dim=1, keepdim=True)
        rot /= norm

        # default rotation where the gripper is vertical
        nan = rot.isnan().any(-1)
        if nan.any():
            N = rotation.dim()
            default = torch.Tensor(self.default_quat).type_as(rot)
            default = default.view(*([1] * (N - 1)), 4)
            rot[nan] = default

        return rot

    def _quat_cont_to_disc(self, quat: torch.Tensor) -> torch.Tensor:
        discrete = (quat + 1) / 2 * self.resolution
        # this should be unlikely (numerical instabilities)
        discrete[discrete < 0] = 0.0
        discrete[discrete >= self.resolution] = self.resolution - 1
        return discrete.float().floor().long()

    def _cont_to_quat_cont(self, cont: torch.Tensor) -> torch.Tensor:
        return norm_tensor(cont)

    def _quat_cont_to_cont(self, quat: torch.Tensor) -> torch.Tensor:
        return norm_tensor(quat)


class EulerRotation(Rotation):
    """
    Helper function when Euler representation

    Reference is XYZ
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.num_dims: int = 3
        self._ref = "XYZ"

    def _disc_to_quat_cont(self, rotation: torch.Tensor):
        # rot = ((rotation - 0.5) / self.resolution * 2 - 1) * math.pi
        rot = (rotation / self.resolution * 2 - 1) * math.pi
        quat = self._cont_to_quat_cont(rot)

        return quat

    def _quat_cont_to_disc(self, quat: torch.Tensor) -> torch.Tensor:
        # discrete = (quat + 1) / 2 * self.resolution
        euler = R.from_quat(quat.detach().cpu()).as_euler(self._ref)
        discrete = torch.from_numpy(euler).type_as(quat)
        discrete[discrete < -math.pi] += 2 * math.pi
        discrete[discrete > math.pi] -= 2 * math.pi
        discrete = (discrete + math.pi) / (2.0 * math.pi) * self.resolution

        # this should be unlikely (numerical instabilities)
        discrete[discrete < 0] = 0.0
        discrete[discrete >= self.resolution] = self.resolution - 1

        discrete = discrete.float().floor().long()

        return discrete

    def _cont_to_quat_cont(self, cont: torch.Tensor) -> torch.Tensor:
        np_quat = R.from_euler(self._ref, cont.detach().cpu()).as_quat()
        quat = torch.from_numpy(np_quat).type_as(cont)
        quat /= torch.linalg.norm(quat, dim=1, keepdim=True)

        # default rotation where the gripper is vertical
        nan = quat.isnan().any(-1)
        if nan.any():
            N = cont.dim()
            default = torch.Tensor(self.default_quat).type_as(cont)
            default = default.view(*([1] * (N - 1)), 4)
            quat[nan] = default

        return quat

    def _quat_cont_to_cont(self, quat: torch.Tensor) -> torch.Tensor:
        cont_np = R.from_quat(quat.cpu()).as_euler(self._ref)
        cont = torch.from_numpy(cont_np).type_as(quat)
        return cont


class ContRotation(Rotation):
    """
    Helper function when using quaternion for rotation prediction
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.num_dims: int = 6

    def _disc_to_quat_cont(self, rotation: torch.Tensor):
        rot = rotation * 2 / self.resolution - 1
        quat = self._cont_to_quat_cont(rot)
        return quat

    def _quat_cont_to_disc(self, quat: torch.Tensor) -> torch.Tensor:
        quat = quat.flatten(0, -2).view(-1, 4)
        so3 = compute_rotation_matrix_from_quaternions(quat)
        cont = so3[:, :2].flatten(-2).to(quat.dtype)

        discrete = (cont + 1) / 2 * self.resolution
        discrete[discrete < 0] = 0.0
        discrete[discrete >= self.resolution] = self.resolution - 1
        discrete = discrete.floor().long()
        return discrete

    def _cont_to_quat_cont(self, cont: torch.Tensor) -> torch.Tensor:
        a1 = cont[..., :3]
        a2 = cont[..., 3:]
        b1 = norm_tensor(a1)
        b2 = norm_tensor(a2 - (b1 * a2).sum(-1, keepdim=True) * b1)
        b3 = torch.cross(b1, b2, dim=-1)
        rot_matrix = torch.cat([b1, b2, b3], dim=-1).view(-1, 3, 3)
        r = R.from_matrix(rot_matrix.detach().cpu())
        quat = torch.from_numpy(r.as_quat())
        quat = quat.type_as(cont)

        # default rotation where the gripper is vertical
        nan = quat.isnan().any(-1)
        if nan.any():
            N = cont.dim()
            default = torch.Tensor(self.default_quat).type_as(cont)
            default = default.view(*([1] * (N - 1)), 4)
            quat[nan] = default

        return quat

    def _quat_cont_to_cont(self, quat: torch.Tensor) -> torch.Tensor:
        quat = quat.flatten(0, -2).view(-1, 4)
        so3 = compute_rotation_matrix_from_quaternions(quat)
        so3 = so3.view(-1, 3, 3)
        cont = so3[:, :2].flatten(-2).type_as(quat)
        return cont


def load_rotation(mode: RotMode = "mse", rot_type: RotType = "quat") -> Rotation:
    if rot_type == "quat":
        return QuatRotation(mode)
    if rot_type == "euler":
        return EulerRotation(mode)
    if rot_type == "cont":
        return ContRotation(mode)
    raise ValueError(f"Unexpected rotation {mode}")
