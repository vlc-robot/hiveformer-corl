import random
import itertools
import pickle
from typing import List, Dict, Optional, Sequence, Tuple, TypedDict, Union, Any
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import einops
# from rlbench.observation_config import ObservationConfig, CameraConfig
# from rlbench.environment import Environment
# from rlbench.backend.observation import Observation
# from rlbench.task_environment import TaskEnvironment
# from rlbench.action_modes.action_mode import MoveArmThenGripper
# from rlbench.action_modes.gripper_action_modes import Discrete
# from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
# from rlbench.backend.exceptions import InvalidActionError
# from rlbench.demo import Demo
# from pyrep.errors import IKError, ConfigurationPathError
# from pyrep.const import RenderMode
from structures import (
    Observation,
    Demo,
    GripperPose,
    Instructions,
    Output,
    RotType,
    RotMode,
    Sample,
    EulerRotation,
    ContRotation,
    QuatRotation,
    Rotation,
    Position,
)


def load_episodes() -> Dict[str, Any]:
    with open(Path(__file__).parent / "episodes.json") as fid:
        return json.load(fid)


def get_max_episode_length(tasks: Tuple[str, ...], variations: Tuple[int, ...]) -> int:
    return 10
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


# class Mover:
#     def __init__(self, task: TaskEnvironment, disabled: bool = False, max_tries: int = 1):
#         self._task = task
#         self._last_action: Optional[np.ndarray] = None
#         self._step_id = 0
#         self._max_tries = max_tries
#         self._disabled = disabled
#
#     def __call__(self, action: np.ndarray):
#         if self._disabled:
#             return self._task.step(action)
#
#         target = action.copy()
#         if self._last_action is not None:
#             action[7] = self._last_action[7].copy()
#
#         images = []
#         try_id = 0
#         obs = None
#         terminate = None
#         reward = 0
#
#         for try_id in range(self._max_tries):
#             obs, reward, terminate, other_obs = self._task.step(action)
#             if other_obs == []:
#                 other_obs = [obs]
#             for o in other_obs:
#                 images.append(
#                     {
#                         k.split("_")[0]: getattr(o, k)
#                         for k in o.__dict__.keys()
#                         if "_rgb" in k and getattr(o, k) is not None
#                     }
#                 )
#
#             pos = obs.gripper_pose[:3]
#             rot = obs.gripper_pose[3:7]
#             dist_pos = np.sqrt(np.square(target[:3] - pos).sum())  # type: ignore
#             dist_rot = np.sqrt(np.square(target[3:7] - rot).sum())  # type: ignore
#             # criteria = (dist_pos < 5e-2, dist_rot < 1e-1, (gripper > 0.5) == (target_gripper > 0.5))
#             criteria = (dist_pos < 5e-2,)
#
#             if all(criteria) or reward == 1:
#                 break
#
#             print(
#                 f"Too far away (pos: {dist_pos:.3f}, rot: {dist_rot:.3f}, step: {self._step_id})... Retrying..."
#             )
#
#         # we execute the gripper action after re-tries
#         action = target
#         if (
#             not reward
#             and self._last_action is not None
#             and action[7] != self._last_action[7]
#         ):
#             obs, reward, terminate, other_obs = self._task.step(action)
#             if other_obs == []:
#                 other_obs = [obs]
#             for o in other_obs:
#                 images.append(
#                     {
#                         k.split("_")[0]: getattr(o, k)
#                         for k in o.__dict__.keys()
#                         if "_rgb" in k and getattr(o, k) is not None
#                     }
#                 )
#
#         if try_id == self._max_tries:
#             print(f"Failure after {self._max_tries} tries")
#
#         self._step_id += 1
#         self._last_action = action.copy()
#
#         return obs, reward, terminate, images


class Model(TypedDict):
    model: nn.Module
    t: Dict[str, torch.Tensor]
    z: Dict[str, torch.Tensor]


# class Actioner:
#     def __init__(
#         self,
#         record_actions: bool = False,
#         replay_actions: Optional[Path] = None,
#         ground_truth_rotation: bool = False,
#         ground_truth_position: bool = False,
#         ground_truth_gripper: bool = False,
#         model: Optional[Model] = None,  # model includes t and z
#         model_rotation: Optional[nn.Module] = None,
#         model_position: Optional[nn.Module] = None,
#         model_gripper: Optional[nn.Module] = None,
#         apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
#         instructions: Optional[Dict] = None,
#         taskvar_token: bool = False,
#     ):
#         self._record_actions = record_actions
#         self._replay_actions = replay_actions
#         self._ground_truth_rotation = ground_truth_rotation
#         self._ground_truth_position = ground_truth_position
#         self._ground_truth_gripper = ground_truth_gripper
#         assert (model is not None) ^ (
#             model_rotation is not None
#             and model_position is not None
#             and model_gripper is not None
#         )
#         self._model = model
#         self._model_rotation = model_rotation
#         self._model_position = model_position
#         self._model_gripper = model_gripper
#         self._apply_cameras = apply_cameras
#         self._instructions = instructions
#         self._taskvar_token = taskvar_token
#
#         if self._taskvar_token:
#             with open(Path(__file__).parent / "tasks.csv", "r") as fid:
#                 self._tasks = [l.strip() for l in fid.readlines()]
#
#         self._actions: Dict = {}
#         self._instr: Optional[torch.Tensor] = None
#         self._taskvar: Optional[torch.Tensor] = None
#         self._task: Optional[str] = None
#
#     def load_episode(
#         self, task_str: str, variation: int, demo_id: int, demo: Union[Demo, int]
#     ):
#         self._task = task_str
#
#         if self._instructions is None:
#             self._instr = None
#         else:
#             instructions = list(self._instructions[task_str][variation])
#             self._instr = random.choice(instructions).unsqueeze(0)
#
#         if self._taskvar_token:
#             task_id = self._tasks.index(task_str)
#             self._taskvar = torch.Tensor([[task_id, variation]]).unsqueeze(0)
#             print(self._taskvar)
#
#         if self._replay_actions is not None:
#             self._actions = torch.load(
#                 self._replay_actions / f"episode{demo_id}" / "actions.pth"
#             )
#         elif (
#             self._ground_truth_rotation
#             or self._ground_truth_position
#             or self._ground_truth_gripper
#         ):
#             if isinstance(demo, int):
#                 raise NotImplementedError()
#             action_ls = self.get_action_from_demo(demo)
#             self._actions = dict(enumerate(action_ls))
#         else:
#             self._actions = {}
#
#     def get_action_from_demo(self, demo: Demo):
#         """
#         Fetch the desired state and action based on the provided demo.
#             :param demo: fetch each demo and save key-point observations
#             :param normalise_rgb: normalise rgb to (-1, 1)
#             :return: a list of obs and action
#         """
#         key_frame = keypoint_discovery(demo)
#         action_ls = []
#         for f in key_frame:
#             obs = demo[f]
#             action_np = np.concatenate([obs.gripper_pose, [obs.gripper_open]])  # type: ignore
#             action = torch.from_numpy(action_np)
#             action_ls.append(action.unsqueeze(0))
#         return action_ls
#
#     def predict(
#         self, step_id: int, rgbs: torch.Tensor, pcds: torch.Tensor, gripper: torch.Tensor
#     ) -> Dict[str, Any]:
#         padding_mask = torch.ones_like(rgbs[:, :, 0, 0, 0, 0]).bool()
#         output: Dict[str, Any] = {"action": None, "attention": {}}
#
#         if self._instr is not None:
#             self._instr = self._instr.to(rgbs.device)
#
#         if self._taskvar is not None:
#             self._taskvar = self._taskvar.to(rgbs.device)
#
#         if self._replay_actions:
#             if step_id not in self._actions:
#                 print(f"Step {step_id} is not prerecorded!")
#                 return output
#             action = self._actions[step_id]
#         elif self._model is None:
#             action = torch.Tensor([]).to(self.device)
#             keys = ("position", "rotation", "gripper")
#             slices = (slice(0, 3), slice(3, 7), slice(7, 8))
#             for key, slice_ in zip(keys, slices):
#                 model = getattr(self, f"_model_{key}")
#                 t = model["t"][self._task][: step_id + 1].unsqueeze(0)
#                 z = model["z"][self._task][: step_id + 1].unsqueeze(0)
#                 pred = model["model"](
#                     rgbs, pcds, padding_mask, t, z, self._instr, gripper, self._taskvar
#                 )
#                 action_key = model["model"].compute_action(pred)
#                 action = torch.cat([action, action_key[slice_]])
#             output["action"] = action
#         else:
#             if self._task is None:
#                 raise ValueError()
#             t = self._model["t"][self._task][: step_id + 1].unsqueeze(0)
#             z = self._model["z"][self._task][: step_id + 1].unsqueeze(0)
#             pred = self._model["model"](
#                 rgbs, pcds, padding_mask, t, z, self._instr, gripper, self._taskvar
#             )
#             output["action"] = self._model["model"].compute_action(pred)  # type: ignore
#             output["attention"] = pred["attention"]
#
#         if self._ground_truth_rotation:
#             if step_id not in self._actions:
#                 print(f"No ground truth available for step {step_id}!")
#                 return output
#             output["action"][:, 3:7] = self._actions[step_id][:, 3:7]
#         if self._ground_truth_position:
#             if step_id not in self._actions:
#                 print(f"No ground truth available for step {step_id}!")
#                 return output
#             output["action"][:, :3] = self._actions[step_id][:, :3]
#         if self._ground_truth_gripper:
#             if step_id not in self._actions:
#                 print(f"No ground truth available for step {step_id}!")
#                 return output
#             output["action"][:, 7] = self._actions[step_id][:, 7]
#
#         if self._record_actions:
#             self._actions[step_id] = output["action"]
#
#         return output
#
#     def save(self, ep_dir):
#         if self._record_actions:
#             torch.save(self._actions, ep_dir / "actions.pth")
#
#     @property
#     def device(self):
#         if self._model is not None:
#             return next(self._model["model"].parameters()).device
#         return next(self._model_position["model"].parameters()).device  # type: ignore
#
#     def eval(self):
#         if self._model is not None:
#             self._model["model"].eval()
#         else:
#             self._model_position["model"].eval()  # type: ignore
#             self._model_rotation["model"].eval()  # type: ignore
#             self._model_gripper["model"].eval()  # type: ignore


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
        plt.colorbar()
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
# RLBench Environment & Related Functions
# --------------------------------------------------------------------------------
# class RLBenchEnv:
#     def __init__(
#         self,
#         data_path,
#         apply_rgb=False,
#         apply_depth=False,
#         apply_pc=False,
#         headless=False,
#         apply_cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
#         gripper_pose: GripperPose = "none",
#     ):
#
#         # setup required inputs
#         self.data_path = data_path
#         self.apply_rgb = apply_rgb
#         self.apply_depth = apply_depth
#         self.apply_pc = apply_pc
#         self.apply_cameras = apply_cameras
#         self.gripper_pose = gripper_pose
#
#         # setup RLBench environments
#         self.obs_config = self.create_obs_config(
#             apply_rgb, apply_depth, apply_pc, apply_cameras
#         )
#         self.action_mode = MoveArmThenGripper(
#             arm_action_mode=EndEffectorPoseViaPlanning(),
#             gripper_action_mode=Discrete(),
#         )
#         self.env = Environment(
#             self.action_mode, str(data_path), self.obs_config, headless=headless
#         )
#
#     def get_obs_action(self, obs):
#         """
#         Fetch the desired state and action based on the provided demo.
#             :param obs: incoming obs
#             :return: required observation and action list
#         """
#
#         # fetch state
#         state_dict = {"rgb": [], "depth": [], "pc": []}
#         for cam in self.apply_cameras:
#             if self.apply_rgb:
#                 rgb = getattr(obs, "{}_rgb".format(cam))
#                 state_dict["rgb"] += [rgb]
#
#             if self.apply_depth:
#                 depth = getattr(obs, "{}_depth".format(cam))
#                 state_dict["depth"] += [depth]
#
#             if self.apply_pc:
#                 pc = getattr(obs, "{}_point_cloud".format(cam))
#                 state_dict["pc"] += [pc]
#
#         # fetch action
#         action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
#         return state_dict, torch.from_numpy(action).float()
#
#     def get_rgb_pcd_gripper_from_obs(
#         self, obs: Observation
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Return rgb, pcd, and gripper from a given observation
#         :param obs: an Observation from the env
#         :return: rgb, pcd, gripper
#         """
#         state_dict, gripper = self.get_obs_action(obs)
#         state = transform(state_dict, augmentation=False)
#         state = einops.rearrange(
#             state,
#             "(m n ch) h w -> n m ch h w",
#             ch=3,
#             n=len(self.apply_cameras),
#             m=2,
#         )
#         rgb = state[:, 0].unsqueeze(0)  # 1, N, C, H, W
#         pcd = state[:, 1].unsqueeze(0)  # 1, N, C, H, W
#         gripper = gripper.unsqueeze(0)  # 1, D
#
#         if "attn" in self.gripper_pose:
#             attns = torch.Tensor([])
#             for cam in self.apply_cameras:
#                 u, v = obs_to_attn(obs, cam)
#                 attn = torch.zeros((1, 1, 1, 128, 128))
#                 if not (u < 0 or u > 127 or v < 0 or v > 127):
#                     attn[0, 0, 0, v, u] = 1
#                 attns = torch.cat([attns, attn], 1)
#             rgb = torch.cat([rgb, attns], 2)
#
#         return rgb, pcd, gripper
#
#     def get_obs_action_from_demo(self, demo: Demo):
#         """
#         Fetch the desired state and action based on the provided demo.
#             :param demo: fetch each demo and save key-point observations
#             :param normalise_rgb: normalise rgb to (-1, 1)
#             :return: a list of obs and action
#         """
#         key_frame = keypoint_discovery(demo)
#         key_frame.insert(0, 0)
#         state_ls = []
#         action_ls = []
#         for f in key_frame:
#             state, action = self.get_obs_action(demo[f])
#             state = transform(state, augmentation=False)
#             state_ls.append(state.unsqueeze(0))
#             action_ls.append(action.unsqueeze(0))
#         return state_ls, action_ls
#
#     def get_demo(self, task_name, variation, episode_index):
#         """
#         Fetch a demo from the saved environment.
#             :param task_name: fetch task name
#             :param variation: fetch variation id
#             :param episode_index: fetch episode index: 0 ~ 99
#             :return: desired demo
#         """
#         demos = self.env.get_demos(
#             task_name=task_name,
#             variation_number=variation,
#             amount=1,
#             from_episode_number=episode_index,
#             random_selection=False,
#         )
#         return demos
#
#     def evaluate(
#         self,
#         task_str: str,
#         max_episodes: int,
#         variation: int,
#         num_demos: int,
#         log_dir: Optional[Path],
#         actioner: Actioner,
#         offset: int = 0,
#         max_tries: int = 1,
#         demos: Optional[List[Demo]] = None,
#         save_attn: bool = False,
#     ):
#         """
#         Evaluate the policy network on the desired demo or test environments
#             :param task_type: type of task to evaluate
#             :param max_episodes: maximum episodes to finish a task
#             :param num_demos: number of test demos for evaluation
#             :param model: the policy network
#             :param demos: whether to use the saved demos
#             :return: success rate
#         """
#
#         self.env.launch()
#         task_type = task_file_to_task_class(task_str)
#         task = self.env.get_task(task_type)
#         task.set_variation(variation)  # type: ignore
#
#         actioner.eval()
#         device = actioner.device
#
#         success_rate = 0.0
#
#         if demos is None:
#             fetch_list = [i for i in range(num_demos)]
#         else:
#             fetch_list = demos
#
#         fetch_list = fetch_list[offset:]
#
#         with torch.no_grad():
#             for demo_id, demo in enumerate(tqdm(fetch_list)):
#
#                 images = []
#                 rgbs = torch.Tensor([]).to(device)
#                 pcds = torch.Tensor([]).to(device)
#                 grippers = torch.Tensor([]).to(device)
#
#                 # reset a new demo or a defined demo in the demo list
#                 if isinstance(demo, int):
#                     _, obs = task.reset()
#                 else:
#                     print("Resetting to demo")
#                     print(demo)
#                     _, obs = task.reset_to_demo(demo)  # type: ignore
#
#                 actioner.load_episode(task_str, variation, demo_id, demo)
#
#                 images.append(
#                     {cam: getattr(obs, f"{cam}_rgb") for cam in self.apply_cameras}
#                 )
#                 move = Mover(task, max_tries=max_tries)
#                 reward = None
#
#                 for step_id in range(max_episodes):
#                     # fetch the current observation, and predict one action
#                     rgb, pcd, gripper = self.get_rgb_pcd_gripper_from_obs(obs)  # type: ignore
#
#                     rgb = rgb.to(device)
#                     pcd = pcd.to(device)
#                     gripper = gripper.to(device)
#
#                     rgbs = torch.cat([rgbs, rgb.unsqueeze(1)], dim=1)
#                     pcds = torch.cat([pcds, pcd.unsqueeze(1)], dim=1)
#                     grippers = torch.cat([grippers, gripper.unsqueeze(1)], dim=1)
#
#                     output = actioner.predict(step_id, rgbs, pcds, grippers)
#                     action = output["action"]
#
#                     if action is None:
#                         break
#
#                     if log_dir is not None and save_attn and output["action"] is not None:
#                         ep_dir = log_dir / f"episode{demo_id}"
#                         fig = plot_attention(
#                             output["attention"][-1],
#                             rgbs[0][-1, :, :3],
#                             pcds[0][-1].view(3, 3, 128, 128),
#                             ep_dir / f"attn_{step_id}.png",
#                         )
#
#                     # update the observation based on the predicted action
#                     try:
#                         action_np = action[-1].detach().cpu().numpy()
#
#                         # HACK tower3
#                         if task_str == "tower3":
#                             step1 = gripper.cpu().numpy()[-1]
#                             if step_id > 0:
#                                 step1[2] += 0.1
#                                 move(step1)
#                             step2 = action_np.copy()
#                             step2[2] += 0.1
#                             step2[7] = step1[7]
#                             move(step2)
#
#                         obs, reward, terminate, step_images = move(action_np)
#
#                         images += step_images
#
#                         if reward == 1:
#                             success_rate += 1 / num_demos
#                             break
#                         if terminate:
#                             print("The episode has terminated!")
#                     except (IKError, ConfigurationPathError, InvalidActionError) as e:
#                         print(task_type, demo, step_id, success_rate, e)
#                         reward = 0
#                         break
#                 print(
#                     task_str,
#                     "Reward",
#                     reward,
#                     "Variation",
#                     variation,
#                     "Step",
#                     demo_id,
#                     "SR: %.2f" % (success_rate * 100),
#                 )
#
#                 if log_dir is not None:
#                     ep_dir = log_dir / task_str / f"episode{demo_id}"
#                     ep_dir.mkdir(exist_ok=True, parents=True)
#                     for frame_id, img_by_cam in enumerate(images):
#                         for cam, im in img_by_cam.items():
#                             cam_dir = ep_dir / cam
#                             cam_dir.mkdir(exist_ok=True, parents=True)
#                             Image.fromarray(im).save(cam_dir / f"{frame_id}.png")
#
#         self.env.shutdown()
#         return success_rate
#
#     def create_obs_config(
#         self, apply_rgb, apply_depth, apply_pc, apply_cameras, **kwargs
#     ):
#         """
#         Set up observation config for RLBench environment.
#             :param apply_rgb: Applying RGB as inputs.
#             :param apply_depth: Applying Depth as inputs.
#             :param apply_pc: Applying Point Cloud as inputs.
#             :param apply_cameras: Desired cameras.
#             :return: observation config
#         """
#         unused_cams = CameraConfig()
#         unused_cams.set_all(False)
#         used_cams = CameraConfig(
#             rgb=apply_rgb,
#             point_cloud=apply_pc,
#             depth=apply_depth,
#             mask=False,
#             render_mode=RenderMode.OPENGL,
#             **kwargs,
#         )
#
#         camera_names = apply_cameras
#         kwargs = {}
#         for n in camera_names:
#             kwargs[n] = used_cams
#
#         obs_config = ObservationConfig(
#             front_camera=kwargs.get("front", unused_cams),
#             left_shoulder_camera=kwargs.get("left_shoulder", unused_cams),
#             right_shoulder_camera=kwargs.get("right_shoulder", unused_cams),
#             wrist_camera=kwargs.get("wrist", unused_cams),
#             overhead_camera=kwargs.get("overhead", unused_cams),
#             joint_forces=False,
#             joint_positions=False,
#             joint_velocities=True,
#             task_low_dim_state=False,
#             gripper_touch_forces=False,
#             gripper_pose=True,
#             gripper_open=True,
#             gripper_matrix=True,
#             gripper_joint_positions=True,
#         )
#
#         return obs_config


# Identify way-point in each RLBench Demo
def _is_stopped(demo, i, obs, stopped_buffer):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].gripper_open
        and obs.gripper_open == demo[i - 1].gripper_open
        and demo[i - 2].gripper_open == demo[i - 1].gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=0.1)
    stopped = (
        stopped_buffer <= 0
        and small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )
    return stopped


def keypoint_discovery(demo: Demo) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # If change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)

    # HACK for tower3 task
    return episode_keypoints


# --------------------------------------------------------------------------------
# General Functions
# --------------------------------------------------------------------------------
def transform(obs_dict, scale_size=(0.75, 1.25), augmentation=False):
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

        if augmentation:
            raise NotImplementedError()  # Deprecated

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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_rotation(mode: RotMode = "mse", rot_type: RotType = "quat") -> Rotation:
    if rot_type == "quat":
        return QuatRotation(mode)
    if rot_type == "euler":
        return EulerRotation(mode)
    if rot_type == "cont":
        return ContRotation(mode)
    raise ValueError(f"Unexpected rotation {mode}")


def load_instructions(instructions: Optional[Path], tasks: Optional[Sequence[str]] = None, variations: Optional[Sequence[int]] = None) -> Optional[Instructions]:
    if instructions is not None:
        with open(instructions, "rb") as fid:
            data: Instructions = pickle.load(fid)

        if tasks is not None:
            data = {
                task: var_instr
                for task, var_instr in data.items()
                if task in tasks 
            }
        
        if variations is not None:
            data = {
                task: {var: instr for var, instr in var_instr.items()
                     if var in variations}
                for task, var_instr in data.items()
            }

        return data

    return None


# def reduce_per_task(losses: Dict[str, torch.Tensor], tasks: List[str]) -> Dict[str, torch.Tensor]:
#     """
#     Reduce the losses tensor per task by averaging them.
#
#     >>> losses = {'position': torch.Tensor([1., 3., 2.])}
#     >>> tasks = ['A', 'B', 'A']
#     >>> reduce_per_task(losses, tasks)
#     >>> losses
#     {'position/total': 2.0, 'position/A': 1.5, 'position/B': 3.}
#     """
#     if 'total' in tasks:
#         raise ValueError(tasks)
#
#     device = next(losses.values()).device
#     reduced = {}
#     for task in set(tasks):
#         mask = torch.tensor([task == t for t in tasks])
#         reduced[


class LossAndMetrics:
    def __init__(self, rotation: Rotation, position: Position):
        self.rot = rotation
        self.pos = position
        task_file = Path(__file__).parent / "tasks.csv"
        with open(task_file) as fid:
            self.tasks = [t.strip() for t in fid.readlines()]

    def compute_loss(
        self, pred: Output, sample: Sample
    ) -> Dict[str, torch.Tensor]:

        device = pred["position"].device
        padding_mask = sample["padding_mask"].to(device)
        action = sample["action"].to(device)[padding_mask]

        losses = {}
        losses.update(self.pos.compute_loss(pred, sample, reduction='mean'))
        losses.update(self.rot.compute_loss(pred, sample, reduction='none'))

        # reduce_per_task(losses, sample['task'])

        losses["gripper"] = F.mse_loss(pred["gripper"], action[:, 7:8])

        if pred["task"] is not None:
            task = torch.Tensor([self.tasks.index(t) for t in sample["task"]])
            task = task.to(device).long()
            losses["task"] = F.cross_entropy(pred["task"], task)

        return losses

    def compute_metrics(
        self, pred: Dict[str, torch.Tensor], sample: Sample
    ) -> Dict[str, torch.Tensor]:
        device = pred["position"].device
        dtype = pred["position"].dtype
        padding_mask = sample["padding_mask"].to(device)
        outputs = sample["action"].to(device)[padding_mask]

        metrics = {}

        metrics.update(self.pos.compute_metrics(pred["position"], outputs[:, :3]))

        pred_gripper = (pred["gripper"] > 0.5).squeeze(-1)
        true_gripper = outputs[:, 7].bool()
        acc = pred_gripper == true_gripper
        metrics["gripper"] = acc.to(dtype).mean()

        metrics.update(self.rot.compute_metrics(pred["rotation"], outputs[:, 3:7]))

        if pred["task"] is not None:
            task = torch.Tensor([self.tasks.index(t) for t in sample["task"]])
            task = task.to(device).long()
            acc = task == pred["task"].argmax(1)
            metrics["task"] = acc.to(dtype).mean()

        return metrics
