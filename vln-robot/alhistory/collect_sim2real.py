from typing import Optional, Tuple, List, Union, Dict
import itertools
from operator import itemgetter
import re
import pickle as pkl
from pydantic.dataclasses import dataclass, Field
import gym
import numpy as np
from PIL import Image
import tap
from torch.utils.data import DataLoader, Dataset
import torch
import muse.envs
from tqdm import tqdm
from pathlib import Path
from utils_sim2real import (
    APPLY_CAMERAS,
    get_gripper_state,
    obs_to_attn,
    transform,
    square,
    get_intrinsics,
    get_extrinsics,
    depth_to_pcd,
    depth_resize,
    rgb_resize,
    Observation,
    Demo,
    Misc,
)


def filter_state(state):
    filtered_state = dict()
    # process tool state information
    state_keys = list(state.keys())
    # filter parts out of state
    filter_keywords = ["rgb", "seg", "depth", "arms_joint_name", "info"]
    pattern = "|".join(f"{k}" for k in filter_keywords)
    pattern = re.compile(pattern)

    for k in state_keys:
        if not pattern.match(k):
            filtered_state[k] = state[k]
    return filtered_state


class Arguments(tap.Tap):
    env_name: str
    output: Path
    num_episodes: int = 1000
    num_workers: int = 20
    seed: int = 0
    offset: int = 0
    num_variations: int = 1
    seed_setup_path: Optional[Path] = None


def _is_stopped(i: int, demo: List[Dict], atol: float):
    return np.allclose(demo[i]["arms_joint_vel"], 0, atol=atol)
    # return np.allclose(demo[i].joint_velocities, 0, atol=0.02)


def keypoint_discovery(demo: List[Dict], patience: float = 10, atol: float = 1e-4):
    episode_keypoints = []
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(i, demo, atol)
        # if stopped:
        #     print(i, obs.joint_velocities.max())
        stopped_buffer = stopped_buffer + 1 if stopped else 0
        if stopped_buffer == patience:
            # print(i, "stopped buffer", stopped_buffer)
            episode_keypoints.append(i)

    # add start and end
    episode_keypoints.insert(0, 0)
    episode_keypoints.append(len(demo) - 1)

    return episode_keypoints


def get_env_name(env_name: str, variation: int) -> str:
    if "DR" in env_name:
        variation_env_name = "-".join(env_name.split("-")[1:])
        variation_env_name = f"DR-Var{variation}-{variation_env_name}"
    else:
        variation_env_name = f"Var{variation}-{env_name}"
    return variation_env_name


@dataclass
class Stats:
    num_steps: List[int] = Field(default_factory=list)
    successful_seeds: List = Field(default_factory=list)
    failure_seeds: List = Field(default_factory=list)
    action_space: List = Field(default_factory=list)
    cam_list: List = Field(default_factory=list)
    num_cameras: List = Field(default_factory=list)
    num_cameras: List = Field(default_factory=list)
    actions: List = Field(default_factory=list)
    obs: List = Field(default_factory=list)

    def process(self, output_path: Path):
        # Process Statistics
        total_num_steps = sum(self.num_steps)
        num_success_traj = len(self.successful_seeds)
        failure_seeds = self.failure_seeds

        print(
            f"[Data Collection] - Number of successful trajectories: {num_success_traj}"
        )
        print(f"[Data Collection] - Number of steps: {total_num_steps}")
        print(f"[Data Collection] - Failure seeds: {failure_seeds}")

        data = {}
        print("[Data Statistics] - Computing dataset statistics")
        # all_actions = collect_stats.pop("actions")
        # for action in all_actions:
        #     for k, v in action.items():
        #         if k not in data:
        #             data[k] = []
        #         data[k].append(v)
        # Mem issue
        # all_obs = collect_stats.pop("obs")
        # for obs in all_obs:
        #     for k, v in obs.items():
        #         if k not in data:
        #             data[k] = []
        #         data[k].append(v)
        velocity_dim = 0

        # for k, v in self.action_space[-1].items():
        #     if "grip" not in k:
        #         velocity_dim += v.shape[0]

        # print(f"[Data Statistics] - Final dataset size {len(all_actions)}")
        stats = {
            "num_cameras": self.num_cameras,
            "cam_list": self.cam_list,
            "action_space": self.action_space,
            "vel_dim": velocity_dim,
            # "dataset_size": len(all_actions),
            "traj_stats": {},
        }

        for k, v in data.items():
            stats["traj_stats"][k] = {
                "mean": np.mean(v, axis=0),
                "std": np.std(v, axis=0),
            }

        with open(str(output_path / "stats.pkl"), "wb") as f:
            pkl.dump(stats, f)


def get_observation(
    name: str, obs_gym: Dict, resolution: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fov = float(obs_gym[f"info_{name}_camera"]["fovy"])
    euler = torch.Tensor(obs_gym[f"info_{name}_camera"]["euler"])
    pos = torch.Tensor(obs_gym[f"info_{name}_camera"]["pos"])
    h, w = resolution
    extrinsics = get_extrinsics(pos, euler)
    intrinsics2 = get_intrinsics(fov, h, w)

    rgb = obs_gym[f"rgb_{name}_camera"]
    rgb = square(rgb)
    rgb2 = rgb_resize(rgb, h, w)

    depth = obs_gym[f"depth_{name}_camera"]
    depth2 = depth_resize(depth, extrinsics, fov, h, w)
    pcd2 = depth_to_pcd(depth2, extrinsics, intrinsics2)

    return rgb2, pcd2, extrinsics, intrinsics2


def convert_observation(
    obs_gym: Dict, blind: bool, resolution: Tuple[int, int]
) -> Observation:
    if not blind:
        # FIXME add wrist camera parameters
        (
            wrist_rgb,
            wrist_pcd,
            wrist_extrinsics,
            wrist_intrinsics,
        ) = get_observation("charlie", obs_gym, resolution)
        ls_rgb, ls_pcd, ls_extrinsics, ls_intrinsics = get_observation(
            "charlie", obs_gym, resolution
        )
        rs_rgb, rs_pcd, rs_extrinsics, rs_intrinsics = get_observation(
            "bravo", obs_gym, resolution
        )

        wrist_intrinsics = wrist_intrinsics[:, :3]
        ls_intrinsics = ls_intrinsics[:, :3]
        rs_intrinsics = rs_intrinsics[:, :3]
    else:
        wrist_rgb = None
        wrist_pcd = None
        wrist_extrinsics = None
        wrist_intrinsics = None
        ls_rgb = None
        ls_pcd = None
        ls_extrinsics = None
        ls_intrinsics = None
        rs_rgb = None
        rs_pcd = None
        rs_extrinsics = None
        rs_intrinsics = None

    misc: Misc = {
        "wrist_camera_extrinsics": wrist_extrinsics,
        "wrist_camera_intrinsics": wrist_intrinsics,
        "left_shoulder_camera_extrinsics": ls_extrinsics,
        "left_shoulder_camera_intrinsics": ls_intrinsics,
        "right_shoulder_camera_extrinsics": rs_extrinsics,
        "right_shoulder_camera_intrinsics": rs_intrinsics,
    }
    pose = np.concatenate([obs_gym["gripper_pos"], obs_gym["gripper_quat"]])
    joint_velocities = obs_gym["arms_joint_vel"]

    return Observation(
        gripper_open=get_gripper_state(obs_gym["gripper_state"]),
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


@dataclass
class Episode:
    buffer: List[Dict] = Field(default_factory=list)
    cameras: Tuple[str, ...] = APPLY_CAMERAS
    keypoints_: Optional[List[int]] = None
    resolution: Tuple[int, int] = (128, 128)

    def record(self, obs: Dict):
        self.buffer.append(obs)

    def __len__(self):
        return len(self.buffer)

    @property
    def keypoints(self) -> List[int]:
        if self.keypoints_ is None:
            self.keypoints_ = keypoint_discovery(self.buffer)
        return self.keypoints_

    def save(self, output: Union[str, Path]):
        if isinstance(output, str):
            output = Path(output)

        output.mkdir(exist_ok=False, parents=True)

        observations = [
            convert_observation(obs_gym, not i in self.keypoints, self.resolution)
            for i, obs_gym in enumerate(self.buffer)
        ]

        # Store images
        for kp in self.keypoints:
            obs = observations[kp]

            for camera in self.cameras:
                rgb_dir = output / f"{camera}_rgb"
                rgb_dir.mkdir(exist_ok=True)
                rgb = Image.fromarray(getattr(obs, f"{camera}_rgb"))
                rgb.save(rgb_dir / f"{kp}.png")
                setattr(obs, f"{camera}_rgb", None)

                pcd_dir = output / f"{camera}_point_cloud"
                pcd_dir.mkdir(exist_ok=True)
                with open(pcd_dir / f"{kp}.pkl", "wb") as fid:
                    pcd = getattr(obs, f"{camera}_pcd")
                    pkl.dump(pcd, fid)
                setattr(obs, f"{camera}_pcd", None)

        with open(output / "low_dim_obs.pkl", "wb") as fid:
            obs_dicts = [o.dict() for o in observations]
            pkl.dump(obs_dicts, fid)


class GenerationError(Exception):
    """
    Indicates an issue occured when generating a sample.
    """


class CollectDataset(Dataset):
    def __init__(
        self,
        env_name: str,
        variations: List[int],
        output_path: Union[Path, str],
        seeds: List[int],
        seed_setup_path: Union[Path, str, None],
        stats: Stats,
    ):

        self._env = None
        self._env_name: Optional[str] = None

        self._stats = stats

        self.output_path = Path(output_path)
        self.seed_setup_path = (
            Path(seed_setup_path) if seed_setup_path is not None else None
        )

        env_names = []
        for variation in variations:
            print(f"Collecting variation {variation}")
            variation_output_path = self.output_path / f"variation{variation}/episodes/"
            variation_output_path.mkdir(parents=True, exist_ok=True)

            variation_env_name = get_env_name(env_name, variation)

            env_names.append((variation_env_name, variation_output_path))

        self.items = [(*en, s) for en, s in itertools.product(env_names, seeds)]

        # sort items by env to reduce number of env init
        self.items = sorted(self.items, key=itemgetter(0))

    def get_env(self, env_name: str) -> gym.Env:
        if self._env_name == env_name and self._env is not None:
            return self._env
        self._env_name = env_name
        self._env = gym.make(self._env_name)
        return self._env

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        env_name, output_path, seed = self.items[index]
        env = self.get_env(env_name)
        env.seed(seed)

        seed_setup_path = (
            self.seed_setup_path / f"{seed:4d}.pkl"
            if self.seed_setup_path is not None
            else None
        )

        try:
            episode = collect_episode(env, seed_setup_path)
        except GenerationError:
            self._stats.failure_seeds.append(seed)
            return

        episode.save(output_path / f"episode{index}")

        self._stats.successful_seeds.append(seed)
        self._stats.num_steps.append(len(episode))

    def __del__(self):
        if self._env is not None:
            self._env.close()


def collect_episode(env: gym.Env, seed_setup_path=None) -> Episode:
    """
    >>> import gym, tempfile
    >>> from collect_sim2real import collect_episode
    >>> env = gym.make('Var0-Stack-v0')
    >>> episode = collect_episode(env)
    >>> assert 0 in episode.keypoints
    """
    episode = Episode()

    if seed_setup_path is not None:
        with open(seed_setup_path, "rb") as f:
            seed_setup = pkl.load(f)
        obs = env.reset(**seed_setup)
    else:
        obs = env.reset()

    agent = env.unwrapped.oracle()

    for _ in range(env.spec.max_episode_steps):
        action = agent.get_action(obs)

        # If oracle is not able to solve the task
        if action is None:
            raise GenerationError()

        episode.record(obs)

        obs, reward, done, info = env.step(action)
        if done:
            break

    return episode


def main(args: Arguments):
    variations = list(range(args.offset, args.offset + args.num_variations))

    seeds = list(range(args.seed, args.seed + args.num_episodes))
    stats = Stats()
    dataset = CollectDataset(
        args.env_name, variations, args.output, seeds, args.seed_setup_path, stats
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
    )

    for _ in tqdm(dataloader):
        continue

    stats.process(args.output)


if __name__ == "__main__":
    args = Arguments().parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    main(args)
