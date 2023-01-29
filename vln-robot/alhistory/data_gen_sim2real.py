"""
Convert Pickle data file into a proper dataset
"""
import random
import itertools
from typing import Tuple, Dict, List, Union
import pickle
from pathlib import Path
import json
from tqdm import tqdm
from PIL import Image
import tap
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import einops
from utils_sim2real import (
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
)


class Arguments(tap.Tap):
    data_dir: Path = Path(__file__).parent / "sim2real"
    episode_desc: Path = Path(__file__).parent / "episodes.json"
    seed: int = 2
    tasks: Tuple[str, ...] = ("tower",)
    cameras: Tuple[str, ...] = ("wrist", "left_shoulder", "right_shoulder")
    output: Path = Path(__file__).parent / "datasets"
    variations: List[int] = [0]
    offset_episode: int = 0
    num_workers: int = 0
    resolution: Tuple[int, int] = (128, 128)


def get_obs_action(obs, cameras: Tuple[str, ...]):
    """
    Fetch the desired state and action based on the provided demo.
        :param obs: incoming obs
        :return: required observation and action list
    """
    # fetch state
    state_dict: Dict[str, List] = {"rgb": [], "pc": []}
    for cam in cameras:
        rgb = getattr(obs, "{}_rgb".format(cam))
        state_dict["rgb"].append(rgb)

        pcd = getattr(obs, "{}_pcd".format(cam))
        state_dict["pc"].append(pcd)

    # fetch action
    action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
    return state_dict, torch.from_numpy(action).float()


def _extract_keypoints_from_demo(demo) -> List[int]:
    """
    >>> import gym, tempfile
    >>> from collect_sim2real import collect_episode
    >>> env = gym.make('Var0-Stack-v0')
    >>> episode = collect_episode(env)
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     episode.save(f'{tmp}/ep')
    ...     del episode
    ...     demo = load_demo(f'{tmp}/ep')
    >>> keypoints = _extract_keypoints_from_demo(demo)
    >>> assert 0 in keypoints
    """
    return [i for i, f in enumerate(demo) if f.left_shoulder_rgb is not None]


def get_obs_action_from_demo(demo: Demo, cameras: Tuple[str, ...]):
    """
    Fetch the desired state and action based on the provided demo.
        :param demo: fetch each demo and save key-point observations
        :return: a list of obs and action
    """
    key_frame = _extract_keypoints_from_demo(demo)
    state_ls = []
    action_ls = []
    for f in key_frame:
        state, action = get_obs_action(demo[f], cameras)
        state = transform(state)
        state_ls.append(state.unsqueeze(0))
        action_ls.append(action.unsqueeze(0))
    return state_ls, action_ls


def get_observation(
    name: str, data: Dict, resolution: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fov = float(data[f"info_{name}"]["fovy"])
    euler = torch.Tensor(data[f"info_{name}"]["euler"])
    pos = torch.Tensor(data[f"info_{name}"]["pos"])
    h, w = resolution
    extrinsics = get_extrinsics(pos, euler)
    intrinsics2 = get_intrinsics(fov, h, w)

    rgb = data[f"rgb_{name}"]
    rgb = square(rgb)
    rgb2 = rgb_resize(rgb, h, w)

    depth = data[f"depth_{name}"]
    depth2 = depth_resize(depth, extrinsics, fov, h, w)
    pcd2 = depth_to_pcd(depth2, extrinsics, intrinsics2)

    return rgb2, pcd2, extrinsics, intrinsics2


def _extract_camera_from_folder_name(f: Path) -> str:
    """
    >>> from pathlib import Path
    >>> f = Path('/tmp/my_camera_rgb')
    >>> _extract_camera_from_folder_name(f)
    'my_camera'
    """
    name = f.stem
    return name.replace("_rgb", "").replace("_point_cloud", "")


def load_demo(ep_dir: Union[str, Path]) -> Demo:
    """
    Load a demo generated with collect_sim2real.py

    >>> import gym, tempfile
    >>> from collect_sim2real import collect_episode
    >>> env = gym.make('Var0-Stack-v0')
    >>> episode = collect_episode(env)
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     episode.save(f'{tmp}/ep')
    ...     num_frames = len(episode)
    ...     del episode
    ...     demo = load_demo(f'{tmp}/ep')
    >>> assert num_frames == len(demo)
    """
    if isinstance(ep_dir, str):
        ep_dir = Path(ep_dir)

    # Load low_dim_obs.pkl
    with open(ep_dir / "low_dim_obs.pkl", "rb") as fid:
        episode: Demo = [Observation(**o) for o in pickle.load(fid)]

    # Load keypoint images
    cameras = [
        _extract_camera_from_folder_name(f) for f in ep_dir.iterdir() if f.is_dir()
    ]
    if cameras == []:
        raise RuntimeError(f"{ep_dir} format is unexpected")

    any_camera_folder = ep_dir / f"{cameras[0]}_rgb"
    if not any_camera_folder.is_dir():
        raise RuntimeError(f"{any_camera_folder} is not a folder")

    keypoints = [int(f.stem) for f in any_camera_folder.glob("*.png")]
    if keypoints == []:
        raise RuntimeError(f"No images were found at {any_camera_folder}")

    for cam, kp in itertools.product(cameras, keypoints):
        with open(ep_dir / f"{cam}_point_cloud" / f"{kp}.pkl", "rb") as fid:
            setattr(episode[kp], f"{cam}_pcd", pickle.load(fid))

        img = Image.open(ep_dir / f"{cam}_rgb" / f"{kp}.png")
        setattr(episode[kp], f"{cam}_rgb", np.asarray(img))

    return episode


def get_attn_indices_from_demo(
    task_str: str, demo: Demo, cameras: Tuple[str, ...]
) -> List[Dict[str, Tuple[int, int]]]:
    keypoints = _extract_keypoints_from_demo(demo)  # type: ignore
    return [{cam: obs_to_attn(demo[i], cam) for cam in cameras} for i in keypoints]


class DatagenDataset(Dataset):
    def __init__(self, args: Arguments):
        self.args = args

        with open(args.episode_desc) as fid:
            episodes = json.load(fid)
        self.max_eps_dict = episodes["max_episode_length"]
        self.variable_lengths = set(episodes["variable_length"])

        for task_str in args.tasks:
            if task_str in self.max_eps_dict:
                continue
            ep_dir = (
                args.data_dir / task_str / f"variation{args.variations[0]}" / "episodes"
            )
            demo = load_demo(ep_dir / "episode0")
            _, state_ls = get_obs_action_from_demo(demo, args.cameras)
            self.max_eps_dict[task_str] = len(state_ls) - 1
            raise ValueError(
                f"Guessing that the size of {task_str} is {len(state_ls) - 1}"
            )

        broken = set(episodes["broken"])
        tasks = [t for t in args.tasks if t not in broken]
        self.items = []
        for task_str, variation in itertools.product(tasks, args.variations):
            ep_dir = args.data_dir / task_str / f"variation{variation}" / "episodes"
            episodes = set(
                [
                    (task_str, variation, int(ep.stem[7:]))
                    for ep in ep_dir.glob("episode*")
                ]
            )
            episodes = set((t, v, e) for t, v, e in episodes if e >= args.offset_episode)
            self.items += sorted(episodes)

        self.num_items = len(self.items)

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, index: int) -> None:
        task, variation, episode = self.items[index]
        taskvar_dir = args.output / f"{task}+{variation}"
        taskvar_dir.mkdir(parents=True, exist_ok=True)

        try:
            ep_dir = self.args.data_dir / task / f"variation{variation}" / "episodes"
            demo = load_demo(ep_dir / f"episode{episode}")
            state_ls, action_ls = get_obs_action_from_demo(demo, self.args.cameras)
        # except (FileNotFoundError, RuntimeError, IndexError, EOFError) as e:
        except EOFError as e:
            print("ERROR", e)
            return

        state_ls = einops.rearrange(
            state_ls,
            "t 1 (m n ch) h w -> t n m ch h w",
            ch=3,
            n=len(args.cameras),
            m=2,
        )

        frame_ids = list(range(len(state_ls) - 1))
        num_frames = len(frame_ids)
        attn_indices = get_attn_indices_from_demo(task, demo, args.cameras)

        if (task in self.variable_lengths and num_frames > self.max_eps_dict[task]) or (
            task not in self.variable_lengths and num_frames != self.max_eps_dict[task]
        ):
            print(f"ERROR ({task}, {variation}, {episode})")
            print(f"\t {len(frame_ids)} != {self.max_eps_dict[task]}")
            return

        state_dict: List = [[] for _ in range(6)]
        state_dict[0].extend(frame_ids)
        state_dict[1].extend(state_ls[:-1])
        state_dict[2].extend(action_ls[1:])
        state_dict[3].extend(attn_indices[:-1])
        state_dict[4].extend(action_ls[:-1])  # gripper pos
        last_frame = [state_ls[-1], action_ls[-1], attn_indices[-1]]
        state_dict[5].append(last_frame)

        np.save(taskvar_dir / f"ep{episode}.npy", state_dict)  # type: ignore


if __name__ == "__main__":
    args = Arguments().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset = DatagenDataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
    )

    for _ in tqdm(dataloader):
        continue
