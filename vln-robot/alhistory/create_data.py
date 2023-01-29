import random
import itertools
from typing import (
    Sequence,
    Union,
    Optional,
    Tuple,
    List,
    Dict,
    Callable,
    TypeVar,
    Generic,
)
from pickle import UnpicklingError
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data
from torch.nn import functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
import einops
from network import GripperPose
from utils import Instructions, Sample
from structures import CameraName


T = TypeVar("T")
U = TypeVar("U")


class Cache(Generic[T, U]):
    def __init__(self, size: int, loader: Callable[[T], U]):
        self._size = size
        self._loader = loader
        self._keys: List[T] = []
        self._cache: Dict[T, U] = {}

    def __call__(self, args: T) -> U:
        if args in self._cache:
            index = self._keys.index(args)
            del self._keys[index]
            self._keys.append(args)
            return self._cache[args]

        # print(args, len(self._keys), self._size)
        value = self._loader(args)

        if len(self._keys) == self._size and self._keys != []:
            key = self._keys[0]
            del self._cache[key]
            del self._keys[0]

        if len(self._keys) < self._size:
            self._keys.append(args)
            self._cache[args] = value

        return value


class DataTransform(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Except tensors as T, N, C, H, W
        """
        keys = list(kwargs.keys())

        if len(keys) == 0:
            raise RuntimeError("No args")

        # Continuous range of scales
        sc = np.random.uniform(*self.scales)

        t, n, c, raw_h, raw_w = kwargs[keys[0]].shape
        kwargs = {n: arg.flatten(0, 1) for n, arg in kwargs.items()}
        resized_size = [int(raw_h * sc), int(raw_w * sc)]

        # Resize based on randomly sampled scale
        kwargs = {
            n: transforms_f.resize(
                arg,
                resized_size,
                transforms.InterpolationMode.NEAREST
                # if "pc" in n
                # else transforms.InterpolationMode.BILINEAR,
            )
            for n, arg in kwargs.items()
        }

        # Adding padding if crop size is smaller than the resized size
        if raw_h > resized_size[0] or raw_w > resized_size[1]:
            right_pad, bottom_pad = max(raw_h - resized_size[1], 0), max(
                raw_w - resized_size[0], 0
            )
            kwargs = {
                n: transforms_f.pad(
                    arg,
                    padding=[0, 0, right_pad, bottom_pad],
                    padding_mode="reflect",
                )
                for n, arg in kwargs.items()
            }

        # Random Cropping
        i, j, h, w = transforms.RandomCrop.get_params(
            kwargs[keys[0]], output_size=(raw_h, raw_w)
        )

        kwargs = {n: transforms_f.crop(arg, i, j, h, w) for n, arg in kwargs.items()}

        kwargs = {
            n: einops.rearrange(arg, "(t n) c h w -> t n c h w", t=t)
            for n, arg in kwargs.items()
        }

        return kwargs


def loader(file: Path) -> Optional[np.ndarray]:
    try:
        return np.load(file, allow_pickle=True)
    except UnpicklingError as e:
        print(f"Can't load {file}: {e}")
    return None


class RLBench(data.Dataset):
    """
    RLBench dataset, 10 tasks
    """

    def __init__(
        self,
        root: Union[Path, str, List[Path], List[str]],
        taskvar: List[Tuple[str, int]],
        instructions: Optional[Instructions],
        gripper_pose: GripperPose,
        taskvar_token: bool,
        max_episode_length: int,
        cache_size: int,
        max_episodes_per_taskvar: int,
        num_iters: Optional[int] = None,
        jitter: bool = False,
        cameras: Tuple[CameraName, ...] = ("wrist", "left_shoulder", "right_shoulder"),
        training: bool = True,
    ):
        self._cache = Cache(cache_size, loader)
        self._gripper_pose = gripper_pose
        self._cameras = cameras
        self._max_episode_length = max_episode_length
        self._max_episodes_per_taskvar = max_episodes_per_taskvar
        self._num_iters = num_iters
        self._training = training
        self._taskvar = taskvar
        self._taskvar_token = taskvar_token
        self._transform = DataTransform((0.75, 1.25))
        if isinstance(root, (Path, str)):
            root = [Path(root)]
        self._root: List[Path] = [Path(r).expanduser() for r in root]

        self._task_to_id: Optional[Dict[str, int]] = None
        if taskvar_token:
            with open(Path(__file__).parent / "tasks.csv", "r") as tid:
                self._task_to_id = {l.strip(): i for i, l in enumerate(tid.readlines())}

        # We keep only useful instructions to save mem
        if instructions is None:
            self._instructions = None
        else:
            self._instructions = defaultdict(dict)
            for task, var in taskvar:
                self._instructions[task][var] = instructions[task][var]

        self._data_dirs = []
        self._episodes = []
        self._num_episodes = 0
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root / f"{task}+{var}"
            if not data_dir.is_dir():
                print(f"WARNING: Can't find dataset folder {data_dir}")
            episodes = [(task, var, ep) for ep in data_dir.glob("*.npy")]
            episodes = episodes[: self._max_episodes_per_taskvar]
            num_episodes = len(episodes)
            if num_episodes == 0:
                print(f"WARNING: Can't find episodes at folder {data_dir}")
            self._data_dirs.append(data_dir)
            self._episodes += episodes
            self._num_episodes += num_episodes

        if jitter and self._training:
            self.jitter = transforms.ColorJitter(
                brightness=32.0 / 255.0,  # type: ignore
                contrast=[0.5, 1.5],  # type: ignore
                saturation=[0.5, 1.5],  # type: ignore
                hue=1 / 24,  # type: ignore
            )
        else:
            self.jitter = None

        print("Num ep.", self._num_episodes)

    def __getitem__(self, episode_id: int) -> Optional[Sample]:
        episode_id %= self._num_episodes
        task, variation, file = self._episodes[episode_id]
        episode = self._cache(file)

        if episode is None:
            return None

        frame_ids = episode[0]
        num_ind = len(frame_ids)
        pad_len = max(0, self._max_episode_length - num_ind)

        states: torch.Tensor = torch.stack([episode[1][i].squeeze(0) for i in frame_ids])
        if states.shape[-1] != 128 or states.shape[-2] != 128:
            raise ValueError(f"{states.shape} {self._episodes[episode_id]}")
        pad_vec = [0] * (2 * states.dim())
        pad_vec[-1] = pad_len
        states = F.pad(states, pad_vec)

        cameras = list(episode[3][0].keys())
        assert all(c in cameras for c in self._cameras)
        index = torch.tensor([cameras.index(c) for c in self._cameras])

        states = states[:, index]
        rgbs = states[:, :, 0]
        pcds = states[:, :, 1]

        if self.jitter is not None:
            rgbs = _apply_jitter(rgbs, self.jitter)

        if "attn" in self._gripper_pose:
            attns = torch.Tensor([])
            for i in frame_ids:
                attn_cams = torch.Tensor([])
                for cam in self._cameras:
                    u, v = episode[3][i][cam]
                    attn = torch.zeros((1, 1, 128, 128))
                    if not (u < 0 or u > 127 or v < 0 or v > 127):
                        attn[0, 0, v, u] = 1
                    attn_cams = torch.cat([attn_cams, attn])
                attns = torch.cat([attns, attn_cams.unsqueeze(0)])
            pad_vec = [0] * (2 * attns.dim())
            pad_vec[-1] = pad_len
            attns = F.pad(attns, pad_vec)
            rgbs = torch.cat([rgbs, attns], 2)

        if self._training:
            modals = self._transform(rgbs=rgbs, pcds=pcds)
            rgbs = modals["rgbs"]
            pcds = modals["pcds"]

        action = torch.cat([episode[2][i] for i in frame_ids])
        shape = [0, 0] * action.dim()
        shape[-1] = pad_len
        action = F.pad(action, tuple(shape), value=0)

        mask = torch.tensor([True] * num_ind + [False] * pad_len)

        instr: Optional[torch.Tensor] = (
            random.choice(list(self._instructions[task][variation]))
            if self._instructions is not None
            else None
        )

        gripper = torch.cat([episode[4][i] for i in frame_ids])
        shape = [0, 0] * gripper.dim()
        shape[-1] = pad_len
        gripper = F.pad(gripper, tuple(shape), value=0)

        tframe_ids = torch.tensor(frame_ids)
        tframe_ids = F.pad(tframe_ids, (0, pad_len), value=-1)

        if self._task_to_id is not None:
            taskvar = torch.Tensor([self._task_to_id[task], variation])
        else:
            taskvar = None

        attn_indices = None # _attn_indices_to_tensor(episode[3], frame_ids, self._cameras)

        return {
            "frame_id": tframe_ids,
            "task": task,
            "variation": variation,
            "rgbs": rgbs,
            "pcds": pcds,
            "action": action,
            "padding_mask": mask,
            "instr": instr,
            "gripper": gripper,
            "taskvar": taskvar,
            "attn_indices": attn_indices,
        }

    def __len__(self):
        if self._num_iters is not None:
            return self._num_iters
        return self._num_episodes


def _apply_jitter(rgbs: torch.Tensor, jitter_fn: Callable):
    T, K, C, H, W = rgbs.shape
    rgbs = (rgbs + 1) * 0.5
    rgbs = rgbs.flatten(0, 1)
    rgbs = jitter_fn(rgbs)
    rgbs = rgbs.view(T, K, C, H, W)
    rgbs = rgbs * 2 - 1
    return rgbs


def _attn_indices_to_tensor(
    indices: List[Dict[str, Tuple[int, int]]],
    frame_ids: List[int],
    cameras: Sequence[CameraName],
) -> torch.Tensor:
    """
    Format the attn indices into a tensor
    """
    attns = -torch.ones((len(frame_ids), len(cameras), 2))
    for i in range(len(frame_ids)):
        for j, cam in enumerate(cameras):
            attns[i, j] = torch.Tensor(indices[i][cam])
    return attns
