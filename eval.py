import random
from typing import Tuple, Optional
from copy import deepcopy
from pathlib import Path
import torch
import numpy as np
import tap
import json
from filelock import FileLock
from network import Hiveformer
from utils import (
    RLBenchEnv,
    load_episodes,
    get_max_episode_length,
    Actioner,
    load_instructions,
)
from train import Arguments as TrainArguments

from baseline.baseline import Baseline


class Arguments(tap.Tap):
    checkpoint: Path
    seed: int = 2
    save_img: bool = True
    device: str = "cuda"
    num_episodes: int = 2
    headless: bool = False
    offset: int = 0
    name: str = "autol"
    max_tries: int = 10
    output: Path = Path(__file__).parent / "records.txt"
    xp: Path = Path(__file__).parent / "xp"
    test_xp: Path = Path(__file__).parent / "test-xp"
    data_dir: Path = Path(__file__).parent / "demos"
    record_actions: bool = False
    replay_actions: Optional[Path] = None
    ground_truth_rotation: bool = False
    ground_truth_position: bool = False
    ground_truth_gripper: bool = False
    tasks: Optional[Tuple[str, ...]] = None
    instructions: Optional[Path] = "instructions.pkl"
    arch: Optional[str] = None
    variations: Tuple[int, ...] = (0,)
    attention: bool = False  # saving attention maps
    offline: int = 1

    # model
    model: str = "develop"
    depth: Optional[int] = 4
    dim_feedforward: Optional[int] = 64
    hidden_dim: Optional[int] = 64
    instr_size: Optional[int] = 512
    mask_obs_prob: float = 0.0
    num_layers: Optional[int] = 1

    # baseline
    position_loss: str = "ce"  # one of "ce", "mse", "bce"
    sample_ghost_points: int = 1
    position_prediction_only: int = 1
    use_ground_truth_position_for_sampling: int = 0
    embedding_dim: int = 128
    num_ghost_point_cross_attn_layers: int = 4
    num_query_cross_attn_layers: int = 4
    relative_attention: int = 0


def get_log_dir(args: Arguments) -> Path:
    log_dir = args.test_xp / args.name

    def get_log_file(version):
        if len(args.tasks) == 1:
            log_file = f"{args.tasks[0]}_version{version}"
        else:
            log_file = f"version{version}"
        return log_file

    version = 0
    while (log_dir / get_log_file(version)).is_dir():
        version += 1

    return log_dir / get_log_file(version)


def copy_args(checkpoint: Path, args: Arguments) -> Arguments:
    args = deepcopy(args)

    if not checkpoint.is_file():
        files = list((args.xp / checkpoint).rglob("mtl_*.pth"))
        assert files != [], args.checkpoint
        files = sorted(files, key=lambda x: x.stat().st_mtime)
        checkpoint = files[0]

    print("Copying args from", checkpoint)

    # Update args accordingly:
    hparams = checkpoint.parent / "hparams.json"
    print(hparams, hparams.is_file())
    if hparams.is_file():
        print("Loading args from checkpoint")
        train_args = TrainArguments()
        train_args.load(str(hparams))
        for key in args.class_variables:
            v = getattr(args, key)
            if v is None and key in train_args.class_variables:
                setattr(args, key, getattr(train_args, key))
                print("Copying", key, ":", getattr(args, key))

    return args


def load_model(checkpoint: Path, args: Arguments) -> Hiveformer:
    args = copy_args(checkpoint, args)
    device = torch.device(args.device)

    if not checkpoint.is_file():
        files = list((args.xp / checkpoint).rglob("mtl_*.pth"))
        assert files != [], args.checkpoint
        files = sorted(files, key=lambda x: x.stat().st_mtime)
        checkpoint = files[0]

    print("Loading model from", checkpoint, flush=True)

    if (
        args.depth is None
        or args.dim_feedforward is None
        or args.hidden_dim is None
        or args.instr_size is None
        or args.mask_obs_prob is None
        or args.num_layers is None
    ):
        raise ValueError("Please provide the missing parameters")

    max_episode_length = get_max_episode_length(args.tasks, args.variations)

    if args.model == "original":
        model = Hiveformer(
            depth=args.depth,
            dim_feedforward=args.dim_feedforward,
            hidden_dim=args.hidden_dim,
            instr_size=args.instr_size,
            mask_obs_prob=args.mask_obs_prob,
            max_episode_length=max_episode_length,
            num_layers=args.num_layers,
        ).to(device)
    elif args.model == "develop":
        if len(args.tasks) == 1:
            model = Baseline(
                depth=args.depth,
                dim_feedforward=args.dim_feedforward,
                hidden_dim=args.hidden_dim,
                instr_size=args.instr_size,
                mask_obs_prob=args.mask_obs_prob,
                max_episode_length=max_episode_length,
                num_layers=args.num_layers,
                gripper_loc_bounds=json.load(open("location_bounds.json", "r"))[args.tasks[0]],
                sample_ghost_points=bool(args.sample_ghost_points),
                use_ground_truth_position_for_sampling=bool(args.use_ground_truth_position_for_sampling),
                position_loss=args.position_loss,
                embedding_dim=args.embedding_dim,
                num_ghost_point_cross_attn_layers=args.num_ghost_point_cross_attn_layers,
                num_query_cross_attn_layers=args.num_query_cross_attn_layers,
                relative_attention=bool(args.relative_attention)
            ).to(device)
        else:
            raise NotImplementedError

    if hasattr(model, "film_gen") and model.film_gen is not None:
        model.film_gen.build(device)

    model_dict = torch.load(checkpoint, map_location="cpu")["weight"]
    model_dict = {(k[7:] if k.startswith("module.") else k): v
                  for k, v in model_dict.items()}
    model.load_state_dict(model_dict)

    model.eval()

    return model


def find_checkpoint(checkpoint: Path) -> Path:
    if checkpoint.is_dir():
        candidates = [c for c in checkpoint.rglob("*.pth") if c.name != "best"]
        candidates = sorted(candidates, key=lambda p: p.lstat().st_mtime)
        assert candidates != [], checkpoint
        return candidates[-1]

    return checkpoint


if __name__ == "__main__":
    args = Arguments().parse_args()

    if args.tasks is None:
        print(args.checkpoint)
        args.tasks = ["_".join(str(args.checkpoint).split("/")[-2].split("_")[:-1])]
        print(f"Automatically setting task to {args.tasks}")

    log_dir = get_log_dir(args)
    log_dir.mkdir(exist_ok=True, parents=True)
    print("log dir", log_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load model and args
    checkpoint = find_checkpoint(args.checkpoint)
    args = copy_args(checkpoint, args)
    if checkpoint is None:
        raise RuntimeError()
    model = load_model(checkpoint, args)

    # load RLBench environment
    env = RLBenchEnv(
        # data_path="",
        data_path=args.data_dir,
        apply_rgb=True,
        apply_pc=True,
        headless=args.headless,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
    )

    instruction = load_instructions(args.instructions)
    if instruction is None:
        raise NotImplementedError()

    actioner = Actioner(model=model, instructions=instruction)
    max_eps_dict = load_episodes()["max_episode_length"]
    for task_str in args.tasks:
        for variation in args.variations:
            success_rate = env.evaluate(
                task_str,
                max_episodes=max_eps_dict[task_str],
                variation=variation,
                num_demos=args.num_episodes,
                offset=args.offset,
                actioner=actioner,
                log_dir=log_dir / task_str if args.save_img else None,
                max_tries=args.max_tries,
                save_attn=args.attention,
                record_videos=True,
                position_prediction_only=bool(args.position_prediction_only),
                offline=bool(args.offline)
            )

            print("Testing Success Rate {}: {:.04f}".format(task_str, success_rate))

            with FileLock(str(args.output.parent / f"{args.output.name}.lock")):
                with open(args.output, "a") as output_id:
                    output_id.write(
                        f"{task_str}-{variation}, {checkpoint}, seed={args.seed}, {success_rate}, {log_dir}\n"
                    )
