import random
from typing import Tuple, Dict, Optional, TypedDict, List
import pickle
import json
from copy import deepcopy
from pathlib import Path
from tqdm import trange
import torch
from torch import nn
import numpy as np
import tap
from filelock import FileLock
from rlbench.demo import Demo
from network import PlainUNet, TransformerUNet
from utils import (
    RLBenchEnv,
    load_episodes,
    get_max_episode_length,
    load_rotation,
    Model,
    Actioner,
    load_instructions,
)
from structures import (
    PointCloudToken,
    BackboneOp,
    TransformerToken,
    GripperPose,
    ZMode,
    PosMode,
    RotMode,
    RotType,
    Position, 
    Workspace
)
from multi_task_baselines import Arguments as TrainArguments, get_dec_len


class Arguments(tap.Tap):
    checkpoint: Optional[Path] = None
    checkpoint_position: Optional[Path] = None
    checkpoint_rotation: Optional[Path] = None
    checkpoint_gripper: Optional[Path] = None
    seed: int = 2
    save_img: bool = False
    device: str = "cuda"
    num_episodes: int = 100
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
    workspace: Workspace = ((-0.325, 0.325, -0.455), (0.455, 0.0, 0.0))
    tasks: Optional[Tuple[str, ...]] = None
    instructions: Optional[Path] = None
    arch: Optional[str] = None
    variations: Tuple[int, ...] = (0,)
    attention: bool = False  # saving attention maps
    # model
    attn_weights: Optional[bool] = None
    backbone: Optional[BackboneOp] = None
    cond: Optional[bool] = None
    depth: Optional[int] = None
    dim_feedforward: Optional[int] = None
    embed_only: Optional[bool] = None
    film: Optional[bool] = None
    film_mlp: Optional[bool] = None
    film_residual: Optional[bool] = None
    gripper_pose: Optional[GripperPose] = None
    hidden_dim: Optional[int] = None
    instr_size: Optional[int] = None
    mask_obs_prob: float = 0.0
    no_residual: Optional[bool] = None
    num_layers: Optional[int] = None
    pcd_token: Optional[PointCloudToken] = None
    rot: Optional[RotMode] = None
    pos: Optional[PosMode] = None
    rot_type: Optional[RotType] = None
    stateless: Optional[bool] = None
    taskvar_token: Optional[bool] = None
    tr_token: Optional[TransformerToken] = None
    temp_len: Optional[int] = None
    z_mode: Optional[ZMode] = None


def get_log_dir(args: Arguments) -> Path:
    log_dir = args.test_xp / args.name
    version = 0
    while (log_dir / f"test-version{version}").is_dir():
        version += 1
    return log_dir / f"test-version{version}"


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


def load_model(checkpoint: Path, args: Arguments) -> Model:
    args = copy_args(checkpoint, args)
    device = torch.device(args.device)

    if not checkpoint.is_file():
        files = list((args.xp / checkpoint).rglob("mtl_*.pth"))
        assert files != [], args.checkpoint
        files = sorted(files, key=lambda x: x.stat().st_mtime)
        checkpoint = files[0]

    print("Loading model from...", checkpoint, flush=True)
    if args.tasks is None:
        raise RuntimeError("Can't find tasks")

    if args.rot is None or args.rot_type is None:
        raise ValueError()
    rotation = load_rotation(args.rot, args.rot_type)

    if args.pos is None:
        raise ValueError()
    position = Position(args.pos, args.workspace)

    dec_len = get_dec_len(args)  # type: ignore

    if args.arch == "mct":
        if (
            args.attn_weights is None
            or args.dim_feedforward is None
            or args.embed_only is None
            or args.gripper_pose is None
            or args.hidden_dim is None
            or args.instr_size is None
            or args.no_residual is None
            or args.num_layers is None
            or args.pcd_token is None
            or args.stateless is None
            or args.taskvar_token is None
            or args.tr_token is None
            or args.z_mode is None
        ):
            raise RuntimeError("Please set these parameters")

        max_episode_length = get_max_episode_length(args.tasks, args.variations)
        model: PlainUNet = TransformerUNet(
            attn_weights=args.attn_weights,
            backbone_op=args.backbone,
            cond=args.cond,
            depth=args.depth,
            dim_feedforward=args.dim_feedforward,
            dec_len=dec_len,
            embed_only=args.embed_only,
            film=args.film,
            film_mlp=args.film_mlp,
            film_residual=args.film_residual,
            gripper_pose=args.gripper_pose,
            hidden_dim=args.hidden_dim,
            instruction=args.instructions is not None,
            max_episode_length=max_episode_length,
            instr_size=args.instr_size,
            no_residual=args.no_residual,
            num_layers=args.num_layers,
            pcd_token=args.pcd_token,
            rot=rotation,
            pos=position,
            stateless=args.stateless,
            temp_len=args.temp_len,
            taskvar_token=args.taskvar_token,
            tr_token=args.tr_token,
            z_mode=args.z_mode,
        ).to(device)
    elif args.arch == "plain":
        if (
            args.backbone is None
            or args.attn_weights is None
            or args.cond is None
            or args.depth is None
            or args.instr_size is None
            or args.film is None
            or args.film_mlp is None
            or args.film_residual is None
            or args.temp_len is None
            or args.z_mode is None
        ):
            raise RuntimeError("Please set these parameters")
        model = PlainUNet(
            attn_weights=args.attn_weights,
            backbone_op=args.backbone,
            cond=args.cond,
            dec_len=dec_len,
            depth=args.depth,
            film=args.film,
            film_mlp=args.film_mlp,
            film_residual=args.film_residual,
            instruction=args.instructions is not None,
            instr_size=args.instr_size,
            rot=rotation,
            pos=position,
            temp_len=args.temp_len,
            z_mode=args.z_mode,
        ).to(device)
    else:
        raise RuntimeError(f"Unexpected arch {args.arch}")

    if hasattr(model, "film_gen") and model.film_gen is not None:
        model.film_gen.build(device)

    model_dict = torch.load(checkpoint, map_location="cpu")
    # DEBUG
    model.load_state_dict(model_dict["weight"], strict=False)
    # model.load_state_dict(model_dict["weight"])
    t_dict = {k: t.to(device) for k, t in model_dict["t"].items()}
    z_dict = {k: z.to(device) for k, z in model_dict["z"].items()}

    model.eval()

    return {"model": model, "t": t_dict, "z": z_dict}


def find_checkpoint(checkpoint: Path) -> Path:
    if checkpoint.is_dir():
        candidates = [c for c in checkpoint.rglob("*.pth") if c.name != "best"]
        candidates = sorted(candidates, key=lambda p: p.lstat().st_mtime)
        assert candidates != [], checkpoint
        return candidates[-1]

    return checkpoint


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)
    log_dir = get_log_dir(args)
    log_dir.mkdir(exist_ok=True, parents=True)
    print("log dir", log_dir)
    # args.save(str(log_dir / "hparams.json"))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load model and args
    if args.checkpoint is not None:
        checkpoint = find_checkpoint(args.checkpoint)
        args = copy_args(checkpoint, args)
        if checkpoint is None:
            raise RuntimeError()
        model = {
            "model": load_model(checkpoint, args),
        }
    elif (
        args.checkpoint_position is not None
        and args.checkpoint_rotation is not None
        and args.checkpoint_gripper is not None
    ):
        checkpoint_position = find_checkpoint(args.checkpoint_position)
        checkpoint_rotation = find_checkpoint(args.checkpoint_rotation)
        checkpoint_gripper = find_checkpoint(args.checkpoint_gripper)
        checkpoint = Path(",".join(
            [str(checkpoint_position), str(checkpoint_rotation), str(checkpoint_gripper)]
        ))
        model = {
            "model_position": load_model(checkpoint_position, args),
            "model_rotation": load_model(checkpoint_rotation, args),
            "model_gripper": load_model(checkpoint_gripper, args),
        }
        args = copy_args(checkpoint_position, args)
    else:
        raise ValueError()

    if args.tasks is None or args.gripper_pose is None or args.taskvar_token is None:
        raise ValueError()

    # load RLBench environment
    env = RLBenchEnv(
        data_path=args.data_dir,
        apply_rgb=True,
        apply_pc=True,
        headless=args.headless,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
        gripper_pose=args.gripper_pose,
    )

    device = torch.device(args.device)
    instructions = load_instructions(args.instructions)
    max_eps_dict = load_episodes()["max_episode_length"]

    for task_str in args.tasks:
        for variation in args.variations:
            demos: Optional[List[Demo]]
            if (
                args.ground_truth_rotation
                or args.ground_truth_position
                or args.ground_truth_gripper
            ):
                print("Loading demos")
                episode_id = -1
                demos = []
                while len(demos) < args.num_episodes:
                    episode_id += 1
                    try:
                        demo = env.get_demo(task_str, variation, episode_id)[0]
                        demos.append(demo)
                    except FileNotFoundError as e:
                        print(e)
                        continue
                    except RuntimeError as e:
                        print(e)
                        continue
                    except IndexError as e:
                        print("Cant find enough samples.")
                        print("Num episodes", episode_id)
                        break
            else:
                demos = None

            actioner = Actioner(
                record_actions=args.record_actions,
                replay_actions=args.replay_actions,
                ground_truth_rotation=args.ground_truth_rotation,
                ground_truth_position=args.ground_truth_position,
                ground_truth_gripper=args.ground_truth_gripper,
                instructions=instructions,
                taskvar_token=args.taskvar_token,
                **model,  # type: ignore
            )

            success_rate = env.evaluate(
                task_str,
                max_episodes=max_eps_dict[task_str],
                variation=variation,
                num_demos=args.num_episodes,
                demos=demos, # type: ignore
                offset=args.offset,
                actioner=actioner,
                log_dir=log_dir / task_str if args.save_img else None,
                max_tries=args.max_tries,
                save_attn=args.attention,
            )

            print("Testing Success Rate {}: {:.04f}".format(task_str, success_rate))

            with FileLock(str(args.output.parent / f"{args.output.name}.lock")):
                with open(args.output, "a") as output_id:
                    output_id.write(
                        f"{task_str}-{variation}, {checkpoint}, seed={args.seed}, {success_rate}, {log_dir}\n"
                    )
