import random
from typing import Tuple, Dict, Optional, TypedDict, List
import pickle
import json
from copy import deepcopy
from pathlib import Path
from tqdm import trange
import tap
import torch
from torch import nn
import numpy as np
import tap
from tap.utils import as_python_object
from filelock import FileLock
from network_sim2real import PlainUNet, TransformerUNet
from utils_sim2real import (
    Sim2RealEnv,
    load_episodes,
    get_max_episode_length,
    RotMode,
    RotType,
    load_rotation,
    Model,
    BackboneOp,
    TransformerToken,
    PointCloudToken,
    GripperPose,
    ZMode,
    InstructionMode,
)
import muse.envs
import gym


class Arguments(tap.Tap):
    task: str
    env_name: str
    checkpoint: Optional[Path] = None
    seed: int = 2
    save_img: bool = False
    device: str = "cuda"
    num_episodes: int = 100
    headless: bool = False
    offset: int = 0
    name: str = "autol"
    output: Path = Path(__file__).parent / "records.txt"
    xp: Path = Path(__file__).parent / "xp"
    test_xp: Path = Path(__file__).parent / "test-xp"
    data_dir: Path = Path(__file__).parent / "demos"
    instructions: Optional[Path] = None
    instr_mode: InstructionMode = "precompute"
    variations: Tuple[int, ...] = (0,)
    attention: bool = False  # saving attention maps
    argmax: bool = False  # predict 3D poitions from attn map (argmax)

    # model
    arch: Optional[str] = None
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
        train_args = Arguments()
        with open(hparams) as f:
            args_dict = json.load(f, object_hook=as_python_object)
        args_dict["task"] = args.task
        args_dict["env_name"] = args.env_name
        train_args.from_dict(args_dict)
        for key in args.class_variables:
            v = getattr(args, key)
            if v is None and key in train_args.class_variables:
                setattr(args, key, getattr(train_args, key))
                print("Copying", key, ":", getattr(args, key))

    return args


def get_dec_len(args: Arguments) -> int:
    if args.temp_len is None or args.hidden_dim is None:
        raise RuntimeError()

    if args.arch == "mct":
        dec_len = 0
        if args.backbone == "cat":
            dec_len += args.temp_len
            if "tnc" == args.tr_token:
                dec_len += 16
                if args.pcd_token != "none":
                    dec_len += 3
            elif "tnhw_cm_sa" == args.tr_token:
                dec_len += 2 * args.hidden_dim
            else:
                dec_len += args.hidden_dim

            if not args.no_residual:
                dec_len += 16

        else:
            dec_len = 16

    elif args.arch == "plain":
        dec_len = 16
        if args.backbone == "cat":
            dec_len += args.temp_len

    else:
        raise RuntimeError()

    return dec_len


def load_model(checkpoint: Path, args: Arguments) -> Model:
    args = copy_args(checkpoint, args)
    device = torch.device(args.device)

    if not checkpoint.is_file():
        files = list((args.xp / checkpoint).rglob("mtl_*.pth"))
        assert files != [], args.checkpoint
        files = sorted(files, key=lambda x: x.stat().st_mtime)
        checkpoint = files[0]

    print("Loading model from...", checkpoint, flush=True)

    if args.rot is None or args.rot_type is None:
        raise ValueError()
    rotation = load_rotation(args.rot, args.rot_type)

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

        max_episode_length = get_max_episode_length((args.task,), args.variations)
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
            or args.temp_len is None
            or args.film is None
            or args.film_mlp is None
            or args.film_residual is None
            or args.instr_size is None
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
    if args.checkpoint is None:
        raise RuntimeError()
    checkpoint = find_checkpoint(args.checkpoint)
    args = copy_args(checkpoint, args)
    model = {
        "model": load_model(checkpoint, args),
    }

    if args.gripper_pose is None or args.taskvar_token is None:
        raise ValueError()

    device = torch.device(args.device)
    max_eps_dict = load_episodes()["max_episode_length"]

    for variation in args.variations:
        if "PushButtons" in args.env_name:
            if "DR" in args.env_name:
                variant_env_name = "-".join(args.env_name.split("-")[1:])
                variant_env_name = f"DR-Var{variation}-{variant_env_name}"
            else:
                variant_env_name = f"Var{variation}-{args.env_name}"
        else:
            variant_env_name = args.env_name

        env = Sim2RealEnv(
            model=model["model"],
            instructions=args.instructions,
            instr_mode=args.instr_mode,
            gripper_pose=args.gripper_pose,
        )

        success_rate = env.evaluate(
            task_str=args.task,
            env_str=variant_env_name,
            max_episodes=max_eps_dict[args.task],
            variation=variation,
            num_demos=args.num_episodes,
            device=device,
            offset=args.offset,
            log_dir=log_dir / args.task if args.save_img else None,
            save_attn=args.attention,
            seed=args.seed,
        )

        print("Testing Success Rate {}: {:.04f}".format(args.task, success_rate))

        with FileLock(str(args.output.parent / f"{args.output.name}.lock")):
            with open(args.output, "a") as output_id:
                output_id.write(
                    f"{args.task}-{variation}, {checkpoint}, seed={args.seed}, {success_rate}, {log_dir}\n"
                )
