import random
from typing import List, Tuple, Dict, Optional, Any, Union
import itertools
import pickle
import os
from pathlib import Path
import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
import numpy as np
from tqdm import tqdm, trange
from filelock import FileLock
import tap
from network import (
    PlainUNet,
    TransformerUNet,
)
from utils import (
    LossAndMetrics,
    load_instructions,
    load_rotation,
    Model,
    RLBenchEnv,
    count_parameters,
    load_episodes,
    get_max_episode_length,
    Actioner,
)
from structures import (
    BackboneOp,
    GripperPose,
    PointCloudToken,
    Position,
    PosMode,
    RotMode,
    RotType,
    TransformerToken,
    Workspace,
    ZMode,
)
from create_data import RLBench


class Arguments(tap.Tap):
    accumulate_grad_batches: int = 1
    cameras: Tuple[str, ...] = ("wrist", "left_shoulder", "right_shoulder")
    checkpoint: Optional[Path] = None
    checkpoint_period: int = 10
    dataset: List[Path]
    device: str = "cuda"
    xp: Path = Path(__file__).parent / "xp"
    valset: Optional[Tuple[Path, ...]] = None
    name: str = "multitask"
    arch: str = "mct"
    num_workers: int = 5
    max_tries: int = 10
    max_episodes_per_taskvar: int = 100
    instructions: Optional[Path] = None
    cache_size: int = 100
    seed: int = 2

    tasks: Tuple[str, ...]
    variations: Tuple[int, ...] = (0,)

    workspace: Workspace = ((-0.325, 0.325, -0.455), (0.455, 0.0, 0.0))

    # Train
    batch_size: int = 32
    lr: float = 0.001
    lr_rotation: float = 0.001
    lr_transformer: float = 0.001
    val_freq: int = 200
    val_batch_size: int = 100
    train_iters: int = 100_000
    jitter: bool = False

    # tests
    headless: bool = False
    output: Path = Path(__file__).parent / "records.txt"

    # model
    attn_weights: bool = False
    avg_instr: bool = False
    backbone: BackboneOp = "cat"
    cond: bool = False
    depth: int = 4
    dim_feedforward: int = 64
    embed_only: bool = False
    film: bool = False
    film_mlp: bool = False
    film_residual: bool = False
    gripper_pose: GripperPose = "none"
    hidden_dim: int = 64
    instr_size: int = 512
    mask_obs_prob: float = 0.0
    no_residual: bool = False
    num_layers: int = 1
    pcd_token: PointCloudToken = "none"
    pcd_noise: bool = False
    pos: PosMode = "mse"
    rot: RotMode = "mse"
    rot_type: RotType = "quat"
    stateless: bool = False
    taskvar_token: bool = False
    tr_token: TransformerToken = "tnc"
    temp_len: int = 64
    z_mode: ZMode = "embed"


def training(
    model: nn.Module,
    t_dict: Dict[str, torch.Tensor],
    z_dict: Dict[str, torch.Tensor],
    optimizer,
    train_loader,
    val_loaders,
    checkpointer,
    loss_and_metrics,
    args: Arguments,
    writer: SummaryWriter,
):
    iter_loader = iter(train_loader)
    device = t_dict[list(t_dict.keys())[0]].device
    with trange(args.train_iters) as tbar:
        for step_id in tbar:
            try:
                sample = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                sample = next(iter_loader)

            rgbs = sample["rgbs"].to(device)
            pcds = sample["pcds"].to(device)
            gripper = sample["gripper"].to(device)
            outputs = sample["action"].to(device)
            padding_mask = sample["padding_mask"].to(device)

            instr = sample["instr"]
            if instr is not None:
                instr = instr.to(device)

            taskvar = sample["taskvar"]
            if taskvar is not None:
                taskvar = taskvar.to(device)

            frame_id = sample["frame_id"]
            tasks = sample["task"]
            t = torch.stack([t_dict[task][fid] for task, fid in zip(tasks, frame_id)])
            z = torch.stack([z_dict[task][fid] for task, fid in zip(tasks, frame_id)])

            if step_id % args.accumulate_grad_batches == 0:
                optimizer.zero_grad()

            pred = model(
                rgbs,
                pcds,
                padding_mask,
                t,
                z,
                instr,
                gripper,
                taskvar,
            )

            train_losses = loss_and_metrics.compute_loss(pred, sample)
            train_losses["total"] = sum(list(train_losses.values()))  # type: ignore

            for n, l in train_losses.items():
                writer.add_scalar(f"train-loss/{n}", l, step_id)

            writer.add_scalar(f"lr/", args.lr, step_id)

            metrics = loss_and_metrics.compute_metrics(pred, sample)
            for n, l in metrics.items():
                writer.add_scalar(f"train-metrics/{n}", l, step_id)

            train_losses["total"].backward()  # type: ignore

            if step_id % args.accumulate_grad_batches == args.accumulate_grad_batches - 1:
                optimizer.step()

            if (step_id + 1) % args.val_freq == 0:
                if val_loaders is not None:
                    val_metrics = validation_step(
                        step_id,
                        val_loaders,
                        model,
                        t_dict,
                        z_dict,
                        writer,
                        loss_and_metrics,
                    )
                    model.train()
                else:
                    val_metrics = {}
                checkpointer(val_metrics)

            tbar.set_postfix(l=float(train_losses["total"]))


def get_dec_len(args: Arguments) -> int:
    if args.arch == "mct":
        dec_len = 0
        if args.backbone == "cat":
            dec_len += args.temp_len
            if "tnc" == args.tr_token:
                dec_len += 16
                if args.pcd_token:
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


def get_log_dir(args: Arguments) -> Path:
    log_dir = args.xp / args.name
    version = int(os.environ.get("SLURM_JOBID", 0))
    while (log_dir / f"version{version}").is_dir():
        version += 1
    return log_dir / f"version{version}"


class CheckpointCallback:
    def __init__(
        self,
        name: str,
        log_dir: Path,
        state_dict: Any,
        minimizing: bool = True,
        checkpoint_period: int = 200,
    ):
        self._name = name
        self._minimizing = minimizing
        self._best = float("inf") if minimizing else -float("inf")
        self._log_dir = log_dir
        self._checkpoint_period = checkpoint_period
        self._step = 0
        self._state_dict = state_dict

    def __call__(self, metrics: Dict[str, torch.Tensor]):
        self._step += 1
        if self._step % self._checkpoint_period != 0:
            return

        value = int(metrics.get(self._name, 0))
        dest = self._log_dir / f"model.step={self._step}-value={value}.pth"
        torch.save(self._state_dict, dest)

        if (self._minimizing and self._best > value) or (
            not self._minimizing and self._best < value
        ):
            best = self._log_dir / "best.pth"
            best.unlink(missing_ok=True)
            best.symlink_to(dest.resolve())
            self._best = value


@torch.no_grad()
def validation_step(
    step_id: int,
    val_loaders: List[DataLoader],
    model,
    t_dict,
    z_dict,
    writer,
    loss_and_metrics,
    val_iters: int = 5,
):
    device = t_dict[list(t_dict.keys())[0]].device
    values = {}
    model.eval()

    for val_id, val_loader in enumerate(val_loaders):
        for i, sample in enumerate(val_loader):
            if i == val_iters:
                break

            rgbs = sample["rgbs"].to(device)
            pcds = sample["pcds"].to(device)
            gripper = sample["gripper"].to(device)
            outputs = sample["action"].to(device)
            padding_mask = sample["padding_mask"].to(device)

            instr = sample["instr"]
            if instr is not None:
                instr = instr.to(device)

            taskvar = sample["taskvar"]
            if taskvar is not None:
                taskvar = taskvar.to(device)

            frame_id = sample["frame_id"]
            tasks = sample["task"]
            t = torch.stack([t_dict[task][fid] for task, fid in zip(tasks, frame_id)])
            z = torch.stack([z_dict[task][fid] for task, fid in zip(tasks, frame_id)])

            pred = model(
                rgbs,
                pcds,
                padding_mask,
                t,
                z,
                instr,
                gripper,
                taskvar,
            )

            losses: Dict[str, torch.Tensor] = loss_and_metrics.compute_loss(pred, sample)
            losses["total"] = torch.stack(list(losses.values())).sum()

            for n, l in losses.items():
                key = f"val-loss-{val_id}/{n}"
                writer.add_scalar(key, l, step_id + i)
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

            writer.add_scalar(f"lr/", args.lr, step_id + i)

            metrics = loss_and_metrics.compute_metrics(pred, sample)
            for n, l in metrics.items():
                key = f"val-metrics-{val_id}/{n}"
                writer.add_scalar(key, l, step_id + i)
                if key not in metrics:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

        key = f"val-loss-{val_id}/total"
        print(f"Validation Loss {val_id}: {values[key].mean():.05f}")
        key = f"val-metrics-{val_id}/position"
        print(f"Validation Position {val_id}: {values[key].mean():.05f}")

    return values


def collate_fn(batch: List[Dict]):
    keys = batch[0].keys()
    return {
        key: default_collate([item[key] for item in batch])
        if batch[0][key] is not None
        else None
        for key in keys
    }


def get_train_loader(args: Arguments) -> DataLoader:
    instruction = load_instructions(
        args.instructions, tasks=args.tasks, variations=args.variations
    )

    if instruction is None:
        taskvar = list(itertools.product(args.tasks, args.variations))
    else:
        taskvar = [
            (task, var)
            for task, var_instr in instruction.items()
            for var in var_instr.keys()
        ]
    print(f"Valset has {len(taskvar)} taskvars")

    max_episode_length = get_max_episode_length(args.tasks, args.variations)

    dataset = RLBench(
        root=args.dataset,
        taskvar=taskvar,
        instructions=instruction,
        gripper_pose=args.gripper_pose,
        taskvar_token=args.taskvar_token,
        max_episode_length=max_episode_length,
        max_episodes_per_taskvar=args.max_episodes_per_taskvar,
        cache_size=args.cache_size,
        num_iters=args.train_iters,
        jitter=args.jitter,
        cameras=args.cameras,  # type: ignore
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    return loader


def get_val_loaders(args: Arguments) -> Optional[List[DataLoader]]:
    if args.valset is None:
        return None

    instruction = load_instructions(
        args.instructions, tasks=args.tasks, variations=args.variations
    )

    if instruction is None:
        taskvar = list(itertools.product(args.tasks, args.variations))
    else:
        taskvar = [
            (task, var)
            for task, var_instr in instruction.items()
            for var in var_instr.keys()
        ]
    print(f"Valset has {len(taskvar)} taskvars")

    max_episode_length = get_max_episode_length(args.tasks, args.variations)
    loaders = []

    for valset in args.valset:
        dataset = RLBench(
            root=valset,
            taskvar=taskvar,
            instructions=instruction,
            gripper_pose=args.gripper_pose,
            taskvar_token=args.taskvar_token,
            max_episode_length=max_episode_length,
            max_episodes_per_taskvar=args.max_episodes_per_taskvar,
            cache_size=args.cache_size,
            training=False,
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        loaders.append(loader)

    print(len(loaders), "validation loaders")

    return loaders


def get_model(args: Arguments) -> Tuple[optim.Optimizer, Model, LossAndMetrics]:
    rotation = load_rotation(args.rot, args.rot_type)
    position = Position(args.pos, args.workspace)
    device = torch.device(args.device)

    max_eps_dict = load_episodes()["max_episode_length"]
    t_dict: Dict[str, torch.Tensor] = {}
    z_dict: Dict[str, torch.Tensor] = {}
    for task_str, max_eps in max_eps_dict.items():
        t_dict[task_str] = torch.rand(
            max_eps, args.temp_len, requires_grad=True, device=device
        )
        z_dict[task_str] = torch.zeros(max_eps, 3, requires_grad=True, device=device)

    dec_len = get_dec_len(args)

    if args.arch == "mct":
        max_episode_length = get_max_episode_length(args.tasks, args.variations)
        model: PlainUNet = TransformerUNet(
            avg_instr=args.avg_instr,
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
            instr_size=args.instr_size,
            instruction=args.instructions is not None,
            mask_obs_prob=args.mask_obs_prob,
            max_episode_length=max_episode_length,
            no_residual=args.no_residual,
            num_layers=args.num_layers,
            pcd_token=args.pcd_token,
            pcd_noise=args.pcd_noise,
            taskvar_token=args.taskvar_token,
            rot=rotation,
            pos=position,
            stateless=args.stateless,
            temp_len=args.temp_len,
            tr_token=args.tr_token,
            z_mode=args.z_mode,
        ).to(device)
        optimizer_grouped_parameters = [
            {"params": list(t_dict.values()), "weight_decay": 0.0, "lr": args.lr},
            {"params": list(z_dict.values()), "weight_decay": 5e-4, "lr": args.lr},
            {"params": [], "weight_decay": 5e-4, "lr": args.lr_transformer},
            {"params": [], "weight_decay": 0.0, "lr": args.lr_rotation},
        ]
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        transformer_modules = ["encoder"]
        rotation_modules = ["quat_decoder"]
        for name, param in model.named_parameters():
            if any(nd in name for nd in no_decay):
                optimizer_grouped_parameters[0]["params"].append(param)  # type: ignore
            elif any(nd in name for nd in transformer_modules):
                optimizer_grouped_parameters[2]["params"].append(param)  # type: ignore
            elif any(nd in name for nd in rotation_modules):
                optimizer_grouped_parameters[3]["params"].append(param)  # type: ignore
            else:
                optimizer_grouped_parameters[1]["params"].append(param)  # type: ignore
        optimizer: optim.Optimizer = optim.AdamW(optimizer_grouped_parameters)

    elif args.arch == "plain":
        model = PlainUNet(
            gripper_pose=args.gripper_pose,
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
        optimizer = optim.Adam(
            [
                {"params": list(model.parameters()) + list(t_dict.values())},  # type: ignore
                {"params": list(z_dict.values()), "weight_decay": 5e-4},
            ],
            lr=args.lr,
        )
    else:
        raise RuntimeError(f"Unexpected arch {args.arch}")

    if args.checkpoint is not None:
        model_dict = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(model_dict["weight"])
        t_dict = {k: t.to(device) for k, t in model_dict["t"].items()}
        z_dict = {k: z.to(device) for k, z in model_dict["z"].items()}
        optimizer.load_state_dict(model_dict["optimizer"])

    print("Number of parameters:")
    model_params = count_parameters(model)
    zlist_params = np.sum(list(max_eps_dict.values())) * 3
    tlist_params = np.sum(list(max_eps_dict.values())) * args.temp_len
    print("- model", model_params)
    print("- z list", zlist_params)
    print("- t list", tlist_params)
    if hasattr(model, "film_gen") and model.film_gen is not None:
        model.film_gen.build(device)
        print("FiLM:", count_parameters(model.film_gen))
    print("Total", model_params + zlist_params + tlist_params)

    position = Position(args.pos, args.workspace)

    loss_and_metrics = LossAndMetrics(rotation, position)

    meta_model = {"model": model, "t": t_dict, "z": z_dict}

    return optimizer, meta_model, loss_and_metrics  # type: ignore


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)
    log_dir = get_log_dir(args)
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Logging:", log_dir)
    args.save(str(log_dir / "hparams.json"))
    writer = SummaryWriter(log_dir=log_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

    optimizer, meta_model, loss_and_metrics = get_model(args)
    model = meta_model["model"]
    t_dict = meta_model["t"]
    z_dict = meta_model["z"]

    # training episode
    model_dict = {
        "weight": model.state_dict(),
        "t": t_dict,
        "z": z_dict,
        "optimizer": optimizer.state_dict(),
    }
    checkpointer = CheckpointCallback(
        "val-metrics/position",
        log_dir,
        model_dict,
        minimizing=False,
        checkpoint_period=args.checkpoint_period,
    )
    model.train()

    val_loaders = get_val_loaders(args)

    if args.train_iters > 0:
        train_loader = get_train_loader(args)
        training(
            model,
            t_dict,
            z_dict,
            optimizer,
            train_loader,
            val_loaders,
            checkpointer,
            loss_and_metrics,
            args,
            writer,
        )

    if val_loaders is not None:
        val_metrics = validation_step(
            args.train_iters,
            val_loaders,
            model,
            t_dict,
            z_dict,
            writer,
            loss_and_metrics,
            val_iters=-1,
        )

    # last checkpoint
    checkpoint = log_dir / f"mtl_{args.seed}_{args.lr}.pth"
    torch.save(model_dict, checkpoint)

    # evaluation
    model.eval()

    env = RLBenchEnv(
        data_path="",
        apply_rgb=True,
        apply_pc=True,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
        headless=args.headless,
        gripper_pose=args.gripper_pose,
    )

    instruction = load_instructions(args.instructions)
    actioner = Actioner(
        model={"model": model, "t": t_dict, "z": z_dict},  # type: ignore
        instructions=instruction,
        taskvar_token=args.taskvar_token,
    )
    max_eps_dict = load_episodes()["max_episode_length"]
    for task_str in args.tasks:
        for variation in args.variations:
            success_rate = env.evaluate(
                task_str,
                actioner=actioner,
                max_episodes=max_eps_dict[task_str],
                variation=variation,
                num_demos=500,
                demos=None,
                log_dir=log_dir,
                max_tries=args.max_tries,
            )

            print("Testing Success Rate {}: {:.04f}".format(task_str, success_rate))

            with FileLock(args.output.parent / f"{args.output.name}.lock"):
                with open(args.output, "a") as oid:
                    oid.write(
                        f"{task_str}-{variation}, {checkpoint}, seed={args.seed}, {success_rate}\n"
                    )
