import random
from typing import List, Tuple, Dict, Optional, Any
import os
from collections import defaultdict
from pathlib import Path
import torch
import json
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import numpy as np
from tqdm import tqdm, trange
import tap

from utils.utils_without_rlbench import (
    LossAndMetrics,
    load_instructions,
    count_parameters,
    get_max_episode_length,
)
from dataset import RLBenchDataset
from model.released_hiveformer.network import Hiveformer
from model.non_analogical_baseline.baseline import Baseline


class Arguments(tap.Tap):
    cameras: Tuple[str, ...] = ("wrist", "left_shoulder", "right_shoulder")
    image_size: str = "128,128"
    max_tries: int = 10
    max_episodes_per_taskvar: int = 100
    instructions: Optional[Path] = "instructions.pkl"
    cache_size: int = 100
    seed: int = 2
    tasks: Tuple[str, ...]
    variations: Tuple[int, ...] = (0,)
    checkpoint: Optional[Path] = None
    accumulate_grad_batches: int = 1
    val_freq: int = 500
    checkpoint_freq: int = 10

    # Training and validation datasets
    dataset: List[Path]
    valset: Optional[Tuple[Path, ...]] = None

    # Logging
    base_log_dir: Path = Path(__file__).parent / "train_logs"
    exp_log_dir: str = "exp"
    run_log_dir: str = "run"

    # Main training parameters
    devices: List[str] = ["cuda:0"]  # ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    num_workers: int = 5 * len(devices)
    batch_size: int = 10 * len(devices)
    lr: float = 5e-5
    train_iters: int = 200_000 // len(devices)

    # Toggle to switch between original HiveFormer and our models
    model: str = "baseline"  # one of "original", "baseline"

    # ---------------------------------------------------------------
    # Original HiveFormer parameters
    # ---------------------------------------------------------------

    depth: int = 4
    dim_feedforward: int = 64
    hidden_dim: int = 64
    instr_size: int = 512
    mask_obs_prob: float = 0.0
    num_layers: int = 1

    # ---------------------------------------------------------------
    # Our non-analogical baseline parameters
    # ---------------------------------------------------------------

    # Loss
    position_prediction_only: int = 1
    position_loss: str = "ce"  # one of "ce", "mse", "bce"
    ground_truth_gaussian_spread: float = 0.01
    compute_loss_at_all_layers: int = 1
    position_loss_coeff: float = 1.0
    rotation_loss_coeff: float = 1.0
    gripper_loss_coeff: float = 1.0
    label_smoothing: float = 0.1

    # Ghost points
    coarse_to_fine_sampling: int = 1
    fine_sampling_cube_size: float = 0.05
    num_ghost_points: int = 1000
    use_ground_truth_position_for_sampling: int = 1

    # Model
    embedding_dim: int = 60
    num_ghost_point_cross_attn_layers: int = 2
    num_query_cross_attn_layers: int = 2
    rotation_pooling_gaussian_spread: float = 0.01  # if 0, no pooling


def training(
    model: nn.Module,
    optimizer,
    train_loader,
    val_loaders,
    checkpointer,
    loss_and_metrics,
    args: Arguments,
    writer: SummaryWriter,
):
    iter_loader = iter(train_loader)

    aggregated_losses = defaultdict(list)
    aggregated_metrics = defaultdict(list)

    with trange(args.train_iters) as tbar:
        for step_id in tbar:
            try:
                sample = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                sample = next(iter_loader)

            if step_id % args.accumulate_grad_batches == 0:
                optimizer.zero_grad()

            model_type = type(model)
            if model_type == nn.DataParallel:
                model_type = type(model.module)
            if model_type == Hiveformer:
                pred = model(
                    sample["rgbs"],
                    sample["pcds"],
                    sample["padding_mask"],
                    sample["instr"],
                    sample["gripper"]
                )
            elif model_type == Baseline:
                pred = model(
                    sample["rgbs"],
                    sample["pcds"],
                    sample["padding_mask"],
                    sample["instr"],
                    sample["gripper"],
                    # Provide ground-truth action to sample ghost points at training time
                    sample["action"]
                )

            train_losses = loss_and_metrics.compute_loss(pred, sample, model)
            train_losses["total"] = sum(list(train_losses.values()))  # type: ignore
            train_losses["total"].backward()  # type: ignore

            metrics = loss_and_metrics.compute_metrics(pred, sample)

            for n, l in train_losses.items():
                aggregated_losses[n].append(l)
            for n, l in metrics.items():
                aggregated_metrics[n].append(l)

            if step_id % args.accumulate_grad_batches == args.accumulate_grad_batches - 1:
                optimizer.step()

            if (step_id + 1) % args.val_freq == 0:
                writer.add_scalar(f"lr/", args.lr, step_id)

                for n, l in aggregated_losses.items():
                    writer.add_scalar(f"train-loss/{n}", torch.mean(torch.stack(l)), step_id)
                for n, l in aggregated_metrics.items():
                    writer.add_scalar(f"train-metrics/{n}", torch.mean(torch.stack(l)), step_id)
                aggregated_losses = defaultdict(list)
                aggregated_metrics = defaultdict(list)

                for n, l in train_losses.items():
                    aggregated_losses[n].append(l)
                for n, l in metrics.items():
                    aggregated_metrics[n].append(l)

                if val_loaders is not None:
                    val_metrics = validation_step(
                        step_id,
                        val_loaders,
                        model,
                        writer,
                        loss_and_metrics,
                    )
                    model.train()
                else:
                    val_metrics = {}
                checkpointer(val_metrics)

            tbar.set_postfix(l=float(train_losses["total"]))


def get_log_dir(args: Arguments) -> Path:
    log_dir = args.base_log_dir / args.exp_log_dir

    def get_log_file(version):
        log_file = f"{args.run_log_dir}_version{version}"
        return log_file

    version = int(os.environ.get("SLURM_JOBID", 0))
    while (log_dir / get_log_file(version)).is_dir():
        version += 1

    return log_dir / get_log_file(version)


class CheckpointCallback:
    def __init__(
        self,
        name: str,
        log_dir: Path,
        state_dict: Any,
        val_freq: int,
        minimizing: bool = True,
        checkpoint_freq: int = 10,
    ):
        self._name = name
        self._minimizing = minimizing
        self._best = float("inf") if minimizing else -float("inf")
        self._log_dir = log_dir
        self._checkpoint_freq = checkpoint_freq
        self._step = 0
        self._val_freq = val_freq
        self._state_dict = state_dict

    def __call__(self, metrics: Dict[str, torch.Tensor]):
        self._step += 1
        if self._step % self._checkpoint_freq != 0:
            return

        value = metrics.get(self._name, torch.tensor(0))
        dest = self._log_dir / f"model.step={self._step * self._val_freq}-value={value.item():.5f}.pth"
        torch.save(self._state_dict, dest)

        cond1 = self._name not in metrics
        cond2 = self._name in metrics and ((self._minimizing and self._best > value) or (not self._minimizing and self._best < value))

        if cond1 or cond2:
            best = self._log_dir / "best.pth"
            best.unlink(missing_ok=True)
            best.symlink_to(dest.resolve())
            self._best = value


@torch.no_grad()
def validation_step(
    step_id: int,
    val_loaders: List[DataLoader],
    model,
    writer,
    loss_and_metrics,
    val_iters: int = 5,
):
    values = {}
    device = next(model.parameters()).device
    model.eval()

    for val_id, val_loader in enumerate(val_loaders):
        for i, sample in enumerate(val_loader):
            if i == val_iters:
                break

            model_type = type(model)
            if model_type == nn.DataParallel:
                model_type = type(model.module)
            if model_type == Hiveformer:
                pred = model(
                    sample["rgbs"],
                    sample["pcds"],
                    sample["padding_mask"],
                    sample["instr"],
                    sample["gripper"]
                )
            elif model_type == Baseline:
                pred = model(
                    sample["rgbs"],
                    sample["pcds"],
                    sample["padding_mask"],
                    sample["instr"],
                    sample["gripper"],
                    # DO NOT provide ground-truth action to sample ghost points at validation time
                    # sample["action"]
                )

            losses: Dict[str, torch.Tensor] = loss_and_metrics.compute_loss(pred, sample, model)
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

        print()
        print(f"Step {step_id}:")
        for key, value in values.items():
            print(f"{key}: {value.mean():.03f}")

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
        raise NotImplementedError()
    else:
        taskvar = [
            (task, var)
            for task, var_instr in instruction.items()
            for var in var_instr.keys()
        ]

    max_episode_length = get_max_episode_length(args.tasks, args.variations)

    dataset = RLBenchDataset(
        root=args.dataset,
        image_size=tuple(int(x) for x in args.image_size.split(",")),  # type: ignore
        taskvar=taskvar,
        instructions=instruction,
        max_episode_length=max_episode_length,
        max_episodes_per_taskvar=args.max_episodes_per_taskvar,
        cache_size=args.cache_size,
        num_iters=args.train_iters,
        cameras=args.cameras,  # type: ignore
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
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
        raise NotImplementedError()
    else:
        taskvar = [
            (task, var)
            for task, var_instr in instruction.items()
            for var in var_instr.keys()
        ]

    max_episode_length = get_max_episode_length(args.tasks, args.variations)
    loaders = []

    for valset in args.valset:
        dataset = RLBenchDataset(
            root=valset,
            image_size=tuple(int(x) for x in args.image_size.split(",")),  # type: ignore
            taskvar=taskvar,
            instructions=instruction,
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

    return loaders


def get_model(args: Arguments) -> Tuple[optim.Optimizer, Hiveformer]:
    max_episode_length = get_max_episode_length(args.tasks, args.variations)
    if args.model == "original":
        _model = Hiveformer(
            depth=args.depth,
            dim_feedforward=args.dim_feedforward,
            hidden_dim=args.hidden_dim,
            instr_size=args.instr_size,
            mask_obs_prob=args.mask_obs_prob,
            max_episode_length=max_episode_length,
            num_layers=args.num_layers,
        )
    elif args.model == "baseline":
        if len(args.tasks) == 1:
            _model = Baseline(
                image_size=tuple(int(x) for x in args.image_size.split(",")),
                use_ground_truth_position_for_sampling=bool(args.use_ground_truth_position_for_sampling),
                position_loss=args.position_loss,
                embedding_dim=args.embedding_dim,
                num_ghost_point_cross_attn_layers=args.num_ghost_point_cross_attn_layers,
                num_query_cross_attn_layers=args.num_query_cross_attn_layers,
                rotation_pooling_gaussian_spread=args.rotation_pooling_gaussian_spread,
                gripper_loc_bounds=json.load(open("tasks/10_autolambda_tasks_location_bounds.json", "r"))[args.tasks[0]],
                num_ghost_points=args.num_ghost_points,
                coarse_to_fine_sampling=bool(args.coarse_to_fine_sampling),
                fine_sampling_cube_size=args.fine_sampling_cube_size,
            )
        else:
            raise NotImplementedError

    devices = [torch.device(d) for d in args.devices]
    model = _model.to(devices[0])
    if args.devices[0] != "cpu":
        assert all("cuda" in d for d in args.devices)
        model = torch.nn.DataParallel(model, device_ids=devices)

    optimizer_grouped_parameters = [
        {"params": [], "weight_decay": 0.0, "lr": args.lr},
        {"params": [], "weight_decay": 5e-4, "lr": args.lr},
    ]
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    for name, param in _model.named_parameters():
        if any(nd in name for nd in no_decay):
            optimizer_grouped_parameters[0]["params"].append(param)  # type: ignore
        else:
            optimizer_grouped_parameters[1]["params"].append(param)  # type: ignore
    optimizer: optim.Optimizer = optim.AdamW(optimizer_grouped_parameters)

    if args.checkpoint is not None:
        model_dict = torch.load(args.checkpoint, map_location="cpu")
        _model.load_state_dict(model_dict["weight"])
        optimizer.load_state_dict(model_dict["optimizer"])

    model_params = count_parameters(_model)
    print("Model parameters:", model_params)

    return optimizer, model


if __name__ == "__main__":
    args = Arguments().parse_args()

    # Force original HiveFormer parameters
    if args.model == "original":
        assert args.image_size == "128,128"
        args.position_loss = "mse"
        args.position_loss_coeff = 3.0
        args.rotation_loss_coeff = 4.0

    print()
    print("Arguments:")
    print(args)

    print()
    print("-" * 100)
    print()

    log_dir = get_log_dir(args)
    log_dir.mkdir(exist_ok=True, parents=True)
    args.save(str(log_dir / "hparams.json"))
    writer = SummaryWriter(log_dir=log_dir)

    print("Logging:", log_dir)
    print("Args devices:", args.devices)
    print("Available devices (CUDA_VISIBLE_DEVICES):", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("Device count", torch.cuda.device_count())

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    optimizer, model = get_model(args)

    print()
    print("-" * 100)
    print()

    loss_and_metrics = LossAndMetrics(
        position_prediction_only=bool(args.position_prediction_only),
        position_loss=args.position_loss,
        compute_loss_at_all_layers=bool(args.compute_loss_at_all_layers),
        ground_truth_gaussian_spread=args.ground_truth_gaussian_spread,
        label_smoothing=args.label_smoothing,
        position_loss_coeff=args.position_loss_coeff,
        rotation_loss_coeff=args.rotation_loss_coeff,
        gripper_loss_coeff=args.gripper_loss_coeff,
        rotation_pooling_gaussian_spread=args.rotation_pooling_gaussian_spread,
    )

    model_dict = {
        "weight": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    checkpointer = CheckpointCallback(
        "val-metrics-0/position_l2",
        log_dir,
        model_dict,
        val_freq=args.val_freq,
        minimizing=True,
        checkpoint_freq=args.checkpoint_freq,
    )
    model.train()

    val_loaders = get_val_loaders(args)

    if args.train_iters > 0:
        train_loader = get_train_loader(args)
        training(
            model,
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
            writer,
            loss_and_metrics,
            val_iters=-1,
        )

    # Last checkpoint
    checkpoint = log_dir / f"mtl_{args.seed}_{args.lr}.pth"
    torch.save(model_dict, checkpoint)
