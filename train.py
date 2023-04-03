import random
from typing import List, Tuple, Dict, Optional, Any
import os
from collections import defaultdict
from pathlib import Path
import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import numpy as np
from tqdm import tqdm, trange
import tap
import wandb

from utils.utils_without_rlbench import (
    LossAndMetrics,
    load_instructions,
    count_parameters,
    get_max_episode_length,
    get_gripper_loc_bounds,
    TASK_TO_ID
)
from dataset import RLBenchDataset, RLBenchAnalogicalDataset
from model.released_hiveformer.network import Hiveformer
from model.non_analogical_baseline.baseline import Baseline
from model.analogical_network.analogical_network import AnalogicalNetwork


class Arguments(tap.Tap):
    cameras: Tuple[str, ...] = ("wrist", "left_shoulder", "right_shoulder")
    image_size: str = "256,256"
    max_tries: int = 10
    max_episodes_per_task: int = 100
    instructions: Optional[Path] = "instructions.pkl"
    cache_size: int = 100
    seed: int = 0
    tasks: Tuple[str, ...]
    variations: Tuple[int, ...] = (0,)
    checkpoint: Optional[Path] = None
    accumulate_grad_batches: int = 1
    val_freq: int = 500
    checkpoint_freq: int = 10

    # Training and validation datasets
    dataset: List[Path]
    valset: Optional[Tuple[Path, ...]] = None

    # Logging to base_log_dir/exp_log_dir/run_log_dir
    logger: Optional[str] = "tensorboard"  # One of "wandb", "tensorboard", None
    base_log_dir: Path = Path(__file__).parent / "train_logs"
    exp_log_dir: str = "exp"
    run_log_dir: str = "run"

    # Main training parameters
    devices: List[str] = ["cuda:0"]  # ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    num_workers: int = 1
    batch_size: int = 16
    batch_size_val: int = 4
    lr: float = 1e-4
    train_iters: int = 200_000
    max_episode_length: int = 5  # -1 for no limit

    # Toggle to switch between original HiveFormer and our models
    model: str = "baseline"  # one of "original", "baseline", "analogical"

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

    # Data augmentations
    image_rescale: str = "0.75,1.25"  # (min, max), "1.0,1.0" for no rescaling
    point_cloud_rotate_yaw_range: float = 0.0  # in degrees, 0.0 for no rotation

    visualize_rgb_attn: int = 0  # deactivate by default during training as this has memory overhead
    gripper_loc_bounds_file: str = "tasks/74_hiveformer_tasks_location_bounds.json"
    single_task_gripper_loc_bounds: int = 0
    gripper_bounds_buffer: float = 0.04

    # Loss
    position_prediction_only: int = 0
    position_loss: str = "ce"  # one of "ce" (our model), "mse" (original HiveFormer)
    ground_truth_gaussian_spread: float = 0.01
    compute_loss_at_all_layers: int = 0
    position_loss_coeff: float = 1.0
    position_offset_loss_coeff: float = 10000.0
    rotation_loss_coeff: float = 10.0
    symmetric_rotation_loss: int = 0
    gripper_loss_coeff: float = 1.0
    label_smoothing: float = 0.0
    regress_position_offset: int = 0

    # Ghost points
    num_sampling_level: int = 3
    fine_sampling_ball_diameter: float = 0.16
    weight_tying: int = 1
    gp_emb_tying: int = 1
    num_ghost_points: int = 1000
    num_ghost_points_val: int = 10000
    use_ground_truth_position_for_sampling_train: int = 1  # considerably speeds up training
    use_ground_truth_position_for_sampling_val: int = 0    # for debugging

    # Model
    backbone: str = "clip"  # one of "resnet", "clip"
    embedding_dim: int = 60
    num_ghost_point_cross_attn_layers: int = 2
    num_query_cross_attn_layers: int = 2
    rotation_parametrization: str = "quat_from_query"  # one of "quat_from_top_ghost", "quat_from_query" for now
    use_instruction: int = 0
    task_specific_biases: int = 0

    # Positional features
    positional_features: Optional[str] = "none"  # one of "xyz_concat", "z_concat", "xyz_add", "z_add", "none"

    # ---------------------------------------------------------------
    # Our analogical network additional parameters
    # ---------------------------------------------------------------

    support_set: str = "others"  # one of "self" (for debugging), "others"
    support_set_size: int = 1
    global_correspondence: int = 0
    num_matching_cross_attn_layers: int = 2


def training(
    model: nn.Module,
    optimizer,
    train_loader,
    val_loaders,
    checkpointer,
    loss_and_metrics,
    args: Arguments,
    writer: Optional[SummaryWriter] = None,
    use_ground_truth_position_for_sampling_train=True,
    use_ground_truth_position_for_sampling_val=False,
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
                    sample["task_id"],
                    # Provide ground-truth action to bias ghost point sampling at training time
                    gt_action=sample["action"] if use_ground_truth_position_for_sampling_train else None,
                )
            elif model_type == AnalogicalNetwork:
                pred = model(
                    sample["rgbs"],
                    sample["pcds"],
                    sample["padding_mask"],
                    sample["instr"],
                    sample["gripper"],
                    sample["task_id"],
                    gt_action_for_support=sample["action"],
                    # Provide ground-truth action to bias ghost point sampling at training time
                    gt_action_for_sampling=sample["action"] if use_ground_truth_position_for_sampling_train else None,
                )

            train_losses = loss_and_metrics.compute_loss(pred, sample)
            train_losses["total"] = sum(list(train_losses.values()))  # type: ignore
            train_losses["total"].backward()  # type: ignore

            metrics = loss_and_metrics.compute_metrics(pred, sample)

            for n, l in train_losses.items():
                aggregated_losses[n].append(l)
            for n, l in metrics.items():
                aggregated_metrics[n].append(l)

            if step_id % args.accumulate_grad_batches == args.accumulate_grad_batches - 1:
                optimizer.step()

            if args.logger == "wandb":
                wandb.log(
                    {
                        "lr": args.lr,
                        **{f"train-loss/{n}": torch.mean(torch.stack(l)) for n, l in aggregated_losses.items()},
                        **{f"train-metrics/{n}": torch.mean(torch.stack(l)) for n, l in aggregated_metrics.items()},
                    },
                    step=step_id,
                )

            if (step_id + 1) % args.val_freq == 0:
                if args.logger == "tensorboard":
                    writer.add_scalar(f"lr/", args.lr, step_id)
                    for n, l in aggregated_losses.items():
                        writer.add_scalar(f"train-loss/{n}", torch.mean(torch.stack(l)), step_id)
                    for n, l in aggregated_metrics.items():
                        writer.add_scalar(f"train-metrics/{n}", torch.mean(torch.stack(l)), step_id)

                aggregated_losses = defaultdict(list)
                aggregated_metrics = defaultdict(list)

                if val_loaders is not None:
                    val_metrics = validation_step(
                        step_id,
                        val_loaders,
                        model,
                        loss_and_metrics,
                        args,
                        writer,
                        use_ground_truth_position_for_sampling_val=use_ground_truth_position_for_sampling_val
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
    loss_and_metrics,
    args: Arguments,
    writer: Optional[SummaryWriter] = None,
    use_ground_truth_position_for_sampling_val=False,
    val_iters: int = 10,
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
                    sample["task_id"],
                    # DO NOT provide ground-truth action to sample ghost points at validation time
                    gt_action=sample["action"] if use_ground_truth_position_for_sampling_val else None
                )
            elif model_type == AnalogicalNetwork:
                pred = model(
                    sample["rgbs"],
                    sample["pcds"],
                    sample["padding_mask"],
                    sample["instr"],
                    sample["gripper"],
                    sample["task_id"],
                    gt_action_for_support=sample["action"],
                    # DO NOT provide ground-truth action to sample ghost points at validation time
                    gt_action_for_sampling=sample["action"] if use_ground_truth_position_for_sampling_val else None
                )

            losses: Dict[str, torch.Tensor] = loss_and_metrics.compute_loss(pred, sample)
            losses["total"] = torch.stack(list(losses.values())).sum()

            metrics = loss_and_metrics.compute_metrics(pred, sample)

            for n, l in losses.items():
                key = f"val-loss-{val_id}/{n}"
                if args.logger == "tensorboard":
                    writer.add_scalar(key, l, step_id + i)
                elif args.logger == "wandb":
                    wandb.log({key: l}, step=step_id + i)
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

            if args.logger == "tensorboard":
                writer.add_scalar(f"lr/", args.lr, step_id + i)
            elif args.logger == "wandb":
                wandb.log({"lr": args.lr}, step=step_id + i)

            for n, l in metrics.items():
                key = f"val-metrics-{val_id}/{n}"
                if args.logger == "tensorboard":
                    writer.add_scalar(key, l, step_id + i)
                elif args.logger == "wandb":
                    wandb.log({key: l}, step=step_id + i)
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

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


def get_train_loader(args: Arguments, gripper_loc_bounds) -> DataLoader:
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
    if args.max_episode_length >= 0:
        max_episode_length = min(args.max_episode_length, max_episode_length)

    if args.model in ["original", "baseline"]:
        dataset = RLBenchDataset(
            root=args.dataset,
            image_size=tuple(int(x) for x in args.image_size.split(",")),  # type: ignore
            taskvar=taskvar,
            instructions=instruction,
            max_episode_length=max_episode_length,
            max_episodes_per_task=args.max_episodes_per_task,
            cache_size=args.cache_size,
            num_iters=args.train_iters,
            cameras=args.cameras,  # type: ignore
            image_rescale=tuple(float(x) for x in args.image_rescale.split(",")),
            point_cloud_rotate_yaw_range=args.point_cloud_rotate_yaw_range,
            gripper_loc_bounds=gripper_loc_bounds,
        )
    elif args.model == "analogical":
        # During train, the training split is both the main dataset and the support dataset
        dataset = RLBenchAnalogicalDataset(
            main_root=args.dataset,
            support_root=args.dataset,
            image_size=tuple(int(x) for x in args.image_size.split(",")),  # type: ignore
            taskvar=taskvar,
            instructions=instruction,
            max_episode_length=max_episode_length,
            max_episodes_per_task=args.max_episodes_per_task,
            cache_size=args.cache_size,
            num_iters=args.train_iters,
            cameras=args.cameras,  # type: ignore
            image_rescale=tuple(float(x) for x in args.image_rescale.split(",")),
            point_cloud_rotate_yaw_range=args.point_cloud_rotate_yaw_range,
            gripper_loc_bounds=gripper_loc_bounds,
            support_set_size=args.support_set_size,
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


def get_val_loaders(args: Arguments, gripper_loc_bounds) -> Optional[List[DataLoader]]:
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
    if args.max_episode_length >= 0:
        max_episode_length = min(args.max_episode_length, max_episode_length)

    loaders = []

    for valset in args.valset:
        if args.model in ["original", "baseline"]:
            dataset = RLBenchDataset(
                root=valset,
                image_size=tuple(int(x) for x in args.image_size.split(",")),  # type: ignore
                taskvar=taskvar,
                instructions=instruction,
                max_episode_length=max_episode_length,
                max_episodes_per_task=args.max_episodes_per_task,
                cache_size=args.cache_size,
                cameras=args.cameras,  # type: ignore
                training=False,
                image_rescale=tuple(float(x) for x in args.image_rescale.split(",")),
                point_cloud_rotate_yaw_range=args.point_cloud_rotate_yaw_range,
                gripper_loc_bounds=gripper_loc_bounds,
            )
        elif args.model == "analogical":
            # During evaluation, the main dataset is the validation split and the support dataset
            # is the training split
            dataset = RLBenchAnalogicalDataset(
                main_root=args.valset,
                support_root=args.dataset,
                image_size=tuple(int(x) for x in args.image_size.split(",")),  # type: ignore
                taskvar=taskvar,
                instructions=instruction,
                max_episode_length=max_episode_length,
                max_episodes_per_task=args.max_episodes_per_task,
                cache_size=args.cache_size,
                cameras=args.cameras,  # type: ignore
                training=False,
                image_rescale=tuple(float(x) for x in args.image_rescale.split(",")),
                point_cloud_rotate_yaw_range=args.point_cloud_rotate_yaw_range,
                gripper_loc_bounds=gripper_loc_bounds,
                support_set_size=args.support_set_size,
            )
        loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size_val,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
        )
        loaders.append(loader)

    return loaders


def get_model(args: Arguments, gripper_loc_bounds) -> Tuple[optim.Optimizer, Hiveformer]:
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
        _model = Baseline(
            backbone=args.backbone,
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=args.embedding_dim,
            num_ghost_point_cross_attn_layers=args.num_ghost_point_cross_attn_layers,
            num_query_cross_attn_layers=args.num_query_cross_attn_layers,
            rotation_parametrization=args.rotation_parametrization,
            gripper_loc_bounds=gripper_loc_bounds,
            num_ghost_points=args.num_ghost_points,
            num_ghost_points_val=args.num_ghost_points_val,
            weight_tying=bool(args.weight_tying),
            gp_emb_tying=bool(args.gp_emb_tying),
            num_sampling_level=args.num_sampling_level,
            fine_sampling_ball_diameter=args.fine_sampling_ball_diameter,
            regress_position_offset=bool(args.regress_position_offset),
            visualize_rgb_attn=bool(args.visualize_rgb_attn),
            use_instruction=bool(args.use_instruction),
            positional_features=args.positional_features,
            task_specific_biases=bool(args.task_specific_biases),
            task_ids=[TASK_TO_ID[task] for task in args.tasks],
        )
    elif args.model == "analogical":
        _model = AnalogicalNetwork(
            backbone=args.backbone,
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=args.embedding_dim,
            num_ghost_point_cross_attn_layers=args.num_ghost_point_cross_attn_layers,
            gripper_loc_bounds=gripper_loc_bounds,
            num_ghost_points=args.num_ghost_points,
            num_ghost_points_val=args.num_ghost_points_val,
            weight_tying=bool(args.weight_tying),
            gp_emb_tying=bool(args.gp_emb_tying),
            num_sampling_level=args.num_sampling_level,
            fine_sampling_ball_diameter=args.fine_sampling_ball_diameter,
            regress_position_offset=bool(args.regress_position_offset),
            use_instruction=bool(args.use_instruction),
            positional_features=args.positional_features,
            task_specific_biases=bool(args.task_specific_biases),
            task_ids=[TASK_TO_ID[task] for task in args.tasks],
            support_set=args.support_set,
            global_correspondence=args.global_correspondence,
            num_matching_cross_attn_layers=args.num_matching_cross_attn_layers,
        )

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
        model_dict_weight = {}
        for key in model_dict["weight"]:
            _key = key[7:]
            # if 'prediction_head.feature_pyramid.inner_blocks' in _key:
            #     _key = _key[:46] + _key[48:]
            # if 'prediction_head.feature_pyramid.layer_blocks' in _key:
            #     _key = _key[:46] + _key[48:]
            model_dict_weight[_key] = model_dict["weight"][key]
        _model.load_state_dict(model_dict_weight)
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
        args.batch_size = 32
        args.train_iters = 100_000

    assert args.batch_size % len(args.devices) == 0

    print()
    print("Arguments:")
    print(args)

    print()
    print("-" * 100)
    print()

    log_dir = get_log_dir(args)
    log_dir.mkdir(exist_ok=True, parents=True)
    args.save(str(log_dir / "hparams.json"))

    if args.logger == "tensorboard":
        writer = SummaryWriter(log_dir=log_dir)
    elif args.logger == "wandb":
        wandb.init(project="analogical_manipulation")
        wandb.run.name = str(log_dir).split("/")[-1]
        wandb.config.update(args.__dict__)
        writer = None
    else:
        writer = None

    print("Logging:", log_dir)
    print("Args devices:", args.devices)
    print("Available devices (CUDA_VISIBLE_DEVICES):", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("Device count", torch.cuda.device_count())

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Gripper workspace is the union of workspaces for all tasks
    if args.single_task_gripper_loc_bounds and len(args.tasks) == 1:
        task = args.tasks[0]
    else:
        task = None
    gripper_loc_bounds = get_gripper_loc_bounds(
        args.gripper_loc_bounds_file, task=task, buffer=args.gripper_bounds_buffer)

    optimizer, model = get_model(args, gripper_loc_bounds)

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
        position_offset_loss_coeff=args.position_offset_loss_coeff,
        rotation_loss_coeff=args.rotation_loss_coeff,
        gripper_loss_coeff=args.gripper_loss_coeff,
        regress_position_offset=bool(args.regress_position_offset),
        symmetric_rotation_loss=bool(args.symmetric_rotation_loss)
    )

    model_dict = {
        "weight": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    checkpointer = CheckpointCallback(
        "val-metrics-0/pos_l2_final",
        log_dir,
        model_dict,
        val_freq=args.val_freq,
        minimizing=True,
        checkpoint_freq=args.checkpoint_freq,
    )
    model.train()

    val_loaders = get_val_loaders(args, gripper_loc_bounds)

    if args.train_iters > 0:
        train_loader = get_train_loader(args, gripper_loc_bounds)
        training(
            model,
            optimizer,
            train_loader,
            val_loaders,
            checkpointer,
            loss_and_metrics,
            args,
            writer,
            use_ground_truth_position_for_sampling_train=bool(args.use_ground_truth_position_for_sampling_train),
            use_ground_truth_position_for_sampling_val=bool(args.use_ground_truth_position_for_sampling_val),
        )

    if val_loaders is not None:
        val_metrics = validation_step(
            args.train_iters,
            val_loaders,
            model,
            loss_and_metrics,
            args,
            writer,
            use_ground_truth_position_for_sampling_val=bool(args.use_ground_truth_position_for_sampling_val),
            val_iters=-1,
        )

    # Last checkpoint
    checkpoint = log_dir / f"mtl_{args.seed}_{args.lr}.pth"
    torch.save(model_dict, checkpoint)
