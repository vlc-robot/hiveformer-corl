from typing import List, Tuple, Dict, Optional
import itertools
import subprocess
import os
from pathlib import Path
import random
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from filelock import FileLock
import tap

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.plugins import DDPPlugin
import numpy as np
from utils import (
    LossAndMetrics,
    load_instructions,
    RLBenchEnv,
    load_episodes,
    get_max_episode_length,
    Actioner,
)
from dataset import RLBenchDataset, Sample
from network import Hiveformer


class Arguments(tap.Tap):
    # PyTorch Lightning
    gpus: int = 1
    num_nodes: int = 1
    use_half: bool = False
    log_every_n_steps: int = 20
    accumulate_grad_batches: int = 1
    eval_only: bool = False

    dataset: Path
    seed: int = 2
    xp: Path = Path(__file__).parent / "xp"
    valset: Optional[Path] = None
    name: str = "multitask"
    arch: str = "mct"
    num_workers: int = 5
    variations: Tuple[int, ...] = (0,)
    max_tries: int = 10
    max_episodes_per_taskvar: int = 100
    instructions: Path
    cache_size: int = 100
    checkpoint: Optional[Path] = None
    checkpoint_period: int = 10

    tasks: Tuple[str, ...] = (
        "reach_target",
        "push_button",
        "pick_and_lift",
        "pick_up_cup",
        "put_knife_on_chopping_board",
        "take_money_out_safe",
        "put_money_in_safe",
        "take_umbrella_out_of_umbrella_stand",
        "stack_wine",
        "slide_block_to_target",
    )
    # Train
    batch_size: int = 32
    lr: float = 1e-5
    val_freq: int = 200
    val_batch_size: int = 100
    train_iters: int = 100_000

    # tests
    headless: bool = False
    output: Path = Path(__file__).parent / "records.txt"

    # model
    dim_feedforward: int = 64
    hidden_dim: int = 64
    instr_size: int = 512
    mask_obs_prob: float = 0.0
    num_layers: int = 1


def get_git_patch():
    current_file = os.path.dirname(os.path.abspath(__file__))
    describe = subprocess.check_output(["git", "diff"], cwd=current_file)
    return describe.strip().decode()


class Module(pl.LightningModule):
    def __init__(self, args: Arguments):
        super().__init__()
        self._args = args
        self._loss_and_metrics = LossAndMetrics()

        max_episode_length = get_max_episode_length(
            self._args.tasks, self._args.variations
        )
        self.model = Hiveformer(
            dim_feedforward=self._args.dim_feedforward,
            hidden_dim=self._args.hidden_dim,
            instr_size=self._args.instr_size,
            mask_obs_prob=self._args.mask_obs_prob,
            max_episode_length=max_episode_length,
            num_layers=args.num_layers,
        )

    def configure_optimizers(self):
        optimizer_grouped_parameters = [
            {
                "params": list(self.t_dict.values()),
                "weight_decay": 0.0,
                "lr": self._args.lr,
            },
            {
                "params": list(self.z_dict.values()),
                "weight_decay": 5e-4,
                "lr": self._args.lr,
            },
            {"params": [], "weight_decay": 5e-4, "lr": self._args.lr_transformer},
            {"params": [], "weight_decay": 0.0, "lr": self._args.lr_rotation},
        ]
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        transformer_modules = ["encoder"]
        zinstr = ["z_pos_instr", "z_proj_instr"]
        rotation_modules = ["quat_decoder"]
        for name, param in self.model.named_parameters():
            if any(nd in name for nd in no_decay):
                optimizer_grouped_parameters[0]["params"].append(param)  # type: ignore
            elif any(nd in name for nd in zinstr):
                optimizer_grouped_parameters[1]["params"].append(param)  # type: ignore
            elif any(nd in name for nd in transformer_modules):
                optimizer_grouped_parameters[2]["params"].append(param)  # type: ignore
            elif any(nd in name for nd in rotation_modules):
                optimizer_grouped_parameters[3]["params"].append(param)  # type: ignore
            else:
                optimizer_grouped_parameters[1]["params"].append(param)  # type: ignore
            optimizer: optim.Optimizer = optim.AdamW(optimizer_grouped_parameters)
            return optimizer

    @property
    def log_dir(self):
        log_dir = (
            Path(self.logger._save_dir)
            / self.logger._name
            / f"version_{self.logger._version}"
        )
        log_dir.mkdir(exist_ok=True, parents=True)
        return log_dir

    def on_fit_start(self):
        self._args.save(str(self.log_dir / "hparams.json"))

        log_dir = Path(self.logger.log_dir)
        log_dir.mkdir(exist_ok=True, parents=True)
        with open(log_dir / "git.patch", "w") as fid:
            fid.write(get_git_patch())

    def forward(self, batch: Sample):  # type: ignore
        return self.model(batch)

    def training_step(self, sample: Sample, *_):  # type: ignore
        rgbs = sample["rgbs"]
        pcds = sample["pcds"]
        gripper = sample["gripper"]
        outputs = sample["action"]
        padding_mask = sample["padding_mask"]
        instr = sample["instr"]
        frame_id = sample["frame_id"]
        tasks = sample["task"]

        pred = self.model(
            rgbs,
            pcds,
            padding_mask,
            instr,
            gripper,
        )

        train_losses = self._loss_and_metrics.compute_loss(pred, sample)
        train_losses["total"] = sum(list(train_losses.values()))  # type: ignore

        self.log_dict(
            {f"train/{key}/loss": v for key, v in train_losses.items()},
            batch_size=len(tasks),
        )

        train_metrics = self._loss_and_metrics.compute_metrics(pred, sample)
        self.log_dict(
            {f"train/{key}/metrics": v for key, v in train_metrics.items()},
            batch_size=len(tasks),
        )

        return {"loss": train_losses["total"]}

    @torch.no_grad()
    def validation_step(self, sample, *_):
        rgbs = sample["rgbs"]
        pcds = sample["pcds"]
        gripper = sample["gripper"]
        outputs = sample["action"]
        padding_mask = sample["padding_mask"]
        instr = sample["instr"]
        frame_id = sample["frame_id"]
        tasks = sample["task"]

        pred = self.model(
            rgbs,
            pcds,
            padding_mask,
            t,
            z,
            instr,
            gripper,
            taskvar,
        )

        val_losses = self._loss_and_metrics.compute_loss(pred, sample)
        val_losses["total"] = sum(list(val_losses.values()))  # type: ignore

        self.log_dict(
            {f"val/{key}/loss": v for key, v in val_losses.items()}, batch_size=len(tasks)
        )

        val_metrics = self._loss_and_metrics.compute_metrics(pred, sample)
        self.log_dict(
            {f"val/{key}/metrics": v for key, v in val_metrics.items()},
            batch_size=len(tasks),
        )


def collate_fn(batch: List[Dict]):
    keys = batch[0].keys()
    return {
        key: default_collate([item[key] for item in batch])
        if batch[0][key] is not None
        else None
        for key in keys
    }


class DataModule(pl.LightningDataModule):
    def __init__(self, args: Arguments):
        super().__init__()
        self._args = args

    def train_dataloader(self) -> DataLoader:
        taskvar = list(itertools.product(self._args.tasks, self._args.variations))
        instruction = load_instructions(self._args.instructions)
        max_episode_length = get_max_episode_length(
            self._args.tasks, self._args.variations
        )
        num_iters = (
            self._args.train_iters
            * self._args.batch_size
            * self._args.num_nodes
            * self._args.gpus
        )
        dataset = RLBenchDataset(
            root=self._args.dataset,
            taskvar=taskvar,
            instructions=instruction,
            num_iters=num_iters,
            max_episode_length=max_episode_length,
            max_episodes_per_taskvar=self._args.max_episodes_per_taskvar,
            cache_size=self._args.cache_size,
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=self._args.batch_size,
            shuffle=True,
            num_workers=self._args.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self._args.num_workers > 0,
        )
        # return AutoReloadLoader(loader)
        return loader

    def val_dataloader(self) -> Optional[DataLoader]:
        if self._args.valset is None:
            return None

        instruction = load_instructions(self._args.instructions)
        taskvar = list(itertools.product(self._args.tasks, self._args.variations))
        max_episode_length = get_max_episode_length(
            self._args.tasks, self._args.variations
        )

        dataset = RLBenchDataset(
            root=self._args.valset,
            instructions=instruction,
            max_episode_length=max_episode_length,
            max_episodes_per_taskvar=self._args.max_episodes_per_taskvar,
            cache_size=0,
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=self._args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            persistent_workers=False,
        )
        return loader


class AutoReloadLoader:
    def __init__(self, loader: DataLoader):
        self._loader = loader
        self._iter = iter(self._loader)

    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self._loader)
            return next(self._iter)


def get_log_dir(args: Arguments) -> Path:
    log_dir = args.xp / args.name
    version = int(os.environ.get("SLURM_JOBID", 0))
    while (log_dir / f"version{version}").is_dir():
        version += 1
    return log_dir / f"version{version}"


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.random.manual_seed(0)

    args = Arguments().parse_args()
    print(args)

    dm = DataModule(args)

    if args.checkpoint is None:
        model = Module(args)
    else:
        model = Module.load_from_checkpoint(args.checkpoint, args=args)

    # callbacks
    checkpoint_period = args.val_freq
    if args.checkpoint_period > 0:
        save_top_k = -1
        checkpoint_period *= args.checkpoint_period
    else:
        save_top_k = 5
    checkpoint_callback = ModelCheckpoint(
        monitor="train/position/metrics",
        save_top_k=save_top_k,
        every_n_train_steps=checkpoint_period,
        mode="max",
    )
    device_stats = DeviceStatsMonitor()
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [lr_monitor, checkpoint_callback]

    default_root_dir = f"{args.xp}/{args.name}" if args.name != "" else None

    trainer = pl.Trainer(
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        precision=16 if args.use_half else 32,
        callbacks=callbacks,
        gradient_clip_val=20.0,
        log_every_n_steps=args.log_every_n_steps,
        limit_train_batches=0 if args.eval_only else 1.0,
        limit_val_batches=5,
        max_steps=args.train_iters,
        val_check_interval=int(args.val_freq),
        default_root_dir=default_root_dir,
        strategy=None,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    if args.eval_only:
        trainer.fit(model, dm)
    else:
        trainer.validate(model, dm)

    # Evaluation
    env = RLBenchEnv(
        data_path="",
        apply_rgb=True,
        apply_pc=True,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
        headless=args.headless,
    )
    log_dir = get_log_dir(args)
    log_dir.mkdir(exist_ok=True, parents=True)

    instruction = load_instructions(args.instructions)
    actioner = Actioner(model=model.model, instructions=instruction)
    max_eps_dict = load_episodes()["max_episode_length"]
    for task_str in args.tasks:
        for variation in args.variations:
            success_rate = env.evaluate(
                task_str,
                actioner=actioner,
                max_episodes=max_eps_dict.get(task_str, 6),
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
                        f"{task_str}-{variation}, na, seed={args.seed}, {success_rate}\n"
                    )
