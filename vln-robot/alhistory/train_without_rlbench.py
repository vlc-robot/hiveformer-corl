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

# If pl is imported after np, the gpu usage is divided by 5
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.plugins import DDPPlugin
import numpy as np
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
from utils_without_rlbench import (
    LossAndMetrics,
    load_instructions,
    load_rotation,
    load_episodes,
    get_max_episode_length,
)
from create_data import RLBench, Sample
from network import (
    PlainUNet,
    TransformerUNet,
)


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
    num_workers: int = 3 * gpus
    variations: Tuple[int, ...] = (0,)
    max_tries: int = 10
    max_episodes_per_taskvar: int = 100
    instructions: Optional[Path] = None
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

    workspace: Workspace = ((-0.325, 0.325, -0.455), (0.455, 0.0, 0.0))

    # model
    gripper_pose: GripperPose = "none"
    z_mode: ZMode = "embed"
    backbone: BackboneOp = "cat"
    cond: bool = False
    depth: int = 4
    dim_feedforward: int = 64
    embed_only: bool = False
    film: bool = False
    film_mlp: bool = False
    film_residual: bool = False
    hidden_dim: int = 64
    instr_size: int = 512
    mask_obs_prob: float = 0.0
    no_residual: bool = False
    num_layers: int = 1
    pcd_token: PointCloudToken = "none"
    taskvar_token: bool = False
    pos: PosMode = "mse"
    rot: RotMode = "mse"
    rot_type: RotType = "quat"
    stateless: bool = False
    tr_token: TransformerToken = "tnc"
    temp_len: int = 64
    attn_weights: bool = False


def get_git_patch():
    current_file = os.path.dirname(os.path.abspath(__file__))
    describe = subprocess.check_output(["git", "diff"], cwd=current_file)
    return describe.strip().decode()


def get_dec_len(args: Arguments) -> int:
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


class Module(pl.LightningModule):
    def __init__(self, args: Arguments):
        super().__init__()
        self._args = args
        self._rotation = load_rotation(self._args.rot, self._args.rot_type)
        self._position = Position(args.pos, args.workspace)
        self._loss_and_metrics = LossAndMetrics(self._rotation, self._position)

        max_eps_dict = load_episodes()["max_episode_length"]
        self.t_dict = nn.ParameterDict()
        self.z_dict = nn.ParameterDict()

        for task_str, max_eps in max_eps_dict.items():
            values = torch.rand(max_eps, self._args.temp_len, requires_grad=True)
            self.t_dict[task_str] = nn.Parameter(values)  # type: ignore
            self.z_dict[task_str] = nn.Parameter(  # type: ignore
                torch.zeros(max_eps, 3, requires_grad=True)
            )

        dec_len = get_dec_len(args)

        if self._args.arch == "mct":
            max_episode_length = get_max_episode_length(
                self._args.tasks, self._args.variations
            )
            self.model: PlainUNet = TransformerUNet(
                pos=self._position,
                gripper_pose=self._args.gripper_pose,
                attn_weights=self._args.attn_weights,
                backbone_op=self._args.backbone,
                cond=self._args.cond,
                depth=self._args.depth,
                dim_feedforward=self._args.dim_feedforward,
                dec_len=dec_len,
                embed_only=self._args.embed_only,
                film=self._args.film,
                film_mlp=self._args.film_mlp,
                film_residual=self._args.film_residual,
                hidden_dim=self._args.hidden_dim,
                instr_size=self._args.instr_size,
                instruction=self._args.instructions is not None,
                mask_obs_prob=self._args.mask_obs_prob,
                max_episode_length=max_episode_length,
                no_residual=self._args.no_residual,
                num_layers=args.num_layers,
                pcd_token=self._args.pcd_token,
                taskvar_token=self._args.taskvar_token,
                rot=self._rotation,
                stateless=self._args.stateless,
                temp_len=self._args.temp_len,
                tr_token=self._args.tr_token,
                z_mode=self._args.z_mode,
            )
        elif self._args.arch == "plain":
            self.model = PlainUNet(
                gripper_pose=self._args.gripper_pose,
                attn_weights=self._args.attn_weights,
                backbone_op=self._args.backbone,
                cond=self._args.cond,
                dec_len=dec_len,
                depth=self._args.depth,
                film=self._args.film,
                film_mlp=self._args.film_mlp,
                film_residual=self._args.film_residual,
                instruction=self._args.instructions is not None,
                instr_size=self._args.instr_size,
                rot=self._rotation,
                temp_len=self._args.temp_len,
                z_mode=self._args.z_mode,
            )
        else:
            raise RuntimeError(f"Unexpected arch {args.arch}")

    def configure_optimizers(self):
        if self._args.arch == "mct":
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

        elif self._args.arch == "plain":
            optimizer = optim.Adam(
                [
                    {"params": list(self.model.parameters()) + list(self.t_dict.values())},  # type: ignore
                    {"params": list(self.z_dict.values()), "weight_decay": 5e-4},
                ],
                lr=self._args.lr,
            )
            return optimizer

        else:
            raise RuntimeError(f"Unexpected arch {args.arch}")

    @property
    def log_dir(self):
        log_dir = (
            Path(self.logger._save_dir)
            / self.logger._name
            / self._args.tasks[0]
            / f"version_{self.logger._version}"
        )
        log_dir.mkdir(exist_ok=True, parents=True)
        return log_dir

    def on_fit_start(self):
        self._args.save(str(self.log_dir / "hparams.json"))
        print("Log dir", self.log_dir)

        log_dir = Path(self.logger.log_dir)
        log_dir.mkdir(exist_ok=True, parents=True)
        with open(log_dir / "hamt.patch", "w") as fid:
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
        taskvar = sample["taskvar"]
        frame_id = sample["frame_id"]
        tasks = sample["task"]
        t = torch.stack([self.t_dict[task][fid] for task, fid in zip(tasks, frame_id)])
        z = torch.stack([self.z_dict[task][fid] for task, fid in zip(tasks, frame_id)])

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

        train_losses = self._loss_and_metrics.compute_loss(pred, sample)
        train_losses["total"] = sum(list(train_losses.values()))  # type: ignore

        # Compute loss per task
        # for task in set(train_batch.task):
        #     mask = (
        #         torch.Tensor([t == task for t in train_batch.task])
        #         .bool()
        #         .to(loss.device)
        #     )
        #     self.log(
        #         f"train-loss/{task}", loss[mask].mean(), on_step=False, on_epoch=True
        #     )

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
        taskvar = sample["taskvar"]
        frame_id = sample["frame_id"]
        tasks = sample["task"]
        t = torch.stack([self.t_dict[task][fid] for task, fid in zip(tasks, frame_id)])
        z = torch.stack([self.z_dict[task][fid] for task, fid in zip(tasks, frame_id)])

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
        dataset = RLBench(
            root=self._args.dataset,
            taskvar=taskvar,
            instructions=instruction,
            num_iters=num_iters,
            gripper_pose=self._args.gripper_pose,
            taskvar_token=self._args.taskvar_token,
            max_episode_length=max_episode_length,
            max_episodes_per_taskvar=self._args.max_episodes_per_taskvar,
            cache_size=self._args.cache_size,
            jitter=self._args.jitter,
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

        dataset = RLBench(
            root=self._args.valset,
            taskvar=taskvar,
            instructions=instruction,
            gripper_pose=self._args.gripper_pose,
            taskvar_token=self._args.taskvar_token,
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
    if not args.eval_only:
        trainer.fit(model, dm)
    else:
        trainer.validate(model, dm)

    # Final test
    # FIXME convert it in PL format
    # ckpt_path = None if args.checkpoint_path == "" else args.checkpoint_path
    # trainer.test(datamodule=dm, ckpt_path=ckpt_path)
    # env = RLBenchEnv(
    #     data_path="",
    #     apply_rgb=True,
    #     apply_pc=True,
    #     apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
    #     headless=args.headless,
    #     gripper_pose=args.gripper_pose,
    # )
    # log_dir = get_log_dir(args)
    # log_dir.mkdir(exist_ok=True, parents=True)
    #
    # instruction = load_instructions(args.instructions)
    # actioner = Actioner(
    #     model={"model": model.model, "t": model.t_dict, "z": model.z_dict},  # type: ignore
    #     instructions=instruction,
    #     taskvar_token=args.taskvar_token,
    # )
    # max_eps_dict = load_episodes()["max_episode_length"]
    # for task_str in args.tasks:
    #     for variation_id in args.variations:
    #         success_rate = env.evaluate(
    #             task_str,
    #             actioner=actioner,
    #             max_episodes=max_eps_dict.get(task_str, 6),
    #             variation=variation_id,
    #             num_demos=500,
    #             demos=None,
    #             log_dir=log_dir,
    #             max_tries=args.max_tries,
    #         )
    #
    #         print("Testing Success Rate {}: {:.04f}".format(task_str, success_rate))
    #
    #         with FileLock(args.output.parent / f"{args.output.name}.lock"):
    #             with open(args.output, "a") as oid:
    #                 oid.write(
    #                     f"{task_str}-{variation_id}, na, seed={args.seed}, {success_rate}\n"
    #                 )
