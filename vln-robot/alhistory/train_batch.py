import random
from typing import List, Tuple, Dict, Optional, Any
import itertools
from pathlib import Path
from structures import Position
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import numpy as np
from tqdm import trange
from filelock import FileLock
from network import (
    PlainUNet,
    TransformerUNet,
)
from utils import (
    Actioner,
    count_parameters,
    get_max_episode_length,
    LossAndMetrics,
    load_instructions,
    load_rotation,
    load_episodes,
    Model,
    RLBenchEnv,
)
from create_data import RLBench
from multi_task_baselines import Arguments


def get_log_dir(args: Arguments) -> Path:
    log_dir = args.xp / args.name
    version = 0
    while (log_dir / f"version{version}").is_dir():
        version += 1
    return log_dir / f"version{version}"


def get_train_loader(args: Arguments) -> List[DataLoader]:
    taskvar = list(itertools.product(args.tasks, args.variations))
    instruction = load_instructions(args.instructions)
    if instruction is not None:
        instruction = {
            key: {v: values[v] for v in args.variations}
            for key, values in instruction.items()
            if key in args.tasks
        }
    max_episode_length = get_max_episode_length(args.tasks, args.variations)
    train_sets = []
    for task, variation in taskvar:
        train_set = RLBench(
            root=args.dataset,
            taskvar=[(task, variation)],
            instructions=instruction,
            gripper_pose=args.gripper_pose,
            taskvar_token=args.taskvar_token,
            max_episode_length=max_episode_length,
            max_episodes_per_taskvar=args.max_episodes_per_taskvar,
            cache_size=args.cache_size,
            num_iters=args.train_iters,
        )
        train_sets.append(train_set)

    train_loaders = [
        DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            persistent_workers=args.num_workers > 0,
        )
        for train_set in train_sets
    ]

    return train_loaders


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


def get_model(args: Arguments, device) -> Tuple[optim.Optimizer, Model, LossAndMetrics]:
    rotation = load_rotation(args.rot, args.rot_type)
    position = Position(args.pos, args.workspace)

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
            gripper_pose=args.gripper_pose,
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
            hidden_dim=args.hidden_dim,
            instr_size=args.instr_size,
            instruction=args.instructions is not None,
            mask_obs_prob=args.mask_obs_prob,
            max_episode_length=max_episode_length,
            no_residual=args.no_residual,
            num_layers=args.num_layers,
            pcd_token=args.pcd_token,
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
        # optimizer.load_state_dict(model_dict["optimizer"])
        t_dict = {k: t.to(device) for k, t in model_dict["t"].items()}
        z_dict = {k: z.to(device) for k, z in model_dict["z"].items()}

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


def collate_fn(batch: List[Dict]):
    keys = batch[0].keys()
    return {
        key: default_collate([item[key] for item in batch])
        if batch[0][key] is not None
        else None
        for key in keys
    }


def get_val_loader(args: Arguments) -> Optional[DataLoader]:
    if args.valset is None:
        return None

    instruction = load_instructions(args.instructions)
    if instruction is not None:
        instruction = {
            key: {v: values[v] for v in args.variations}
            for key, values in instruction.items()
            if key in args.tasks
        }
    taskvar = list(itertools.product(args.tasks, args.variations))
    max_episode_length = get_max_episode_length(args.tasks, args.variations)

    dataset = RLBench(
            root=args.valset, # type: ignore
        taskvar=taskvar,
        instructions=instruction,
        gripper_pose=args.gripper_pose,
        taskvar_token=args.taskvar_token,
        max_episode_length=max_episode_length,
        max_episodes_per_taskvar=args.max_episodes_per_taskvar,
        cache_size=0,
        jitter=args.jitter,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    return loader


@torch.no_grad()
def validation_step(
    step_id: int, val_loader: DataLoader, model, t_dict, z_dict, writer, loss_and_metrics
):
    device = t_dict[list(t_dict.keys())[0]].device
    values = {}
    model.eval()

    for i, sample in enumerate(val_loader):
        if i == 5:
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
            key = f"val-loss/{n}"
            writer.add_scalar(key, l, step_id + i)
            if key not in values:
                values[key] = torch.Tensor([]).to(device)
            values[key] = torch.cat([values[key], l.unsqueeze(0)])

        writer.add_scalar(f"lr/", args.lr, step_id + i)

        metrics = loss_and_metrics.compute_metrics(pred, sample)
        for n, l in metrics.items():
            key = f"val-metrics/{n}"
            writer.add_scalar(key, l, step_id + i)
            if key not in metrics:
                values[key] = torch.Tensor([]).to(device)
            values[key] = torch.cat([values[key], l.unsqueeze(0)])

    print(f'Validation Loss: {values["val-loss/total"].mean():.05f}')
    return values


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

        dest = self._log_dir / f"model.step={self._step}.pth"
        torch.save(self._state_dict, dest)


if __name__ == "__main__":
    args = Arguments().parse_args()
    log_dir = get_log_dir(args)
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Logging:", log_dir)
    args.save(str(log_dir / "hparams.json"))
    writer = SummaryWriter(log_dir=log_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)
    optimizer, meta_model, loss_and_metrics = get_model(args, device)
    model = meta_model["model"]
    t_dict = meta_model["t"]
    z_dict = meta_model["z"]

    train_loaders = get_train_loader(args)
    val_loader = get_val_loader(args)
    # training episode
    model.train()
    train_datasets = [iter(train_loader) for train_loader in train_loaders]
    model_dict = {
        "weight": model.state_dict(),
        "t": t_dict,
        "z": z_dict,
        "optimizer": optimizer.state_dict(),
    }
    checkpointer = CheckpointCallback(
        "train/metrics/position",
        log_dir,
        model_dict,
        minimizing=False,
        checkpoint_period=args.checkpoint_period,
    )

    with trange(args.train_iters) as tbar:
        for i in tbar:
            for task_id, task_str in enumerate(args.tasks):
                try:
                    sample = train_datasets[task_id].next()
                except:
                    print(task_id, "is done")
                    train_datasets[task_id] = iter(train_loaders[task_id])
                    sample = train_datasets[task_id].next()

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

                if i % args.accumulate_grad_batches == 0:
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
                    writer.add_scalar(f"train-loss/{task_str}/{n}", l, i)

                metrics = loss_and_metrics.compute_metrics(pred, sample)
                for n, l in metrics.items():
                    writer.add_scalar(f"train-metrics/{task_str}/{n}", l, i)

                train_losses["total"].backward()  # type: ignore

                if i % args.accumulate_grad_batches == args.accumulate_grad_batches - 1:
                    optimizer.step()

                if (i + 1) % args.val_freq == 0:
                    if val_loader is not None:
                        val_metrics = validation_step(
                            i,
                            val_loader,
                            model,
                            t_dict,
                            z_dict,
                            writer,
                            loss_and_metrics,
                        )
                        checkpointer(val_metrics)

                tbar.set_postfix(l=float(train_losses["total"]))

    # last checkpoint
    model_dict = {"weight": model.state_dict(), "t": t_dict, "z": z_dict}
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
        for variation_id in args.variations:
            success_rate = env.evaluate(
                task_str,
                actioner=actioner,
                max_episodes=max_eps_dict.get(task_str, 6),
                variation=variation_id,
                num_demos=500,
                demos=None,
                log_dir=log_dir,
                max_tries=args.max_tries,
            )

            print("Testing Success Rate {}: {:.04f}".format(task_str, success_rate))

            with FileLock(args.output.parent / f"{args.output.name}.lock"):
                with open(args.output, "a") as oid:
                    oid.write(
                        f"{task_str}-{variation_id}, {checkpoint}, seed={args.seed}, {success_rate}\n"
                    )
