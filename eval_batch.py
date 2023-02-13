import random
import torch
from torch.utils.tensorboard import SummaryWriter  # type: ignore
import numpy as np
from filelock import FileLock
from utils import (
    Actioner,
    load_instructions,
    load_episodes,
    RLBenchEnv,
)

from train_batch import (
    Arguments,
    get_log_dir,
    get_model,
)

import torch.multiprocessing as mp


def eval_online_one_task(task_str, variation_id, model, t_dict, z_dict, log_dir, args):
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

    with FileLock(log_dir / "online_eval_scores.txt.lock"):
        with open(log_dir / "online_eval_scores.txt", "a") as oid:
            oid.write(
                f"{task_str}-{variation_id}, {args.checkpoint}, seed={args.seed}, {success_rate}\n"
            )
    print(log_dir)


def eval_online():
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

    # evaluation
    model.eval()
    model.share_memory()
    for k, v in t_dict.items():
        v.share_memory_()
        t_dict[k] = v.detach()
        # v.requires_grad = False
    for k, v in z_dict.items():
        v.share_memory_()
        z_dict[k] = v.detach()
        # v.requires_grad = False

    procs = []
    for task_str in args.tasks:
        for variation_id in args.variations:
            print("add", task_str, variation_id, "proc")
            proc = mp.Process(
                target=eval_online_one_task,
                args=(task_str, variation_id, model, t_dict, z_dict, log_dir, args),
            )
            procs.append(proc)
            proc.start()

    for proc in procs:
        proc.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")

    eval_online()
