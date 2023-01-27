import tap
from typing import List, Tuple, Optional
from pathlib import Path
import torch
import pprint
import json

from utils import (
    load_instructions,
    get_max_episode_length,
)
from dataset import RLBenchDataset


class Arguments(tap.Tap):
    cameras: Tuple[str, ...] = ("wrist", "left_shoulder", "right_shoulder")
    dataset: List[Path]
    max_episodes_per_taskvar: int = 100
    instructions: Optional[Path] = "instructions.pkl"
    cache_size: int = 100

    tasks: Tuple[str, ...] = (
        "pick_and_lift",
        "pick_up_cup",
        "push_button",
        "put_knife_on_chopping_board",
        "put_money_in_safe",
        "reach_target",
        "slide_block_to_target",
        "stack_wine",
        "take_money_out_safe",
        "take_umbrella_out_of_umbrella_stand",
    )
    variations: Tuple[int, ...] = (0,)


if __name__ == "__main__":
    args = Arguments().parse_args()

    instruction = load_instructions(
        args.instructions, tasks=args.tasks, variations=args.variations
    )
    taskvar = [
        (task, var)
        for task, var_instr in instruction.items()
        for var in var_instr.keys()
    ]
    max_episode_length = get_max_episode_length(args.tasks, args.variations)

    dataset = RLBenchDataset(
        root=args.dataset,
        taskvar=taskvar,
        instructions=instruction,
        max_episode_length=max_episode_length,
        max_episodes_per_taskvar=args.max_episodes_per_taskvar,
        cache_size=args.cache_size,
        cameras=args.cameras,  # type: ignore
    )

    bounds = {task: [] for task in args.tasks}

    print(f"Computing gripper location bounds for tasks {args.tasks} from dataset of "
          f"length {len(dataset)}")

    for i in range(len(dataset)):
        ep = dataset[i]
        bounds[ep["task"]].append(ep["action"][:, :3])

    bounds = {
        task: [
            torch.stack(gripper_locs, dim=0).min(dim=0).values.min(dim=0).values.tolist(),
            torch.stack(gripper_locs, dim=0).max(dim=0).values.max(dim=0).values.tolist()
        ]
        for task, gripper_locs in bounds.items()
        if len(gripper_locs) > 0
    }

    pprint.pprint(bounds)
    json.dump(bounds, open("location_bounds.json", "w"), indent=4)
