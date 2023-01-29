"""
Convert Pickle data file into a proper dataset
"""
import random
from itertools import permutations
from typing import List, TypedDict, Dict, Tuple, Any, Union
from typing_extensions import NotRequired
from pathlib import Path
import json
from tqdm import tqdm
import tap
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class Stats(TypedDict):
    intact: NotRequired[int]
    augment: NotRequired[int]


class Arguments(tap.Tap):
    data_dir: Path = Path(__file__).parent / "sim2real"
    output: Path = Path(__file__).parent / "datasets"
    num_workers: int = 0
    max_variations: int = 200
    variations: Path = Path(__file__).parent / "push_buttons.json"
    instruction: Path = Path(__file__).parent / "annotations-push_repeated_buttons.json"


Color = List[float]
NamedColor = List[List[Union[str, Color]]]


def augment_variations(
    variation_file: Path,
) -> Tuple[List[NamedColor], Dict[int, Tuple[int, NamedColor]]]:
    # load variations
    with open(variation_file) as fid:
        old_variations: List[NamedColor] = json.load(fid)
    old_variations = old_variations[: args.max_variations]

    random.seed(0)
    new_variations: Dict[int, Tuple[int, NamedColor]] = {}
    for i, var in enumerate(old_variations):
        new_variations[i] = (i, var)

        augmented_variations: List[NamedColor] = [var]
        for c in var:
            aug_vars: List[NamedColor] = [avar for avar in permutations(list(var) + [c])]
            augmented_variations += aug_vars

        for j, avar in enumerate(augmented_variations):
            key = 10_000 + i * 100 + j
            if key in new_variations:
                raise RuntimeError(key)
            new_variations[key] = (i, avar)

    return old_variations, new_variations


def generate_repeated_buttons_instructions(colors):
    rtn0 = "push the %s button" % colors[0]
    rtn1 = "press the %s button" % colors[0]
    rtn2 = "push down the button with the %s base" % colors[0]

    for i, color in enumerate(colors):
        if i == 0:
            continue
        elif color == colors[i - 1]:
            rtn0 += " twice"
            rtn1 += ", then do it again"
            rtn2 += ", then the %s one" % color
        elif color in colors[: i - 1]:
            rtn0 += ", then the %s button" % color
            rtn1 += ", then press the %s one again" % color
            rtn2 += ", then the %s one again" % color
        else:
            rtn0 += ", then the %s button" % color
            rtn1 += ", then press the %s one" % color
            rtn2 += ", then the %s one" % color
    return [rtn0, rtn1, rtn2]


def load_items(args: Arguments):
    args = args

    orig_variations, new_variations = augment_variations(args.variations)

    items = []
    instructions = []
    for avar_id, (ovar_id, avar) in tqdm(new_variations.items()):
        orig_dir = args.data_dir / f"push_buttons+{ovar_id}"
        dest_dir = args.output / f"push_repeated_buttons+{avar_id}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        episodes = [(ep, dest_dir / ep.name, avar_id) for ep in orig_dir.glob("ep*.npy")]
        items += sorted(episodes)

        colors = [c[0] for c in avar]
        for instr in generate_repeated_buttons_instructions(colors):
            instructions.append(
                {
                    "fields": {
                        "task": "push_repeated_buttons",
                        "variation": avar_id,
                        "instruction": instr,
                    }
                }
            )

    with open(args.instruction, "w") as fid:
        json.dump(instructions, fid, indent=2)

    return items, orig_variations, new_variations


class ConvertDataset(Dataset):
    def __init__(self, items, orig_variations, new_variations):
        self.items = items
        self.num_items = len(self.items)
        self.orig_variations = orig_variations
        self.new_variations = new_variations

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, index: int) -> Stats:
        orig_file, dest_file, avar_id = self.items[index]
        ovar_id, avar = self.new_variations[avar_id]
        ovar = self.orig_variations[ovar_id]

        sample = np.load(orig_file, allow_pickle=True)

        if ovar == avar:
            np.save(dest_file, sample)
            return {"intact": 1}

        # add final frame
        last_state, last_gripper, last_attn_indices = sample[5][0]
        sample[1].append(last_state)
        sample[4].append(last_gripper)
        sample[3].append(last_attn_indices)

        frame_ids = list(range(len(avar)))
        state_ls = [sample[1][0]]
        action_ls = []
        attn_indices = [sample[3][0]]
        gripper_pos = [sample[4][0]]

        for step in avar:
            orig_step_id = ovar.index(step)

            state_ls.append(sample[1][1 + 2 * orig_step_id])
            state_ls.append(sample[1][2 + 2 * orig_step_id])

            action_ls.append(sample[2][2 * orig_step_id])
            action_ls.append(sample[2][1 + 2 * orig_step_id])

            attn_indices.append(sample[3][1 + 2 * orig_step_id])
            attn_indices.append(sample[3][2 + 2 * orig_step_id])

            gripper_pos.append(sample[4][1 + 2 * orig_step_id])
            gripper_pos.append(sample[4][2 + 2 * orig_step_id])

        new_sample = [
            frame_ids,
            state_ls[:-1],
            action_ls,
            attn_indices[:-1],
            gripper_pos[:-1],
            [state_ls[-1], gripper_pos[-1]],
        ]
        print(dest_file)

        np.save(dest_file, new_sample)
        return {"augment": 1}


if __name__ == "__main__":
    args = Arguments().parse_args()

    items, orig_variations, new_variations = load_items(args)

    dataset = ConvertDataset(items, orig_variations, new_variations)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
    )

    stats: Stats = {"augment": 0, "intact": 0}

    for batch in tqdm(dataloader):
        s: Stats
        for s in batch:
            for k, v in s.items():
                stats[k] += v

    print(stats)
