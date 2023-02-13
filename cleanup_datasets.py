from pickle import UnpicklingError
from pathlib import Path
import numpy
from tqdm import tqdm
from utils import load_episodes


class ObsoleteError(Exception):
    """The file is too old"""


files = sorted(list(Path("datasets").rglob("ep*.npy")))
episodes = load_episodes()

for f in tqdm(files):
    try:
        data = numpy.load(f, allow_pickle=True)
        if len(data) < 5:
            raise ObsoleteError

    except (UnpicklingError, EOFError, ObsoleteError):
        print("REMOVING", f)
        # f.unlink()
        continue

    task_str = f.parent.name.split("+")[0]

    frame_ids = data[0]
    num_frames = len(frame_ids)

    if task_str not in episodes["max_episode_length"]:
        print(f"{task_str} is unknown. Size is {num_frames}")
        continue

    expect = episodes["max_episode_length"][task_str]

    if task_str in episodes["variable_length"]:
        if num_frames > expect:
            print("TOO LONG", num_frames, expect, f)
            f.unlink()
    elif num_frames != expect:
        print("UNEXPECTED SIZE", num_frames, expect, f)
        f.unlink()
