"""
Estimate a dataset length
"""
from pickle import UnpicklingError
from pathlib import Path
import numpy as np
from tqdm import tqdm
from utils import load_episodes


# load all tasks
with open(Path(__file__).parent / "tasks.csv", "r") as tid:
    tasks = [l.strip() for l in tid.readlines()]

dataset = Path("datasets") / "dataset-0"
episodes = load_episodes()

for task in tasks:
    if task in episodes["max_episode_length"]:
        continue
    lengths = []
    episode_files = list(dataset.glob(f"{task}+*/ep*.npy"))
    episode_files = episode_files[:20]
    for ep in tqdm(episode_files):
        data = np.load(ep, allow_pickle=True)

        frame_ids = data[0]
        num_frames = len(frame_ids)
        lengths.append(num_frames)
        # expect = episodes['max_episode_length'][task_str]

    if lengths == []:
        print("No episode for", task)
        continue

    np_lengths = np.array(lengths)
    print(task, np.percentile(np_lengths, 0.9), np.std(np_lengths))
