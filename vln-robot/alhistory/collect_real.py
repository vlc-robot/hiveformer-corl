from typing import List, Optional, Tuple
from copy import copy
import pickle as pkl
import json
import math
from pathlib import Path
from tqdm import trange
import numpy as np
import tap
import rospy
import tf
from utils_real import (
    Vector3,
    Item,
    pick,
    place,
    sequential_execution,
    GripperState,
    center_item_position,
    reset_gripper,
    Robot,
)


class Arguments(tap.Tap):
    """Data collection script"""

    output: Path  # Directory to save trajectory
    task: str
    variation: int

    init_seed: int = 0
    variation_desc: Path = Path(__file__).parent / "tower.json"
    num_items: int = 3
    cameras: List[str] = ["charlie", "bravo"]
    num_episodes: int = 11
    crop: bool = False  # Crop observations
    visualize: bool = False  # Visualize sim2real comparison

    cube_height: float = 0.05
    radius: float = 0.1

    initial_xy: List[Tuple[float, float]] = [
        (-0.745, 0.225),
        (-0.745, 0.0),
        (-0.745, -0.225),
        (-0.81, 0.115),
        (-0.81, -0.115),
    ]
    target: Vector3 = (-0.48, 0.0, 0.0001)
    initial_height: np.ndarray = np.array([0.06, 0.10])
    default_orn: Vector3 = (math.pi, 0, math.pi / 2)
    workspace: np.ndarray = np.array([[-0.695, -0.175, 0.00], [-0.295, 0.175, 0.2]])
    boundary: np.ndarray = np.array([[-0.85, -0.225, 0.00], [-0.295, 0.225, 0.2]])

    def parse_args(self):
        args = super().parse_args()
        if len(args.initial_xy) < args.num_items:
            raise ValueError("Add initial locations")
        return args


class Environment:
    def __init__(self, args: Arguments):
        self.args = args
        self.seed: Optional[int] = 0

        # load description of variations
        with open(self.args.variation_desc) as fid:
            variations = json.load(fid)
        variation = variations[args.variation]
        rospy.loginfo(f"variation {args.variation}: {variation}")

        z = self.args.cube_height / 2
        quat = tf.transformations.quaternion_from_euler(*self.args.default_orn)

        self.items = [
            Item(desc, (x, y, z), quat)
            for desc, (x, y) in zip(variation, self.args.initial_xy)
        ]
        self.items = self.items[: self.args.num_items]
        num_items = len(self.items)
        rospy.loginfo(f"items: {num_items}")

        dist_positions = self.args.initial_xy[num_items : self.args.num_items]
        self.distractors = [
            Item(f"distractor {i}", (x, y, z), quat)
            for i, (x, y) in enumerate(dist_positions)
        ]
        rospy.loginfo(f"num distractors: {len(self.distractors)}")
        # print("distractor", self.distractors)

        self.target = Item("target", self.args.target, quat)

        self.robot = Robot(self.args.cameras, self.args.workspace, self.args.boundary)

        # Sanity check
        self.robot.render()

    def reset_item(
        self,
        item: Item,
        obstacles: List[Item],
        random_state: np.random.RandomState,
        max_tries: int = 100,
    ) -> bool:
        position: Optional[Vector3] = None
        working = False
        for _ in range(max_tries):
            # sample position
            position = random_state.uniform(  # type: ignore
                self.args.workspace[0], self.args.workspace[1]
            )
            if not isinstance(position, tuple):
                raise RuntimeError()
            position = tuple([position[0], position[1], self.args.cube_height / 2])

            # check constraints
            working = True
            for obs in obstacles:
                obs_pos = np.array(obs.position[:2])
                dist = np.sqrt(((obs_pos - position[:2]) ** 2).mean())
                if dist < self.args.radius:
                    working = False
                    break

            if working:
                break

        if position is None or not working:
            return False

        states = pick(item)
        item.position = position
        states += place(item)

        return sequential_execution(states, self.robot)

    def _build_trajectory(self) -> List[Tuple[bool, GripperState]]:
        record_states = []
        target = [*self.target.position[:2], self.args.cube_height / 2]

        for item in self.items:
            pick_states = pick(item)
            record_states += [
                (i == len(pick_states) - 2, state) for i, state in enumerate(pick_states)
            ]

            item.position = tuple(target)
            place_states = place(item)
            record_states += [
                (i == len(place_states) - 2, state)
                for i, state in enumerate(place_states)
            ]

            target[-1] += self.args.cube_height

        return record_states[:-1]

    def step(self, action):
        raise NotImplementedError()

    def collect(self, seed: int):
        """
        Collect a demonstration on the robot
        """
        random_state = np.random.RandomState(seed)
        sorted_items = sorted(
            self.items + self.distractors, key=lambda x: x.position[2], reverse=True
        )

        # reset items location starting by the highest items
        for i, item in enumerate(sorted_items):
            rospy.loginfo(f"Resetting {item}")
            obstacles = [item_ for j, item_ in enumerate(sorted_items) if j != i]
            obstacles.append(self.target)
            for i in range(2):
                if self.reset_item(item, obstacles, random_state):
                    break
                if i == 1:
                    raise RuntimeError(f"Can't reset item {item}")

            # improve accuracy by centering around its current location
            trajectory = center_item_position(item)
            success = sequential_execution(trajectory, self.robot)
            if not success:
                rospy.logerr(f"Can't center item {item}")
                return []

        # reset gripper position
        gripper_workspace = self.args.workspace.copy()
        gripper_workspace[:, 2] = self.args.initial_height
        trajectory = reset_gripper(gripper_workspace, random_state)
        success = sequential_execution(trajectory, self.robot)
        if not success:
            rospy.logerr(f"Can't reset gripper {seed}")
            return []

        recorded_states = self._build_trajectory()
        records = [self.robot.render()]

        for record, state in recorded_states:
            self.robot.execute(state)
            if record:
                records.append(self.robot.render())

            if rospy.is_shutdown():
                break

        return records

    # def dismount(self):
    #     """
    #     Put objects at their initial locations
    #     """
    #     items_highest = sorted(items, itemgetter(
    #     states = build_trajectory(self.items, self.args.workspace)
    #     records = []

    #     for record, state in recorded_states:
    #         state.execute(self.robot)
    #         if record:
    #             records.append(self.robot.render())

    #         if rospy.is_shutdown():
    #             break

    #     return records

    def save(self, seed: int, traj: List):
        # create output dirs
        real_exp_dir = (
            self.args.output
            / self.args.task
            / f"variation{self.args.variation}"
            / "episodes"
        )
        real_exp_dir.mkdir(parents=True, exist_ok=True)

        for i, step in enumerate(traj):
            with open(str(real_exp_dir / f"{seed:04d}_{i:05d}.pkl"), "wb") as f:
                pkl.dump(step, f)


def main(args: Arguments):

    env = Environment(args)

    last_seed = args.init_seed + args.num_episodes
    for seed in trange(args.init_seed, last_seed):
        seed_var = args.variation * 100 + seed
        traj = env.collect(seed_var)
        env.save(seed, traj)

        # if seed != last_seed - 1:
        #     env.dismount()

    if rospy.is_shutdown():
        exit()


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    main(args)
