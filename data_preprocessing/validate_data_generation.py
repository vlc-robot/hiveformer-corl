import glob
import tap
from typing import Tuple

from utils.utils_with_rlbench import RLBenchEnv
from utils.utils_without_rlbench import load_episodes


class Arguments(tap.Tap):
    cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist")
    image_size: str = "256,256"
    headless: bool = False
    max_tries: int = 10

    # raw_dir: str = "/projects/katefgroup/analogical_manipulation/rlbench/raw"
    # packaged_dir: str = "/projects/katefgroup/analogical_manipulation/rlbench/packaged"
    raw_dir: str = "/home/zhouxian/git/datasets/raw"
    packaged_dir: str = "/home/zhouxian/git/datasets/packaged"

    # train_dir: str = "18_peract_tasks_train"
    # val_dir: str = "18_peract_tasks_val"
    # train_dir: str = "74_hiveformer_tasks_train"
    # val_dir: str = "74_hiveformer_tasks_val"
    train_dir: str = "peract_problematic_tasks_train"
    val_dir: str = "peract_problematic_tasks_val"

    validate_num_episodes: int = 0
    validate_successful_demos: int = 1


if __name__ == "__main__":
    args = Arguments().parse_args()

    # Validate that we have generated the expected number of episodes for each task
    if args.validate_num_episodes:
        for split in [args.train_dir, args.val_dir]:
            print()
            print()
            print()
            print("Split: ", split)
            print()

            raw_dirs = glob.glob(f"{args.raw_dir}/{split}/*")

            for raw_dir in raw_dirs:
                task_str = raw_dir.split("/")[-1]
                raw_variation_dirs = glob.glob(f"{raw_dir}/*")
                packaged_variation_dirs = glob.glob(f"{args.packaged_dir}/{split}/{task_str}*")
                raw_eps_per_variation = [len(glob.glob(f"{d}/episodes/*")) for d in raw_variation_dirs]
                packaged_eps_per_variation = [len(glob.glob(f"{d}/*")) for d in packaged_variation_dirs]
                print("=========================================")
                print(task_str)
                print(f"Variations: {len(raw_variation_dirs)} raw, {len(packaged_variation_dirs)} packaged")
                print(f"Episodes per variation: {raw_eps_per_variation} raw, {packaged_eps_per_variation} packaged")
                print(f"Total episodes: {sum(raw_eps_per_variation)} raw, {sum(packaged_eps_per_variation)} packaged")

    # Validate that the generated demos are successful
    if args.validate_successful_demos:
        max_eps_dict = load_episodes()["max_episode_length"]

        for split in [args.train_dir, args.val_dir]:
            print()
            print()
            print()
            print("Split: ", split)
            print()

            env = RLBenchEnv(
                data_path=f"{args.raw_dir}/{split}",
                image_size=[int(x) for x in args.image_size.split(",")],
                apply_rgb=True,
                apply_pc=True,
                headless=args.headless,
                apply_cameras=args.cameras,
            )

            task_dirs = glob.glob(f"{args.raw_dir}/{split}/*")

            for task_dir in task_dirs:
                task_str = task_dir.split("/")[-1]
                var_dirs = glob.glob(f"{task_dir}/*")

                print("=========================================")
                print(f"{task_str} with {len(var_dirs)} variations")
                success_rates = {}
                for var_dir in var_dirs:
                    variation = int(var_dir.split("variation")[-1])
                    ep_dirs = glob.glob(f"{var_dir}/episodes/*")
                    num_demos = len(ep_dirs)
                    success_rate = env.verify_demos(
                        task_str=task_str,
                        variation=variation,
                        num_demos=num_demos,
                        max_tries=args.max_tries,
                        verbose=False
                    )
                    success_rates[variation] = success_rate
                print("Success rates:", success_rates)
