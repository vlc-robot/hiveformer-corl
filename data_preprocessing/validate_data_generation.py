import glob


RAW_DIR = f"/projects/katefgroup/analogical_manipulation/rlbench/raw"
PACKAGED_DIR = f"/projects/katefgroup/analogical_manipulation/rlbench/packaged"
TRAIN_DIR = "18_peract_tasks_train"
VAL_DIR = "18_peract_tasks_val"
# TRAIN_DIR = "74_hiveformer_tasks_train"
# VAL_DIR = "74_hiveformer_tasks_val"


if __name__ == "__main__":
    for split in [TRAIN_DIR, VAL_DIR]:
        print()
        print()
        print()
        print("Split: ", split)
        print()

        raw_dirs = glob.glob(f"{RAW_DIR}/{split}/*")

        for raw_val_dir in raw_dirs:
            task = raw_val_dir.split("/")[-1]
            raw_variation_dirs = glob.glob(f"{raw_val_dir}/*")
            packaged_variation_dirs = glob.glob(f"{PACKAGED_DIR}/{split}/{task}*")
            raw_eps_per_variation = [len(glob.glob(f"{d}/episodes/*")) for d in raw_variation_dirs]
            packaged_eps_per_variation = [len(glob.glob(f"{d}/*")) for d in packaged_variation_dirs]
            print("=========================================")
            print(task)
            print(f"Variations: {len(raw_variation_dirs)} raw, {len(packaged_variation_dirs)} packaged")
            print(f"Episodes per variation: {raw_eps_per_variation} raw, {packaged_eps_per_variation} packaged")
            print(f"Total episodes: {sum(raw_eps_per_variation)} raw, {sum(packaged_eps_per_variation)} packaged")
