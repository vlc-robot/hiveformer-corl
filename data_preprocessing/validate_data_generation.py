import glob


ROOT = "/home/zhouxian/git"
DATA_DIR = f"{ROOT}/datasets/raw"
OUTPUT_DIR = f"{ROOT}/datasets/packaged"
TRAIN_DIR = "18_peract_tasks_train"
VAL_DIR = "18_peract_tasks_val"
TASK_FILE = "tasks/74_hiveformer_tasks.csv"


if __name__ == "__main__":
    raw_val_dirs = glob.glob(f"{DATA_DIR}/{VAL_DIR}/*")
    for raw_val_dir in raw_val_dirs:
        task = raw_val_dir.split("/")[-1]
        variation_dirs = glob.glob(f"{raw_val_dir}/*")
        eps_per_variation = [len(glob.glob(f"{variation_dir}/*/episodes/*"))
                             for variation_dir in variation_dirs]
        print(f"{task}: {len(variation_dirs)} variations, {eps_per_variation} episodes per variation")
