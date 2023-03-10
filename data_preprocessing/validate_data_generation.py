import glob


ROOT = "/home/zhouxian/git"
DATA_DIR = f"{ROOT}/datasets/raw"
OUTPUT_DIR = f"{ROOT}/datasets/packaged"
TRAIN_DIR = "74_hiveformer_tasks_train"
VAL_DIR = "74_hiveformer_tasks_val"
TASK_FILE = "tasks/74_hiveformer_tasks.csv"


if __name__ == "__main__":
    raw_files = glob.glob(f"{DATA_DIR}/{VAL_DIR}/*")
    print(raw_files)
