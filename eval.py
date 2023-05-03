import random
from typing import Tuple, Optional
from copy import deepcopy
from pathlib import Path
import torch
import numpy as np
import tap
import json
import os

from train import Arguments as TrainArguments
from model.released_hiveformer.network import Hiveformer
from model.non_analogical_baseline.baseline import Baseline
from model.analogical_network.analogical_network import AnalogicalNetwork
from utils.utils_with_rlbench import (
    RLBenchEnv,
    Actioner,
)
from utils.utils_without_rlbench import (
    load_episodes,
    load_instructions,
    get_max_episode_length,
    get_gripper_loc_bounds,
    TASK_TO_ID,
    round_floats
)


class Arguments(tap.Tap):
    checkpoint: Path
    seed: int = 2
    save_img: bool = True
    device: str = "cuda"
    num_episodes: int = 1
    headless: int = 0
    max_tries: int = 10
    record_actions: bool = False
    replay_actions: Optional[Path] = None
    ground_truth_rotation: bool = False
    ground_truth_position: bool = False
    ground_truth_gripper: bool = False
    tasks: Optional[Tuple[str, ...]] = None
    instructions: Optional[Path] = "instructions.pkl"
    arch: Optional[str] = None
    variations: Tuple[int, ...] = (0,)
    data_dir: Path = Path(__file__).parent / "demos"
    cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist")
    image_size: str = "256,256"
    verbose: int = 0
    output_file: Path = Path(__file__).parent / "eval.json"
    
    # Logging to base_log_dir/exp_log_dir/run_log_dir
    base_log_dir: Path = Path(__file__).parent / "eval_logs"
    exp_log_dir: str = "exp"
    run_log_dir: str = "run"
    
    # Toggle to switch between offline and online evaluation
    offline: int = 0

    # Toggle to switch between original HiveFormer and our models
    model: str = "baseline"  # one of "original", "baseline", "analogical"

    record_videos: int = 0
    max_steps: int = 50
    collision_checking: int = 0

    # ---------------------------------------------------------------
    # Original HiveFormer parameters
    # ---------------------------------------------------------------

    depth: int = 4
    dim_feedforward: int = 64
    hidden_dim: int = 64
    instr_size: int = 512
    mask_obs_prob: float = 0.0
    num_layers: int = 1

    # ---------------------------------------------------------------
    # Our non-analogical baseline parameters
    # ---------------------------------------------------------------

    visualize_rgb_attn: int = 0
    gripper_loc_bounds_file: str = "tasks/74_hiveformer_tasks_location_bounds.json"
    single_task_gripper_loc_bounds: int = 0
    gripper_bounds_buffer: float = 0.04

    position_prediction_only: int = 0
    regress_position_offset: int = 0

    # Ghost points
    num_sampling_level: int = 3
    fine_sampling_ball_diameter: float = 0.16
    weight_tying: int = 1
    gp_emb_tying: int = 1
    num_ghost_points: int = 10000
    num_ghost_points_val: int = 10000

    # Model
    backbone: str = "clip"  # one of "resnet", "clip"
    embedding_dim: int = 60
    num_ghost_point_cross_attn_layers: int = 2
    num_query_cross_attn_layers: int = 2
    # one of "quat_from_top_ghost", "quat_from_query", "6D_from_top_ghost", "6D_from_query"
    rotation_parametrization: str = "quat_from_query"
    use_instruction: int = 0
    task_specific_biases: int = 0

    # Positional features
    positional_features: Optional[str] = "none"  # one of "xyz_concat", "z_concat", "xyz_add", "z_add", "none"

    # ---------------------------------------------------------------
    # Our analogical network additional parameters
    # ---------------------------------------------------------------

    support_set: str = "others"  # one of "self" (for debugging), "others"
    support_set_size: int = 1
    global_correspondence: int = 0
    num_matching_cross_attn_layers: int = 2


def get_log_dir(args: Arguments) -> Path:
    log_dir = args.base_log_dir / args.exp_log_dir

    def get_log_file(version):
        log_file = f"{args.run_log_dir}_version{version}"
        return log_file

    version = 0
    while (log_dir / get_log_file(version)).is_dir():
        version += 1

    return log_dir / get_log_file(version)


def copy_args(checkpoint: Path, args: Arguments) -> Arguments:
    args = deepcopy(args)

    print("Copying args from", checkpoint)

    # Update args accordingly:
    hparams = checkpoint.parent / "hparams.json"
    print(hparams, hparams.is_file())
    if hparams.is_file():
        print("Loading args from checkpoint")
        train_args = TrainArguments()
        train_args.load(str(hparams))
        for key in args.class_variables:
            v = getattr(args, key)
            if v is None and key in train_args.class_variables:
                setattr(args, key, getattr(train_args, key))
                print("Copying", key, ":", getattr(args, key))

    return args


def load_model(checkpoint: Path, args: Arguments) -> Hiveformer:
    args = copy_args(checkpoint, args)
    device = torch.device(args.device)

    print("Loading model from", checkpoint, flush=True)

    if (
        args.depth is None
        or args.dim_feedforward is None
        or args.hidden_dim is None
        or args.instr_size is None
        or args.mask_obs_prob is None
        or args.num_layers is None
    ):
        raise ValueError("Please provide the missing parameters")

    max_episode_length = get_max_episode_length(args.tasks, args.variations)

    # Gripper workspace is the union of workspaces for all tasks
    if args.single_task_gripper_loc_bounds and len(args.tasks) == 1:
        task = args.tasks[0]
    else:
        task = None
    gripper_loc_bounds = get_gripper_loc_bounds(
        args.gripper_loc_bounds_file, task=task, buffer=args.gripper_bounds_buffer)

    if args.model == "original":
        model = Hiveformer(
            depth=args.depth,
            dim_feedforward=args.dim_feedforward,
            hidden_dim=args.hidden_dim,
            instr_size=args.instr_size,
            mask_obs_prob=args.mask_obs_prob,
            max_episode_length=max_episode_length,
            num_layers=args.num_layers,
        ).to(device)
    elif args.model == "baseline":
        model = Baseline(
            backbone=args.backbone,
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=args.embedding_dim,
            num_ghost_point_cross_attn_layers=args.num_ghost_point_cross_attn_layers,
            num_query_cross_attn_layers=args.num_query_cross_attn_layers,
            rotation_parametrization=args.rotation_parametrization,
            gripper_loc_bounds=gripper_loc_bounds,
            num_ghost_points=args.num_ghost_points,
            num_ghost_points_val=args.num_ghost_points_val,
            weight_tying=bool(args.weight_tying),
            gp_emb_tying=bool(args.gp_emb_tying),
            num_sampling_level=args.num_sampling_level,
            fine_sampling_ball_diameter=args.fine_sampling_ball_diameter,
            regress_position_offset=bool(args.regress_position_offset),
            visualize_rgb_attn=bool(args.visualize_rgb_attn),
            use_instruction=bool(args.use_instruction),
            task_specific_biases=bool(args.task_specific_biases),
            positional_features=args.positional_features,
            task_ids=[TASK_TO_ID[task] for task in args.tasks],
        ).to(device)
    elif args.model == "analogical":
        raise NotImplementedError
        model = AnalogicalNetwork(
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=args.embedding_dim,
            num_ghost_point_cross_attn_layers=args.num_ghost_point_cross_attn_layers,
            rotation_parametrization=args.rotation_parametrization,
            gripper_loc_bounds=gripper_loc_bounds,
            num_ghost_points=args.num_ghost_points,
            coarse_to_fine_sampling=bool(args.coarse_to_fine_sampling),
            fine_sampling_ball_diameter=args.fine_sampling_ball_diameter,
            separate_coarse_and_fine_layers=bool(args.separate_coarse_and_fine_layers),
            regress_position_offset=bool(args.regress_position_offset),
            support_set=args.support_set,
            global_correspondence=args.global_correspondence,
            num_matching_cross_attn_layers=args.num_matching_cross_attn_layers,
            use_instruction=bool(args.use_instruction),
            task_specific_biases=bool(args.task_specific_biases),
            task_ids=[TASK_TO_ID[task] for task in args.tasks],
        ).to(device)

    if hasattr(model, "film_gen") and model.film_gen is not None:
        model.film_gen.build(device)

    # model_dict = torch.load(checkpoint, map_location="cpu")["weight"]
    # model_dict = {(k[7:] if k.startswith("module.") else k): v
    #               for k, v in model_dict.items()}

    model_dict = torch.load(checkpoint, map_location="cpu")
    model_dict_weight = {}
    for key in model_dict["weight"]:
        _key = key[7:]
        # if 'prediction_head.feature_pyramid.inner_blocks' in _key:
        #     _key = _key[:46] + _key[48:]
        # if 'prediction_head.feature_pyramid.layer_blocks' in _key:
        #     _key = _key[:46] + _key[48:]
        model_dict_weight[_key] = model_dict["weight"][key]
    model.load_state_dict(model_dict_weight)

    model.eval()

    return model


def find_checkpoint(checkpoint: Path) -> Path:
    if checkpoint.is_dir():
        candidates = [c for c in checkpoint.rglob("*.pth") if c.name != "best"]
        candidates = sorted(candidates, key=lambda p: p.lstat().st_mtime)
        assert candidates != [], checkpoint
        return candidates[-1]

    return checkpoint


if __name__ == "__main__":
    args = Arguments().parse_args()

    if args.tasks is None:
        print(args.checkpoint)
        args.tasks = ["_".join(str(args.checkpoint).split("/")[-2].split("_")[:-1])]
        print(f"Automatically setting task to {args.tasks}")

    log_dir = get_log_dir(args)
    log_dir.mkdir(exist_ok=True, parents=True)
    print("log dir", log_dir)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load model and args
    checkpoint = find_checkpoint(args.checkpoint)
    args = copy_args(checkpoint, args)
    if checkpoint is None:
        raise RuntimeError()
    model = load_model(checkpoint, args)

    # load RLBench environment
    env = RLBenchEnv(
        data_path=args.data_dir,
        image_size=[int(x) for x in args.image_size.split(",")],
        apply_rgb=True,
        apply_pc=True,
        headless=bool(args.headless),
        apply_cameras=args.cameras,
        collision_checking=bool(args.collision_checking),
        # TODO Is there a way to display the fine sampling ball transparently with Open3D?
        # fine_sampling_ball_diameter=args.fine_sampling_ball_diameter if model != "original" else None,
    )

    instruction = load_instructions(args.instructions)
    if instruction is None:
        raise NotImplementedError()

    actioner = Actioner(model=model, instructions=instruction)
    max_eps_dict = load_episodes()["max_episode_length"]
    task_success_rates = {}

    for task_str in args.tasks:
        var_success_rates = env.evaluate_task_on_multiple_variations(
            task_str,
            max_steps=max_eps_dict[task_str] if args.max_steps == -1 else args.max_steps,
            num_variations=args.variations[-1] + 1,
            num_demos=args.num_episodes,
            actioner=actioner,
            log_dir=log_dir / task_str if args.save_img else None,
            max_tries=args.max_tries,
            save_attn=False,
            record_videos=bool(args.record_videos),
            position_prediction_only=bool(args.position_prediction_only),
            offline=bool(args.offline),
            verbose=bool(args.verbose),
        )
        print()
        print(f"{task_str} variation success rates:", round_floats(var_success_rates))
        print(f"{task_str} mean success rate:", round_floats(var_success_rates["mean"]))

        task_success_rates[task_str] = var_success_rates
        with open(args.output_file, "w") as f:
            json.dump(round_floats(task_success_rates), f, indent=4)
