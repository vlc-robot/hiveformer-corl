import random
from typing import Tuple, Dict, Optional, TypedDict, List
import pickle
import json
from copy import deepcopy
from pathlib import Path
from tqdm import trange
import tap
import torch
from torch import nn
import numpy as np
import tap
from PIL import Image
from tap.utils import as_python_object
from filelock import FileLock
import einops
import rospy
import sys
from structures import (
    RotMode,
    Observation,
    BackboneOp,
    TransformerToken,
    Output,
    InstructionMode,
    GripperPose,
    ZMode,
    Instructions,
    PointCloudToken,
)
from network_sim2real import PlainUNet, TransformerUNet
from utils_sim2real import (
    load_episodes,
    get_max_episode_length,
    transform,
    get_extrinsics,
    get_intrinsics,
    square,
    rgb_resize,
    depth_resize,
    depth_to_pcd,
    get_gripper_state,
    obs_to_attn,
    Misc,
    RotType,
    load_rotation,
    Model,
    plot_attention,
    get_obs_action,
)
import rospy
from utils_real import (
    Vector3,
    Vector4,
    Item,
    place,
    pick,
    sequential_execution,
    reset_gripper,
    Robot,
)
from utils_instructions import MicToText, TextToToken


def load_instructions(instructions: Optional[Path]) -> Optional[Instructions]:
    if instructions is not None:
        with open(instructions, "rb") as fid:
            print(instructions)
            return pickle.load(fid)
    return None


def record_all_observations(ep_dir: Path, **kwargs):
    with open(ep_dir / "states.npy", "wb") as fid:
        pickle.dump(kwargs, fid)


class Arguments(tap.Tap):
    task: str
    variation: int
    checkpoint: Optional[Path] = None
    seed: int = 2
    save_img: bool = False
    device: str = "cuda"
    num_episodes: int = 100
    headless: bool = False
    offset: int = 0
    name: str = "autol"
    output: Path = Path(__file__).parent / "records.txt"
    xp: Path = Path(__file__).parent / "xp"
    test_xp: Path = Path(__file__).parent / "test-xp"
    data_dir: Path = Path(__file__).parent / "demos"
    instructions: Optional[Path] = None
    variations: Tuple[int, ...] = (0,)
    attention: bool = False  # saving attention maps
    instr_mode: InstructionMode = "precompute"
    # model
    arch: Optional[str] = None
    attn_weights: Optional[bool] = None
    backbone: Optional[BackboneOp] = None
    cond: Optional[bool] = None
    depth: Optional[int] = None
    dim_feedforward: Optional[int] = None
    embed_only: Optional[bool] = None
    film: Optional[bool] = None
    film_mlp: Optional[bool] = None
    film_residual: Optional[bool] = None
    gripper_pose: Optional[GripperPose] = None
    hidden_dim: Optional[int] = None
    instr_size: Optional[int] = None
    mask_obs_prob: float = 0.0
    no_residual: Optional[bool] = None
    num_layers: Optional[int] = None
    pcd_token: Optional[PointCloudToken] = None
    rot: Optional[RotMode] = None
    rot_type: Optional[RotType] = None
    stateless: Optional[bool] = None
    taskvar_token: Optional[bool] = None
    tr_token: Optional[TransformerToken] = None
    temp_len: Optional[int] = None
    z_mode: Optional[ZMode] = None

    # real world parameters
    cameras: List[str] = ["charlie", "bravo"]
    initial_height: np.ndarray = np.array([0.06, 0.10])
    workspace: np.ndarray = np.array([[-0.695, -0.175, 0.00], [-0.295, 0.175, 0.2]])
    boundary: np.ndarray = np.array([[-0.85, -0.225, 0.00], [-0.295, 0.225, 0.2]])


def get_log_dir(args: Arguments) -> Path:
    log_dir = args.test_xp / args.name
    version = 0
    while (log_dir / f"test-version{version}").is_dir():
        version += 1
    return log_dir / f"test-version{version}"


def copy_args(checkpoint: Path, args: Arguments) -> Arguments:
    args = deepcopy(args)

    if not checkpoint.is_file():
        files = list((args.xp / checkpoint).rglob("mtl_*.pth"))
        assert files != [], args.checkpoint
        files = sorted(files, key=lambda x: x.stat().st_mtime)
        checkpoint = files[0]

    print("Copying args from", checkpoint)

    # Update args accordingly:
    hparams = checkpoint.parent / "hparams.json"
    print(hparams, hparams.is_file())
    if hparams.is_file():
        print("Loading args from checkpoint")
        train_args = Arguments()
        with open(hparams) as f:
            args_dict = json.load(f, object_hook=as_python_object)
        args_dict["task"] = args.task
        args_dict["variation"] = args.variation
        train_args.from_dict(args_dict)
        for key in args.class_variables:
            v = getattr(args, key)
            if v is None and key in train_args.class_variables:
                setattr(args, key, getattr(train_args, key))
                print("Copying", key, ":", getattr(args, key))

    return args


def get_dec_len(args: Arguments) -> int:
    if args.temp_len is None or args.hidden_dim is None:
        raise RuntimeError()

    if args.arch == "mct":
        dec_len = 0
        if args.backbone == "cat":
            dec_len += args.temp_len
            if "tnc" == args.tr_token:
                dec_len += 16
                if args.pcd_token:
                    dec_len += 3
            elif "tnhw_cm_sa" == args.tr_token:
                dec_len += 2 * args.hidden_dim
            else:
                dec_len += args.hidden_dim

            if not args.no_residual:
                dec_len += 16

        else:
            dec_len = 16

    elif args.arch == "plain":
        dec_len = 16
        if args.backbone == "cat":
            dec_len += args.temp_len

    else:
        raise RuntimeError()

    return dec_len


def load_model(checkpoint: Path, args: Arguments) -> Model:
    args = copy_args(checkpoint, args)
    device = torch.device(args.device)

    if not checkpoint.is_file():
        files = list((args.xp / checkpoint).rglob("mtl_*.pth"))
        assert files != [], args.checkpoint
        files = sorted(files, key=lambda x: x.stat().st_mtime)
        checkpoint = files[0]

    print("Loading model from...", checkpoint, flush=True)

    if args.rot is None or args.rot_type is None:
        raise ValueError()
    rotation = load_rotation(args.rot, args.rot_type)

    dec_len = get_dec_len(args)  # type: ignore

    if args.arch == "mct":
        if (
            args.attn_weights is None
            or args.dim_feedforward is None
            or args.embed_only is None
            or args.gripper_pose is None
            or args.hidden_dim is None
            or args.instr_size is None
            or args.no_residual is None
            or args.num_layers is None
            or args.pcd_token is None
            or args.stateless is None
            or args.taskvar_token is None
            or args.tr_token is None
            or args.z_mode is None
        ):
            raise RuntimeError("Please set these parameters")

        max_episode_length = get_max_episode_length((args.task,), args.variations)
        model: PlainUNet = TransformerUNet(
            attn_weights=args.attn_weights,
            backbone_op=args.backbone,
            cond=args.cond,
            depth=args.depth,
            dim_feedforward=args.dim_feedforward,
            dec_len=dec_len,
            embed_only=args.embed_only,
            film=args.film,
            film_mlp=args.film_mlp,
            film_residual=args.film_residual,
            gripper_pose=args.gripper_pose,
            hidden_dim=args.hidden_dim,
            instruction=args.instructions is not None,
            max_episode_length=max_episode_length,
            instr_size=args.instr_size,
            no_residual=args.no_residual,
            num_layers=args.num_layers,
            pcd_token=args.pcd_token,
            rot=rotation,
            stateless=args.stateless,
            temp_len=args.temp_len,
            taskvar_token=args.taskvar_token,
            tr_token=args.tr_token,
            z_mode=args.z_mode,
        ).to(device)
    elif args.arch == "plain":
        if (
            args.backbone is None
            or args.attn_weights is None
            or args.cond is None
            or args.depth is None
            or args.temp_len is None
            or args.film is None
            or args.film_mlp is None
            or args.film_residual is None
            or args.instr_size is None
            or args.z_mode is None
        ):
            raise RuntimeError("Please set these parameters")
        model = PlainUNet(
            attn_weights=args.attn_weights,
            backbone_op=args.backbone,
            cond=args.cond,
            dec_len=dec_len,
            depth=args.depth,
            film=args.film,
            film_mlp=args.film_mlp,
            film_residual=args.film_residual,
            instruction=args.instructions is not None,
            instr_size=args.instr_size,
            rot=rotation,
            temp_len=args.temp_len,
            z_mode=args.z_mode,
        ).to(device)
    else:
        raise RuntimeError(f"Unexpected arch {args.arch}")

    if hasattr(model, "film_gen") and model.film_gen is not None:
        model.film_gen.build(device)

    model_dict = torch.load(checkpoint, map_location="cpu")
    # DEBUG
    model.load_state_dict(model_dict["weight"], strict=False)
    # model.load_state_dict(model_dict["weight"])
    t_dict = {k: t.to(device) for k, t in model_dict["t"].items()}
    z_dict = {k: z.to(device) for k, z in model_dict["z"].items()}

    model.eval()

    return {"model": model, "t": t_dict, "z": z_dict}


def find_checkpoint(checkpoint: Path) -> Path:
    if checkpoint.is_dir():
        candidates = [c for c in checkpoint.rglob("*.pth") if c.name != "best"]
        candidates = sorted(candidates, key=lambda p: p.lstat().st_mtime)
        assert candidates != [], checkpoint
        return candidates[-1]

    return checkpoint


def save_images(step_id: int, images: Dict[str, np.ndarray], ep_dir: Path):
    ep_dir.mkdir(exist_ok=True, parents=True)
    for cam, im in images.items():
        cam_dir = ep_dir / cam
        cam_dir.mkdir(exist_ok=True, parents=True)
        Image.fromarray(im).save(cam_dir / f"{step_id}.png")


def get_observation(
    name: str, obs_mujoco: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fov = float(obs_mujoco[f"info_{name}"]["fovy"])
    euler = torch.Tensor(obs_mujoco[f"info_{name}"]["euler"])
    pos = torch.Tensor(obs_mujoco[f"info_{name}"]["pos"])
    h, w = 128, 128
    extrinsics = get_extrinsics(pos, euler)
    intrinsics2 = get_intrinsics(fov, h, w)

    rgb = obs_mujoco[f"rgb_{name}"]
    rgb = square(rgb)
    rgb2 = rgb_resize(rgb, h, w)

    depth = obs_mujoco[f"depth_{name}"]
    depth2 = depth_resize(depth, extrinsics, fov, h, w)
    pcd2 = depth_to_pcd(depth2, extrinsics, intrinsics2)

    return rgb2, pcd2, extrinsics, intrinsics2


def get_rgb_pcd_gripper_from_obs(
    obs_mujoco: Dict,
    gripper_pose: GripperPose,
    cameras: Tuple[str, ...] = ("wrist", "left_shoulder", "right_shoulder"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return rgb, pcd, and gripper from a given observation
    :param obs: an Observation from the env
    :return: rgb, pcd, gripper
    """
    pose = np.concatenate([obs_mujoco["gripper_pos"], obs_mujoco["gripper_quat"]])

    # FIXME add wrist camera parameters
    (
        wrist_rgb,
        wrist_pcd,
        wrist_extrinsics,
        wrist_intrinsics,
    ) = get_observation("charlie", obs_mujoco)
    ls_rgb, ls_pcd, ls_extrinsics, ls_intrinsics = get_observation("charlie", obs_mujoco)
    rs_rgb, rs_pcd, rs_extrinsics, rs_intrinsics = get_observation("bravo", obs_mujoco)

    gripper_open = get_gripper_state(obs_mujoco["gripper_state"])
    # # DEBUG
    # gripper_open = obs_mujoco["gripper_state"] == -2

    misc: Misc = {
        "wrist_camera_extrinsics": wrist_extrinsics,
        "wrist_camera_intrinsics": wrist_intrinsics[:, :3],
        "left_shoulder_camera_extrinsics": ls_extrinsics,
        "left_shoulder_camera_intrinsics": ls_intrinsics[:, :3],
        "right_shoulder_camera_extrinsics": rs_extrinsics,
        "right_shoulder_camera_intrinsics": rs_intrinsics[:, :3],
    }

    obs = Observation(
        gripper_open=gripper_open,
        gripper_pose=pose,
        # joint_velocities=joint_velocities,
        wrist_rgb=wrist_rgb,
        wrist_pcd=wrist_pcd,
        left_shoulder_rgb=ls_rgb,
        left_shoulder_pcd=ls_pcd,
        right_shoulder_rgb=rs_rgb,
        right_shoulder_pcd=rs_pcd,
        misc=misc,
        loading=False,
    )

    state_dict, gripper = get_obs_action(obs)
    state = transform(state_dict)
    state = einops.rearrange(
        state,
        "(m n ch) h w -> n m ch h w",
        ch=3,
        n=len(cameras),
        m=2,
    )
    rgb = state[:, 0].unsqueeze(0)  # 1, N, C, H, W
    pcd = state[:, 1].unsqueeze(0)  # 1, N, C, H, W
    gripper = gripper.unsqueeze(0)  # 1, D

    if "attn" in gripper_pose:
        attns = torch.Tensor([])
        for cam in cameras:
            u, v = obs_to_attn(obs, cam)
            attn = torch.zeros((1, 1, 1, 128, 128))
            if not (u < 0 or u > 127 or v < 0 or v > 127):
                attn[0, 0, 0, v, u] = 1
            attns = torch.cat([attns, attn], 1)
        rgb = torch.cat([rgb, attns], 2)

    return rgb, pcd, gripper


class Environment:
    def __init__(
        self,
        model: Model,
        instructions: Optional[Path],
        robot: Robot,
        gripper_pose: GripperPose,
        instr_mode: InstructionMode,
        initial_height: np.ndarray,
    ):
        self.gripper_pose: GripperPose = gripper_pose
        self.initial_height: np.ndarray = initial_height
        self.robot = robot

        self._model = model
        self._instr_mode = instr_mode

        if self._instr_mode == "precompute":
            if instructions is None:
                raise ValueError("Please provide path to precomputed instructions")
            self._instructions: Optional[Instructions] = load_instructions(instructions)
        elif self._instr_mode == "mic":
            self._instructions = None
            self._mic_to_text = MicToText()
            self._text_to_token = TextToToken()
        elif self._instr_mode == "text":
            self._instructions = None
            self._text_to_token = TextToToken()
        else:
            raise ValueError(f"Unexpected instr mode {self._instr_mode}.")

    def evaluate(
        self,
        task_str: str,
        max_episodes: int,
        variation: int,
        num_demos: int,
        device,
        log_dir: Optional[Path],
        offset: int = 0,
        save_attn: bool = False,
        seed: int = 0,
    ):
        """
        Evaluate the policy network on the desired demo or test environments
            :param task_type: type of task to evaluate
            :param max_episodes: maximum episodes to finish a task
            :param num_demos: number of test demos for evaluation
            :param model: the policy network
            :param demos: whether to use the saved demos
            :return: success rate
        """
        success_rate = 0.0
        num_trials = 0.0
        random_state = np.random.RandomState(seed)

        with torch.no_grad():
            for demo_id in range(offset, num_demos):

                rgbs = torch.Tensor([]).to(device)
                pcds = torch.Tensor([]).to(device)
                grippers = torch.Tensor([]).to(device)

                # reset a new demo or a defined demo in the demo list
                self.reset(random_state)
                obs = self.robot.render()

                reward = None
                instr_embed = self._get_instruction(task_str, variation).to(device)

                for step_id in range(max_episodes):
                    images = {k: v for k, v in obs.items() if "rgb_" in k}
                    if log_dir is not None:
                        ep_dir = log_dir / f"episode{demo_id}"
                        save_images(step_id, images, ep_dir)

                    # fetch the current observation, and predict one action
                    rgb, pcd, gripper = get_rgb_pcd_gripper_from_obs(
                        obs,
                        self.gripper_pose,
                    )

                    rgb = rgb.to(device)
                    pcd = pcd.to(device)
                    gripper = gripper.to(device)

                    rgbs = torch.cat([rgbs, rgb.unsqueeze(1)], dim=1)
                    pcds = torch.cat([pcds, pcd.unsqueeze(1)], dim=1)
                    grippers = torch.cat([grippers, gripper.unsqueeze(1)], dim=1)
                    padding_mask = torch.ones_like(rgbs[:, :, 0, 0, 0, 0]).bool()

                    output: Output = self._model["model"](
                        rgbs, pcds, padding_mask, None, None, instr_embed, grippers, None
                    )

                    if (
                        log_dir is not None
                        and save_attn
                        and output["attention"] is not None
                    ):
                        ep_dir = log_dir / f"episode{demo_id}"
                        plot_attention(
                            output["attention"][-1],
                            rgbs[0][-1, :, :3],
                            pcds[0][-1].view(3, 3, 128, 128),
                            ep_dir / f"attn_{step_id}.png",
                        )

                    # update the observation based on the predicted action
                    # gripper_open = bool((output["gripper"][-1] < 0))
                    # DEBUG
                    gripper_open = step_id % 2 == 1
                    position = tuple(output["position"][-1].detach().cpu().tolist())
                    rotation = tuple(output["rotation"][-1].detach().cpu().tolist())
                    # print(f"Position: {position} - Gripper: {gripper_open}")
                    obs = self.move(
                        position,
                        rotation,
                        gripper_open,
                    )

                # import ipdb
                # ipdb.set_trace()
                if log_dir is not None:
                    ep_dir = log_dir / f"episode{demo_id}"
                    record_all_observations(
                        ep_dir,
                        rgbs=rgbs.cpu(),
                        pcds=pcds.cpu(),
                        instr_embed=instr_embed.cpu(),
                        gripper=grippers.cpu(),
                    )

                ask = input("S=Success. F=Failure. I=Ignore. Q=Quit >\n")
                if ask == "S":
                    num_trials += 1
                    success_rate += 1 / num_trials
                elif ask == "F":
                    num_trials += 1
                elif ask == "Q":
                    sys.exit(0)

                rospy.loginfo(
                    task_str,
                    "Reward",
                    reward,
                    "Variation",
                    variation,
                    "Step",
                    demo_id,
                    "SR: %.2f" % (success_rate * 100),
                )

        return success_rate

    def _get_instruction(self, task_str: str, variation: int) -> torch.Tensor:
        if self._instr_mode == "precompute":
            if self._instructions is None:
                raise RuntimeError()
            instrs = self._instructions[task_str][variation]
            instr = random.choice(instrs)
            instr = instr.unsqueeze(0)
            print(instr.shape)
            return instr
        if self._instr_mode == "text":
            text = input("What is your order, my lord? (blank for exit)\n")
            if text == "":
                sys.exit(0)
            instr = self._text_to_token(text)
            import pdb

            pdb.set_trace()
            return instr
        if self._instr_mode == "mic":
            text = self._mic_to_text()
            print("Decoded instructions:")
            print(text)
            instr = self._text_to_token(text)
            return instr
        raise RuntimeError(f"Unexpected {self._instr_mode}")

    def move(self, position: Vector3, rotation: Vector4, openness: bool):
        item = Item("virtual", position, rotation)
        fn = place if openness else pick
        states = fn(item)
        # import ipdb
        # ipdb.set_trace()
        success = sequential_execution(states[:-1], self.robot)
        if not success:
            success = sequential_execution(states[:-1], self.robot)
            if not success:
                raise RuntimeError("Can't move robot.")
        obs = self.robot.render()
        success = self.robot.execute(states[-1])

        # try twice
        if not success:
            success = self.robot.execute(states[-1])
            if not success:
                raise RuntimeError("Can't move robot.")

        return obs

    def reset(self, random_state: np.random.RandomState):
        # Put the gripper on a random location
        gripper_workspace = self.robot.workspace.copy()
        gripper_workspace[:, 2] = self.initial_height[:]
        trajectory = reset_gripper(gripper_workspace, random_state)
        success = sequential_execution(trajectory, self.robot)
        if not success:
            raise RuntimeError("Can't reset env")

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


def main():
    args = Arguments().parse_args()
    rospy.loginfo(str(args))
    log_dir = get_log_dir(args)
    log_dir.mkdir(exist_ok=True, parents=True)
    rospy.loginfo(f"Storing outputs in {log_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load model and args
    if args.checkpoint is None:
        raise RuntimeError()
    checkpoint = find_checkpoint(args.checkpoint)
    args = copy_args(checkpoint, args)
    model = {
        "model": load_model(checkpoint, args),
    }

    if args.gripper_pose is None or args.taskvar_token is None:
        raise ValueError()

    # load Mujoco environment
    instructions = load_instructions(args.instructions)
    if instructions is None:
        raise ValueError(args.instructions)

    device = torch.device(args.device)
    max_eps_dict = load_episodes()["max_episode_length"]

    robot = Robot(args.cameras, args.workspace, args.boundary)
    # Sanity check
    robot.render()

    env = Environment(
        model=model["model"],
        instructions=args.instructions,
        gripper_pose=args.gripper_pose,
        initial_height=args.initial_height,
        robot=robot,
        instr_mode=args.instr_mode,
    )

    success_rate = env.evaluate(
        args.task,
        max_episodes=max_eps_dict[args.task],
        variation=args.variation,
        num_demos=args.num_episodes,
        device=device,
        offset=args.offset,
        log_dir=log_dir / args.task if args.save_img else None,
        save_attn=args.attention,
        seed=args.seed,
    )

    print("Testing Success Rate {}: {:.04f}".format(args.task, success_rate))

    with FileLock(str(args.output.parent / f"{args.output.name}.lock")):
        with open(args.output, "a") as output_id:
            output_id.write(
                f"{args.task}-{args.variation}, {checkpoint}, seed={args.seed}, {success_rate}, {log_dir}\n"
            )

    if rospy.is_shutdown():
        # break
        return


if __name__ == "__main__":
    main()
