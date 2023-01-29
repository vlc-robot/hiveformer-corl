import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Literal, Union, Dict, NamedTuple
import math
import numpy as np
import torch
from PIL import Image
import rospy
import tf
import tf2_ros
import moveit_commander
import moveit_msgs
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState, Image
from prl_ur5_demos.utils import make_pose

RotMode = Literal["mse", "ce", "none"]
RotType = Literal["quat", "euler", "cont"]
Instructions = Dict[str, Dict[int, torch.Tensor]]
ZMode = Literal["embed", "imgdec", "instr", "instr2"]
GripperPose = Literal["none", "token", "attn", "tokenattn"]
TransformerToken = Literal["tnhw", "tnc", "tnhw_cm", "tnhw_cm_sa"]
BackboneOp = Literal["sum", "cat", "max"]
InstructionMode = Literal["precompute", "text", "mic"]

Vector3 = Tuple[float, float, float]
Vector4 = Tuple[float, float, float, float]
CameraMode = Literal["rgb", "depth"]
CLOSE = False
OPEN = True


@dataclass
class Item:
    desc: Any
    position: Vector3
    quaternion: Vector4


@dataclass
class GripperState:
    """
    Convert Euler --> Quaternion using ROS convention:
    > qt = tf.transformations.quaternion_from_euler(*orientation)
    """

    position: Vector3
    quaternion: Vector4
    state: bool


Trajectory = List[GripperState]


def clip_position(position: Vector3, workspace: np.ndarray) -> Vector3:
    np_position = np.array(position)
    np_position = np.clip(np_position, workspace[0], workspace[1])
    return np_position.tolist()


class Camera:
    def __init__(self, topic):
        self._topic = topic

    def record_image(self, timeout=4.0, dtype: np.typing.NBitBase = np.uint8):  # type: ignore
        """Return next received image as numpy array in specified encoding.
        @param timeout: time in seconds
        """
        msg = rospy.wait_for_message(self._topic, Image, timeout=timeout)

        data = np.frombuffer(msg.data, dtype=dtype)
        data = data.reshape((msg.height, msg.width, -1))

        return data


@dataclass
class ImageMessage:
    """
    Typing for image message
    """

    data: np.ndarray
    height: int
    width: int


class CameraAsync:
    def __init__(self, topic):
        self._topic = topic
        self._im_msg: Optional[ImageMessage] = None
        self._sub = rospy.Subscriber(
            topic, Image, self.save_last_image, queue_size=1, buff_size=2 ** 24
        )

        deadline = rospy.Time.now() + rospy.Duration(1.0)
        while not rospy.core.is_shutdown() and self._im_msg is None:
            if rospy.Time.now() > deadline:
                rospy.logwarn_throttle(1.0, "Waiting for an image ({})...".format(topic))
            rospy.rostime.wallsleep(0.01)

        if rospy.core.is_shutdown():
            raise rospy.exceptions.ROSInterruptException("rospy shutdown")

    def save_last_image(self, msg: ImageMessage):
        self._im_msg = msg

    def record_image(self, dtype: np.typing.NBitBase = np.uint8):  # type: ignore
        """Return next received image as numpy array in specified encoding.
        @param timeout: time in seconds
        """
        if self._im_msg is None:
            raise RuntimeError("Camera async is not properly initialized")
        data = np.frombuffer(self._im_msg.data, dtype=dtype)
        data = data.reshape((self._im_msg.height, self._im_msg.width, -1))
        return data


def init_cameras(
    cameras: List[str], async_obs: bool, mode: CameraMode
) -> Dict[str, Union[CameraAsync, Camera]]:
    if mode == "rgb":
        topic = "color/image_raw"
    elif mode == "depth":
        topic = "aligned_depth_to_color/image_raw"
    else:
        raise ValueError(f"Unexpected {mode}")

    CameraClass = CameraAsync if async_obs else Camera

    camera_recorders = {name: CameraClass(f"/{name}_camera/{topic}") for name in cameras}

    return camera_recorders


def initialize_moveit(
    eef_frame: str = "left_gripper_grasp_frame",
    max_velocity_scaling_factor: float = 0.2,
    max_acceleration_scaling_factor: float = 0.2,
    planning_time: float = 2.0,
):
    # Create ros node
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("collect_real", anonymous=True)

    # Initialize the robot
    commander = moveit_commander.RobotCommander()

    # Configure the planning pipeline
    commander.left_arm.set_max_velocity_scaling_factor(max_velocity_scaling_factor)
    commander.left_arm.set_max_acceleration_scaling_factor(
        max_acceleration_scaling_factor
    )
    commander.left_arm.set_planning_time(planning_time)
    commander.left_arm.set_planner_id("RRTstar")

    # Set eef link
    commander.left_arm.set_end_effector_link(eef_frame)

    return commander


@dataclass
class TFTranslation:
    x: float
    y: float
    z: float


@dataclass
class TFRotation:
    x: float
    y: float
    z: float
    w: float


@dataclass
class TFTransform:
    translation: TFTranslation
    rotation: TFRotation


class TFRecorder:
    def __init__(self, source_frame, target_frame):
        self._source_frame = source_frame
        self._target_frame = target_frame

        self.tf_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf_buffer)

    def record_tf(self, timeout=4.0, now=False) -> TFTransform:
        now = rospy.Time.now() if now else rospy.Time(0)
        transform = self.tf_buffer.lookup_transform(
            self._source_frame, self._target_frame, now, rospy.Duration(timeout)
        )

        return transform.transform


class DisplayTrajectoryPublisher:
    """
    Publish a DisplayTrajectory that can be visualized on rviz.
    """

    def __init__(self):
        self._publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

    def publish(self, plan, current_state):
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = current_state
        display_trajectory.trajectory.append(plan)
        # Publish
        self._publisher.publish(display_trajectory)


def _check_position_in_scene(
    position: Union[Vector3, np.ndarray], boundary: np.ndarray
) -> None:
    if not isinstance(position, np.ndarray):
        position = np.asarray(position)

    if np.any((position < boundary[0]) | (position > boundary[0])):
        raise ValueError(f"The position {position.tolist()} is not allowed!")


class Robot:
    def __init__(
        self,
        cameras: List[str],
        workspace: np.ndarray,
        boundary: np.ndarray,
        command_ros_topic: str = "/left_arm/scaled_pos_joint_traj_controller/command",
        robot_base_frame: str = "prl_ur5_base",
        eef_frame: str = "left_gripper_grasp_frame",
        async_obs: bool = False,
        depth: bool = True,
        cartesian: bool = True,
        eef_steps: float = 0.01,
        jump_threshold: float = 0.0,
        debug: bool = False,
    ):
        self._cartesian = cartesian
        self.workspace = workspace
        self.boundary = boundary
        self._eef_steps = eef_steps
        self._eef_frame = eef_frame
        self._jump_threshold = jump_threshold
        self._robot_base_frame = robot_base_frame
        self._debug = debug

        self._commander = initialize_moveit(eef_frame=self._eef_frame)

        # Cameras configuration
        bravo_info = {
            "pos": np.array([-0.494061, 0.692729, 0.400215]),
            "euler": np.array([1.03091702, 0.00556305, -3.11407431]),
            "fovy": 42.5,
        }
        charlie_info = {
            "pos": np.array([-1.201099, 0.005, 0.403127]),
            "euler": np.array([1.04368278, -0.00250582, -1.56810664]),
            "fovy": 42.5,
        }
        self._cam_info = {"bravo_camera": bravo_info, "charlie_camera": charlie_info}

        # Trajectory Publisher
        self._traj_publisher = rospy.Publisher(
            command_ros_topic,
            JointTrajectory,
            queue_size=1,
        )
        self._display_traj_publisher = DisplayTrajectoryPublisher()

        # Transformations
        self.tf_listener = tf.TransformListener()

        # Gripper record position
        self._eef_tf_recorder = TFRecorder(robot_base_frame, eef_frame)
        # Cameras
        self._rgb_cameras = init_cameras(cameras, async_obs, "rgb")
        self._depth_cameras = init_cameras(cameras if depth else [], async_obs, "depth")

        # Grasped flag
        self._open = CLOSE
        self.set_openness(OPEN)

    @property
    def eef_pose(self):
        eef_tf = self._eef_tf_recorder.record_tf()
        position = (
            eef_tf.translation.x,
            eef_tf.translation.y,
            eef_tf.translation.z,
        )
        rotation = (
            eef_tf.rotation.x,
            eef_tf.rotation.y,
            eef_tf.rotation.z,
            eef_tf.rotation.w,
        )
        return GripperState(position, rotation, self._open)

    def set_pose(self, position: Vector3, quaternion: Vector4, wait: bool = True):
        """
        Move the gripper to its internal variable
        """
        _check_position_in_scene(position, self.boundary)

        pose_stamped = make_pose(position, quaternion, frame_id=self._robot_base_frame)
        pose = pose_stamped.pose

        if self._cartesian:
            return self._set_cartesian_path(pose, wait)

        return self._set_planned_path(pose_stamped, wait)

    def _set_planned_path(self, pose_stamped, wait: bool):
        self._commander.left_arm.set_pose_target(pose_stamped)
        return self._commander.left_arm.go(wait=wait)

    def _set_cartesian_path(self, pose, wait: bool):
        # current_pose = make_pose(self.eef_pose.position, self.eef_pose.rotation, frame_id=self._robot_base_frame).pose
        # trajectory = [current_pose, pose]
        trajectory = [pose]

        left_path, left_fraction = self._commander.left_arm.compute_cartesian_path(
            trajectory,
            eef_step=self._eef_steps,
            jump_threshold=self._jump_threshold,
        )
        if left_fraction < 1.0:
            rospy.logerr(f"Failed to plan cartesian path. Fraction Path: {left_fraction}")
            return False

        if self._debug:
            current_state = self._commander.get_current_state()
            self._display_traj_publisher.publish(left_path, current_state)
            ask = input("Should we proceed? (Y/N)")
            if ask not in ("Y", "y", "yes"):
                rospy.logerr("Stopping the execution of a plan")
                return

        return self._commander.left_arm.execute(left_path, wait=wait)

    def set_openness(self, is_open: bool, wait: bool = True) -> bool:
        # It is already at the right state
        if self._open == is_open:
            return True

        state = "open" if is_open else "close"
        self._commander.left_gripper.set_named_target(state)
        self._open = is_open

        return self.go(wait=wait)

    def go(self, wait: bool):
        success = self._commander.left_gripper.go(wait=wait)

        if success:
            return True

        rospy.logerr("Can't set gripper. Trying again...")
        return self._commander.left_gripper.go(wait=wait)

    def render(self):
        obs = {}

        for name, cam in self._rgb_cameras.items():
            obs[f"rgb_{name}"] = cam.record_image(dtype=np.uint8)
            obs[f"info_{name}"] = self._cam_info[f"{name}_camera"]

        for name, cam in self._depth_cameras.items():
            img = cam.record_image(dtype=np.uint16).astype(np.float32)
            obs[f"depth_{name}"] = img.squeeze(-1) / 1000
            obs[f"info_{name}"] = self._cam_info[f"{name}_camera"]

        obs.update(self.get_gripper_pose())

        return obs

    def get_gripper_pose(self) -> Dict[str, np.ndarray]:
        eef_tf = self._eef_tf_recorder.record_tf()
        return {
            "gripper_pos": np.array(
                [
                    eef_tf.translation.x,
                    eef_tf.translation.y,
                    eef_tf.translation.z,
                ]
            ),
            "gripper_quat": np.array(
                [
                    eef_tf.rotation.x,
                    eef_tf.rotation.y,
                    eef_tf.rotation.z,
                    eef_tf.rotation.w,
                ]
            ),
            "gripper_state": np.ndarray([self._open == OPEN]),
        }

    def execute(self, state: GripperState) -> bool:
        success = self.set_pose(state.position, state.quaternion)
        if not success:
            return False

        success = self.set_openness(state.state)
        return success


def center_item_position(item: Item) -> Trajectory:
    # Center Y-axis
    quat1 = tf.transformations.quaternion_from_euler(math.pi, 0, math.pi / 2)
    gs1 = GripperState(item.position, quat1, CLOSE)
    gs2 = GripperState(item.position, quat1, OPEN)

    # Center X-axis
    quat2 = tf.transformations.quaternion_from_euler(math.pi, 0, 0)
    gs3 = GripperState(item.position, quat2, CLOSE)
    gs4 = GripperState(item.position, quat2, OPEN)

    pos = np.array(item.position)
    above_pos = (pos + np.array([0, 0, 0.1])).tolist()
    gs5 = GripperState(above_pos, quat2, OPEN)

    return [gs1, gs2, gs3, gs4, gs5]
    # return [gs3, gs4, gs5]


def reset_gripper(
    workspace: np.ndarray,
    random_state: np.random.RandomState,
) -> Trajectory:
    """
    Reset the gripper to any location on the workspace

    >>> workspace = np.ndarray([[-1, -1, -1], [1,1,1]])
    >>> state, = reset_gripper(workspace, np.random)
    >>> assert _check_position_in_scene(state, workspace), state
    """
    position: Vector3 = random_state.uniform(workspace[0], workspace[1])  # type: ignore
    quaternion = tf.transformations.quaternion_from_euler(math.pi, 0, math.pi / 2)
    state = GripperState(position, quaternion, OPEN)
    return [state]


def sequential_execution(states: Trajectory, robot: Robot) -> bool:
    for state in states:
        success = robot.execute(state)
        if not success:
            return False
    return True


def pick(item: Item) -> Trajectory:
    """
    Create a trajectory to pick an object

    >>> item = Item('', [0,0,0], [0,0,0,1])
    >>> traj = pick(item)
    >>> assert traj[0].position == np.ndarray([0,0,0.1])
    """
    pos = np.array(item.position)
    above_pos = (pos + np.array([0, 0, 0.1])).tolist()

    return [
        GripperState(above_pos, item.quaternion, True),
        GripperState(item.position, item.quaternion, False),
        GripperState(above_pos, item.quaternion, False),
    ]


def place(item: Item) -> Trajectory:
    """
    Create a trajectory to place an object

    >>> item = Item('', [0,0,0], [0,0,0,1])
    >>> traj = pick(item)
    >>> assert traj[0].position == np.ndarray([0,0,0.1])
    """
    pos = np.array(item.position)
    above_pos = (pos + np.array([0, 0, 0.1])).tolist()

    return [
        GripperState(above_pos, item.quaternion, False),
        GripperState(item.position, item.quaternion, True),
        GripperState(above_pos, item.quaternion, True),
    ]
