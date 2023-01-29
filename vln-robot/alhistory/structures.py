"""
This is a collection of structures with very little dependencies.
"""
from abc import abstractmethod, ABC
import math
from typing import Literal, Dict, Optional, List, Union, Tuple
from typing_extensions import TypedDict
from pydantic import BaseModel
from scipy.spatial.transform import Rotation as R
import torch
from torch.nn import functional as F
import numpy as np

PosMode = Literal["mse", "ce"]
RotMode = Literal["mse", "ce", "none"]
RotType = Literal["quat", "euler", "cont"]
ZMode = Literal["embed", "imgdec", "instr", "instr2"]
GripperPose = Literal["none", "token", "attn", "tokenattn"]
TransformerToken = Literal["tnhw", "tnc", "tnhw_cm", "tnhw_cm_sa"]
BackboneOp = Literal["sum", "cat", "max"]
InstructionMode = Literal["precompute", "text", "mic"]
PointCloudToken = Literal["none", "mean", "max", "median"]
CameraName = Literal["wrist", "left_shoulder", "right_shoulder"]

APPLY_CAMERAS = ("wrist", "left_shoulder", "right_shoulder")

Instructions = Dict[str, Dict[int, torch.Tensor]]
Workspace = Tuple[Tuple[float, float, float], Tuple[float, float, float]]


class Output(TypedDict):
    position: torch.Tensor
    rotation: torch.Tensor
    gripper: torch.Tensor
    attention: torch.Tensor
    task: Optional[torch.Tensor]


class Sample(TypedDict):
    frame_id: torch.Tensor
    task: Union[List[str], str]
    variation: Union[List[int], int]
    rgbs: torch.Tensor
    pcds: torch.Tensor
    action: torch.Tensor
    padding_mask: torch.Tensor
    instr: Optional[torch.Tensor]
    gripper: torch.Tensor
    taskvar: Optional[torch.Tensor]
    attn_indices: Optional[torch.Tensor]


class Misc(TypedDict):
    wrist_camera_extrinsics: Optional[np.ndarray]
    wrist_camera_intrinsics: Optional[np.ndarray]
    left_shoulder_camera_extrinsics: Optional[np.ndarray]
    left_shoulder_camera_intrinsics: Optional[np.ndarray]
    right_shoulder_camera_extrinsics: Optional[np.ndarray]
    right_shoulder_camera_intrinsics: Optional[np.ndarray]


class Observation(BaseModel):
    """
    This is similar to RLBench observation object
    """

    gripper_open: bool
    gripper_pose: np.ndarray
    # joint_velocities: np.ndarray
    misc: Misc
    loading: bool
    joint_velocities: Optional[np.ndarray] = None
    wrist_rgb: Optional[np.ndarray] = None
    wrist_pcd: Optional[np.ndarray] = None
    left_shoulder_rgb: Optional[np.ndarray] = None
    left_shoulder_pcd: Optional[np.ndarray] = None
    right_shoulder_rgb: Optional[np.ndarray] = None
    right_shoulder_pcd: Optional[np.ndarray] = None

    class Config:
        arbitrary_types_allowed = True


Demo = List[Observation]


class Position(ABC):
    def __init__(self, mode: PosMode, workspace: Workspace):
        self.mode = mode
        self.num_dims = 3
        self.resolution: int = 100
        self.default_quat: torch.Tensor = torch.Tensor([0, 1, 0, 0])
        self.workspace = torch.Tensor(workspace)
        self.size = self.workspace[1] - self.workspace[0]

    def compute_action(self, logit: torch.Tensor) -> torch.Tensor:
        """We force the rotation being normalized"""
        if self.mode == "mse":
            assert logit.shape[-1] == self.num_dims
            return logit
        elif self.mode == "ce":
            raise NotImplementedError()
        else:
            raise RuntimeError(f"Unexpected mode {self.mode}")

    def compute_metrics(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
        reduction: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        pred = self.compute_action(pred)
        acc = ((pred - true) ** 2).sum(1).sqrt() < 0.01
        acc = acc.to(pred.dtype)

        if reduction == "mean":
            acc = acc.mean()

        return {"position": acc}

    def compute_loss(self, pred: Output, sample: Sample, reduction: str) -> Dict[str, torch.Tensor]:
        device = pred["position"].device
        padding_mask = sample["padding_mask"].to(device)

        if self.mode == "mse":
            action = sample["action"].to(device)[padding_mask]
            logit = pred["position"]
            position = action[:, :3]
            return {"position": F.mse_loss(logit, position, reduction=reduction) * 3}

        elif self.mode == "ce":
            raise NotImplementedError()

        raise ValueError(f"Unexpected mode {self.mode}")


class Rotation(ABC):
    def __init__(self, mode: RotMode):
        self.mode = mode
        self.num_dims = -1
        self.resolution: int = 90
        self.default_quat: torch.Tensor = torch.Tensor([0, 1, 0, 0])

    def compute_action(self, logit: torch.Tensor) -> torch.Tensor:
        """We force the rotation being normalized"""
        if self.mode == "mse":
            assert logit.shape[-1] == self.num_dims
            return self._cont_to_quat_cont(logit)
        elif self.mode == "ce":
            return self._disc_to_quat_cont(logit.argmax(2))
        elif self.mode == "none":
            default = self.default_quat.type_as(logit)
            N = logit.dim()
            default = default.view(*([1] * (N - 1)), 4)
            b = logit.shape[0]
            return default.repeat(b, *([1] * (N - 1)))
        else:
            raise RuntimeError(f"Unexpected mode {self.mode}")

    def compute_metrics(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
        reduction: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        pred = self.compute_action(pred)
        acc = (pred - true).abs().max(1).values < 0.05
        acc = acc.to(pred.dtype)

        if reduction == "mean":
            acc = acc.mean()
        return {"rotation": acc}

    def compute_loss(
            self, pred: Output, sample: Sample, reduction: str = 'mean'
    ) -> Dict[str, torch.Tensor]:
        logit = pred["rotation"]
        device = logit.device
        padding_mask = sample["padding_mask"].to(device)
        rot = sample["action"].to(device)[padding_mask][:, 3:7]

        # losses.update(self.rot.compute_loss(pred["rotation"], action[:, 3:7]))
        rot_ = -rot.clone()

        if self.mode == "mse":
            rot_loss = self._compute_mse_loss(logit, rot, reduction)
            rot_loss_ = self._compute_mse_loss(logit, rot, reduction)
        elif self.mode == "ce":
            rot_loss = self._compute_ce_loss(logit, rot, reduction)
            rot_loss_ = self._compute_ce_loss(logit, rot_, reduction)
        elif self.mode == "none":
            return {}
        else:
            raise ValueError(f"Unexpected mode {self.mode}")

        # TODO What is this supposed to do apart from multiplying the loss by 4?
        losses = {}
        for (key, loss), (_, loss_) in zip(rot_loss.items(), rot_loss_.items()):
            select_mask = (loss < loss_).float()
            loss = 4 * (select_mask * loss + (1 - select_mask) * loss_)
            losses[key] = loss.mean()

        return losses

    @abstractmethod
    def _disc_to_quat_cont(self, rotation: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def _cont_to_quat_cont(self, rotation: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def _quat_cont_to_disc(self, rotation: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def _quat_cont_to_cont(self, quat: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def _compute_ce_loss(
            self, logit: torch.Tensor, true_cont: torch.Tensor, reduction: str
    ) -> Dict[str, torch.Tensor]:
        true_disc = self._quat_cont_to_disc(true_cont)
        loss = torch.tensor(0.0).type_as(logit)
        for i in range(self.num_dims):
            loss += F.cross_entropy(logit[:, i], true_disc[:, i], reduction=reduction)
        loss /= self.num_dims
        return {f"rotation": loss}

    def _compute_mse_loss(
        self, logit: torch.Tensor, true: torch.Tensor, reduction: str
    ) -> Dict[str, torch.Tensor]:
        dtype = logit.dtype
        true = self._quat_cont_to_cont(true).to(dtype)
        loss = F.mse_loss(logit, true, reduction=reduction).to(dtype)
        return {f"rotation": loss}


class QuatRotation(Rotation):
    """
    Helper function when using quaternion for rotation prediction
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.num_dims: int = 4

    def _disc_to_quat_cont(self, rotation: torch.Tensor):
        rot = rotation / self.resolution * 2 - 1
        norm = torch.linalg.norm(rot, dim=1, keepdim=True)
        rot /= norm

        # default rotation where the gripper is vertical
        nan = rot.isnan().any(-1)
        if nan.any():
            N = rotation.dim()
            default = torch.Tensor(self.default_quat).type_as(rot)
            default = default.view(*([1] * (N - 1)), 4)
            rot[nan] = default

        return rot

    def _quat_cont_to_disc(self, quat: torch.Tensor) -> torch.Tensor:
        discrete = (quat + 1) / 2 * self.resolution
        # this should be unlikely (numerical instabilities)
        discrete[discrete < 0] = 0.0
        discrete[discrete >= self.resolution] = self.resolution - 1
        return discrete.float().floor().long()

    def _cont_to_quat_cont(self, cont: torch.Tensor) -> torch.Tensor:
        return norm_tensor(cont)

    def _quat_cont_to_cont(self, quat: torch.Tensor) -> torch.Tensor:
        return norm_tensor(quat)


def norm_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / torch.linalg.norm(tensor, ord=2, dim=-1, keepdim=True)


class EulerRotation(Rotation):
    """
    Helper function when Euler representation

    Reference is XYZ
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.num_dims: int = 3
        self._ref = "XYZ"

    def _disc_to_quat_cont(self, rotation: torch.Tensor):
        # rot = ((rotation - 0.5) / self.resolution * 2 - 1) * math.pi
        rot = (rotation / self.resolution * 2 - 1) * math.pi
        quat = self._cont_to_quat_cont(rot)

        return quat

    def _quat_cont_to_disc(self, quat: torch.Tensor) -> torch.Tensor:
        # discrete = (quat + 1) / 2 * self.resolution
        euler = R.from_quat(quat.detach().cpu()).as_euler(self._ref)
        discrete = torch.from_numpy(euler).type_as(quat)
        discrete[discrete < -math.pi] += 2 * math.pi
        discrete[discrete > math.pi] -= 2 * math.pi
        discrete = (discrete + math.pi) / (2.0 * math.pi) * self.resolution

        # this should be unlikely (numerical instabilities)
        discrete[discrete < 0] = 0.0
        discrete[discrete >= self.resolution] = self.resolution - 1

        discrete = discrete.float().floor().long()

        return discrete

    def _cont_to_quat_cont(self, cont: torch.Tensor) -> torch.Tensor:
        np_quat = R.from_euler(self._ref, cont.detach().cpu()).as_quat()
        quat = torch.from_numpy(np_quat).type_as(cont)
        quat /= torch.linalg.norm(quat, dim=1, keepdim=True)

        # default rotation where the gripper is vertical
        nan = quat.isnan().any(-1)
        if nan.any():
            N = cont.dim()
            default = torch.Tensor(self.default_quat).type_as(cont)
            default = default.view(*([1] * (N - 1)), 4)
            quat[nan] = default

        return quat

    def _quat_cont_to_cont(self, quat: torch.Tensor) -> torch.Tensor:
        cont_np = R.from_quat(quat.cpu()).as_euler(self._ref)
        cont = torch.from_numpy(cont_np).type_as(quat)
        return cont


class ContRotation(Rotation):
    """
    Helper function when using quaternion for rotation prediction
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.num_dims: int = 6

    def _disc_to_quat_cont(self, rotation: torch.Tensor):
        rot = rotation * 2 / self.resolution - 1
        quat = self._cont_to_quat_cont(rot)
        return quat

    def _quat_cont_to_disc(self, quat: torch.Tensor) -> torch.Tensor:
        quat = quat.flatten(0, -2).view(-1, 4)
        so3 = compute_rotation_matrix_from_quaternions(quat)
        cont = so3[:, :2].flatten(-2).to(quat.dtype)

        discrete = (cont + 1) / 2 * self.resolution
        discrete[discrete < 0] = 0.0
        discrete[discrete >= self.resolution] = self.resolution - 1
        discrete = discrete.floor().long()
        return discrete

    def _cont_to_quat_cont(self, cont: torch.Tensor) -> torch.Tensor:
        a1 = cont[..., :3]
        a2 = cont[..., 3:]
        b1 = norm_tensor(a1)
        b2 = norm_tensor(a2 - (b1 * a2).sum(-1, keepdim=True) * b1)
        b3 = torch.cross(b1, b2, dim=-1)
        rot_matrix = torch.cat([b1, b2, b3], dim=-1).view(-1, 3, 3)
        r = R.from_matrix(rot_matrix.detach().cpu())
        quat = torch.from_numpy(r.as_quat())
        quat = quat.type_as(cont)

        # default rotation where the gripper is vertical
        nan = quat.isnan().any(-1)
        if nan.any():
            N = cont.dim()
            default = torch.Tensor(self.default_quat).type_as(cont)
            default = default.view(*([1] * (N - 1)), 4)
            quat[nan] = default

        return quat

    def _quat_cont_to_cont(self, quat: torch.Tensor) -> torch.Tensor:
        quat = quat.flatten(0, -2).view(-1, 4)
        so3 = compute_rotation_matrix_from_quaternions(quat)
        so3 = so3.view(-1, 3, 3)
        cont = so3[:, :2].flatten(-2).type_as(quat)
        return cont


def compute_rotation_matrix_from_quaternions(quats):
    """
    Copy from https://github.com/ylabbe/cosypose/blob/8e284acc39b13c7a172164ad0f482e3c1c83ca0d/cosypose/lib3d/rotations.py#L166
    Thanks Yann!
    """
    assert quats.shape[-1] == 4
    quats = quats / torch.norm(quats, p=2, dim=-1, keepdim=True)  # type: ignore
    mat = quat2mat(quats)[..., :3, :3]
    return mat


def quat2mat(quat):
    q_xyzw = quat
    q_wxyz = quat.clone()
    q_wxyz[..., 0] = q_xyzw[..., -1]
    q_wxyz[..., 1:] = q_xyzw[..., :-1]
    return angle_axis_to_rotation_matrix(quaternion_to_angle_axis(q_wxyz))


def angle_axis_to_rotation_matrix(angle_axis):
    """Convert 3d vector of axis-angle rotation to 4x4 rotation matrix

    Args:
        angle_axis (Tensor): tensor of 3d vector of axis-angle rotations.

    Returns:
        Tensor: tensor of 4x4 rotation matrices.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = tgm.angle_axis_to_rotation_matrix(input)  # Nx4x4
    """

    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1
        )
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = (
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    )
    return rotation_matrix  # Nx4x4


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(quaternion))
        )

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape Nx4 or 4. Got {}".format(quaternion.shape)
        )
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta),
    )

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis
