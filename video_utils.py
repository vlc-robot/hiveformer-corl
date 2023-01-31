# Start from https://github.com/MohitShridhar/YARR/blob/peract/yarr/utils/video_utils.py
import os
import shutil
import cv2
import torch
import json
import open3d
import einops
import numpy as np
import trimesh.transformations as tra
from typing import List, Optional
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from rlbench import Environment
from rlbench.backend.observation import Observation
from baseline.utils import sample_ghost_points


def get_gripper_control_points_open3d(grasp, show_sweep_volume=False, color=(0.2, 0.8, 0.)):
    """
    Open3D Visualization of parallel-jaw grasp.
    From https://github.com/adithyamurali/TaskGrasp/blob/master/visualize.py

    Arguments:
        grasp: [4, 4] np array
    """

    meshes = []
    align = tra.euler_matrix(np.pi / 2, -np.pi / 2, 0)

    # Cylinder 3,5,6
    cylinder_1 = open3d.geometry.TriangleMesh.create_cylinder(
        radius=0.005, height=0.139)
    transform = np.eye(4)
    transform[0, 3] = -0.03
    transform = np.matmul(align, transform)
    transform = np.matmul(grasp, transform)
    cylinder_1.paint_uniform_color(color)
    cylinder_1.transform(transform)

    # Cylinder 1 and 2
    cylinder_2 = open3d.geometry.TriangleMesh.create_cylinder(
        radius=0.005, height=0.07)
    transform = tra.euler_matrix(0, np.pi / 2, 0)
    transform[0, 3] = -0.065
    transform = np.matmul(align, transform)
    transform = np.matmul(grasp, transform)
    cylinder_2.paint_uniform_color(color)
    cylinder_2.transform(transform)

    # Cylinder 5,4
    cylinder_3 = open3d.geometry.TriangleMesh.create_cylinder(
        radius=0.005, height=0.06)
    transform = tra.euler_matrix(0, np.pi / 2, 0)
    transform[2, 3] = 0.065
    transform = np.matmul(align, transform)
    transform = np.matmul(grasp, transform)
    cylinder_3.paint_uniform_color(color)
    cylinder_3.transform(transform)

    # Cylinder 6, 7
    cylinder_4 = open3d.geometry.TriangleMesh.create_cylinder(
        radius=0.005, height=0.06)
    transform = tra.euler_matrix(0, np.pi / 2, 0)
    transform[2, 3] = -0.065
    transform = np.matmul(align, transform)
    transform = np.matmul(grasp, transform)
    cylinder_4.paint_uniform_color(color)
    cylinder_4.transform(transform)

    cylinder_1.compute_vertex_normals()
    cylinder_2.compute_vertex_normals()
    cylinder_3.compute_vertex_normals()
    cylinder_4.compute_vertex_normals()

    meshes.append(cylinder_1)
    meshes.append(cylinder_2)
    meshes.append(cylinder_3)
    meshes.append(cylinder_4)

    # Just for visualizing - sweep volume
    if show_sweep_volume:
        finger_sweep_volume = open3d.geometry.TriangleMesh.create_box(
            width=0.06, height=0.02, depth=0.14)
        transform = np.eye(4)
        transform[0, 3] = -0.06 / 2
        transform[1, 3] = -0.02 / 2
        transform[2, 3] = -0.14 / 2

        transform = np.matmul(align, transform)
        transform = np.matmul(grasp, transform)
        finger_sweep_volume.paint_uniform_color([0, 0.2, 0.8])
        finger_sweep_volume.transform(transform)
        finger_sweep_volume.compute_vertex_normals()

        meshes.append(finger_sweep_volume)

    return meshes


def get_point_cloud_images(vis: List[open3d.visualization.Visualizer],
                           rgb_obs: np.array, pcd_obs: np.array,
                           custom_cam_params: bool,
                           gt_keyframe_gripper_matrices: Optional[np.array] = None,
                           pred_keyframe_gripper_matrices: Optional[np.array] = None,
                           pred_location_heatmap: Optional[np.array] = None,
                           gripper_loc_bounds: Optional[np.array] = None):
    num_cams = rgb_obs.shape[0]
    assert len(vis) == (num_cams + 1)  # Last visualizer is for aggregate

    def plot_geometries(geometries, vis, custom_cam_params):
        if custom_cam_params:
            ctr = vis.get_view_control()
            window_name = vis.get_window_name()
            param_orig = ctr.convert_to_pinhole_camera_parameters()
        for geom in geometries:
            vis.add_geometry(geom)
            vis.update_geometry(geom)
        if custom_cam_params and window_name in ["left_shoulder", "right_shoulder"]:
            ctr.convert_from_pinhole_camera_parameters(param_orig)
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(do_render=True)
        img = np.fliplr(np.flipud((np.array(img) * 255).astype(np.uint8)[:, :, ::-1]))
        vis.clear_geometries()
        if custom_cam_params and window_name in ["left_shoulder", "right_shoulder"]:
            ctr.convert_from_pinhole_camera_parameters(param_orig)
        return img

    all_geometries = []
    imgs = []

    # Visualize sampled gripper ghost points
    if gripper_loc_bounds is not None:
        ghost_points = sample_ghost_points(gripper_loc_bounds)
        ghost_points_colors = np.zeros_like(ghost_points)
        ghost_points_colors[:, 0] = 0.2
        ghost_points_colors[:, 1] = 0.8
        ghost_points_opcd = open3d.geometry.PointCloud()
        ghost_points_opcd.points = open3d.utility.Vector3dVector(ghost_points)
        ghost_points_opcd.colors = open3d.utility.Vector3dVector(ghost_points_colors)
        all_geometries.append(ghost_points_opcd)

    # Visualize location prediction heatmap
    pred_location_heatmap_opcds = []
    if pred_location_heatmap is not None:
        pred_location_heatmap_opcd = open3d.geometry.PointCloud()
        pred_location_heatmap_opcd.points = open3d.utility.Vector3dVector(pred_location_heatmap)
        pred_location_heatmap_colors = np.zeros_like(pred_location_heatmap)
        pred_location_heatmap_colors[:, 0] = 1.0
        pred_location_heatmap_colors[:, 1] = 0.0
        pred_location_heatmap_colors[:, 2] = 1.0
        pred_location_heatmap_opcd.colors = open3d.utility.Vector3dVector(pred_location_heatmap_colors)
        pred_location_heatmap_opcds.append(pred_location_heatmap_opcd)
        all_geometries.append(pred_location_heatmap_opcd)

    # Add gripper keyframe actions to point clouds for visualization
    keyframe_action_geometries = []
    if gt_keyframe_gripper_matrices is not None:
        for grasp in gt_keyframe_gripper_matrices:
            keyframe_action_geometries += get_gripper_control_points_open3d(grasp, color=(0.2, 0.8, 0.))
    if pred_keyframe_gripper_matrices is not None:
        for grasp in pred_keyframe_gripper_matrices:
            keyframe_action_geometries += get_gripper_control_points_open3d(grasp, color=(1.0, 0.2, 1.0))
    all_geometries += keyframe_action_geometries

    for cam in range(num_cams):
        rgb = einops.rearrange(rgb_obs[cam, :3], "c h w -> (h w) c")
        pcd = einops.rearrange(pcd_obs[cam], "c h w -> (h w) c")
        opcd = open3d.geometry.PointCloud()
        opcd.points = open3d.utility.Vector3dVector(pcd)
        opcd.colors = open3d.utility.Vector3dVector(rgb)
        all_geometries.append(opcd)
        view_geometries = ([opcd, ghost_points_opcd, *pred_location_heatmap_opcds, *keyframe_action_geometries]
                           if vis[cam].get_window_name() != "wrist" else [opcd])
        imgs.append(plot_geometries(view_geometries, vis[cam], custom_cam_params))

    imgs.append(plot_geometries(all_geometries, vis[-1], custom_cam_params))
    return imgs


class CameraMotion(object):
    def __init__(self, cam: VisionSensor):
        self.cam = cam

    def step(self):
        raise NotImplementedError()

    def save_pose(self):
        self._prev_pose = self.cam.get_pose()

    def restore_pose(self):
        self.cam.set_pose(self._prev_pose)


class CircleCameraMotion(CameraMotion):

    def __init__(self, cam: VisionSensor, origin: Dummy,
                 speed: float, init_rotation: float = np.deg2rad(180)):
        super().__init__(cam)
        self.origin = origin
        self.speed = speed  # in radians
        self.origin.rotate([0, 0, init_rotation])

    def step(self):
        self.origin.rotate([0, 0, self.speed])


class TaskRecorder(object):

    def __init__(self, obs_cameras, env: Environment, cam_motion: CameraMotion,
                 task_str: str, fps=30, obs_record_freq=1, custom_cam_params=False):
        """
        Arguments:
            obs_cameras: observation camera view points
            env: environment that generates observations to record
            cam_motion: motion for 3rd person camera recording
            task_str: task string
            fps: frames per second
            obs_record_freq: frequency of first-person observation recording
            custom_cam_params: if True, record point cloud observations with custom camera
             params instead of default top-down view
        """
        self._env = env
        self._cam_motion = cam_motion
        self._fps = fps
        self._obs_record_freq = obs_record_freq
        self._custom_cam_params = custom_cam_params
        self._3d_person_snaps = []
        self._obs_cameras = obs_cameras
        self._pcd_views = [*self._obs_cameras, "aggregate"]
        self._pcd_snaps = [[] for _ in range(len(self._pcd_views))]
        self._rgb_snaps = [[] for _ in range(len(self._obs_cameras))]
        self._gt_keyframe_gripper_matrices = None
        self._pred_keyframe_gripper_matrices = None
        self._pred_location_heatmap = None
        self._gripper_loc_bounds = json.load(open("location_bounds.json", "r"))[task_str]

        def get_extrinsic(sensor: VisionSensor) -> np.array:
            # Note: The extrinsic and intrinsic matrices are in the observation,
            # no need to compute them here
            pose = sensor.get_pose()
            position, rot_quaternion = pose[:3], pose[3:]
            rot_matrix = open3d.geometry.get_rotation_matrix_from_quaternion(
                np.array((rot_quaternion[3], rot_quaternion[0], rot_quaternion[1], rot_quaternion[2]))
            )
            extrinsic = np.eye(4)
            rot_matrix = rot_matrix.T
            position = - rot_matrix @ position
            extrinsic[:3, :3] = rot_matrix
            extrinsic[:3, 3] = position
            return extrinsic

        # Create Open3D point cloud visualizers
        self._open3d_pcd_vis = []
        assert len(self._pcd_views) <= 4
        for i, view in enumerate(self._pcd_views):
            if i == 0:
                left, top = 0, 0
            elif i == 1:
                left, top = 640, 0
            elif i == 2:
                left, top = 0, 480
            elif i == 3:
                left, top = 640, 480

            vis = open3d.visualization.Visualizer()
            vis.create_window(window_name=view, width=640, height=480, left=left, top=top)
            self._open3d_pcd_vis.append(vis)

            if self._custom_cam_params:
                ctr = vis.get_view_control()
                param = ctr.convert_to_pinhole_camera_parameters()

                if view == "left_shoulder":
                    sensor = VisionSensor("cam_over_shoulder_left")
                    param.extrinsic = get_extrinsic(sensor)
                    ctr.convert_from_pinhole_camera_parameters(param)
                elif view == "right_shoulder":
                    sensor = VisionSensor("cam_over_shoulder_right")
                    param.extrinsic = get_extrinsic(sensor)
                    ctr.convert_from_pinhole_camera_parameters(param)

    def take_snap(self,
                  obs: Observation,
                  gt_keyframe_gripper_matrices: Optional[np.array] = None,
                  pred_keyframe_gripper_matrices: Optional[np.array] = None,
                  pred_location_heatmap: Optional[np.array] = None):
        """
        Take observation snapshot.

        Arguments:
            obs: observations to use in snapshot
            gt_keyframe_gripper_matrices: if not None, ground-truth keyframe gripper poses
             to record and display in all snapshots until save() is called
            pred_keyframe_gripper_matrices: if not None, predicted keyframe gripper poses
             to record and display in all snapshots until save() is called
        """
        if gt_keyframe_gripper_matrices is not None:
            self._gt_keyframe_gripper_matrices = gt_keyframe_gripper_matrices
        if pred_keyframe_gripper_matrices is not None:
            self._pred_keyframe_gripper_matrices = pred_keyframe_gripper_matrices
        if pred_location_heatmap is not None:
            self._pred_location_heatmap = pred_location_heatmap

        # Third-person snap
        self._cam_motion.step()
        self._3d_person_snaps.append(
            (self._cam_motion.cam.capture_rgb() * 255.).astype(np.uint8))

        # Obs point cloud and RGB snaps
        if len(self._3d_person_snaps) % self._obs_record_freq == 0:
            rgb_obs = np.stack([getattr(obs, f"{cam}_rgb") for cam in self._obs_cameras])
            pcd_obs = np.stack([getattr(obs, f"{cam}_point_cloud") for cam in self._obs_cameras])
            for i in range(len(self._rgb_snaps)):
                self._rgb_snaps[i].append(rgb_obs[i])
            rgb_obs = einops.rearrange(rgb_obs, "n_cam h w c -> n_cam c h w")
            # normalise to [-1, 1]
            rgb_obs = rgb_obs / 255.0
            rgb_obs = 2 * (rgb_obs - 0.5)
            pcd_obs = einops.rearrange(pcd_obs, "n_cam h w c -> n_cam c h w")
            pcd_imgs = get_point_cloud_images(
                self._open3d_pcd_vis, rgb_obs, pcd_obs,
                self._custom_cam_params,
                self._gt_keyframe_gripper_matrices,
                self._pred_keyframe_gripper_matrices,
                self._pred_location_heatmap,
                self._gripper_loc_bounds
            )
            for i in range(len(self._pcd_snaps)):
                self._pcd_snaps[i].append(pcd_imgs[i])

    def save(self, path, lang_goal):
        print(f"Saving eval video at {path}")
        os.makedirs(path, exist_ok=True)

        # OpenCV QT version can conflict with PyRep, so import here
        import cv2

        # Third-person video
        image_size = self._cam_motion.cam.get_resolution()
        video = cv2.VideoWriter(
            f"{path}/3rd_person.mp4",
            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
            self._fps,
            tuple(image_size)
        )
        for i, image in enumerate(self._3d_person_snaps):
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = (0.45 * image_size[0]) / 640
            font_thickness = 2
            lang_textsize = cv2.getTextSize(lang_goal, font, font_scale, font_thickness)[0]
            lang_textX = (image_size[0] - lang_textsize[0]) // 2
            frame = cv2.putText(frame, lang_goal, org=(lang_textX, image_size[1] - 35),
                                fontScale=font_scale, fontFace=font, color=(0, 0, 0),
                                thickness=font_thickness, lineType=cv2.LINE_AA)
            video.write(frame)
        video.release()
        self._3d_person_snaps = []

        # Obs point cloud and RGB videos
        for (view, snaps) in zip(self._pcd_views, self._pcd_snaps):
            if len(snaps) == 0:
                continue
            image_size = (640, 480)
            video = cv2.VideoWriter(
                f"{path}/{view}_pcd_obs.mp4",
                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                self._fps // self._obs_record_freq,
                tuple(image_size)
            )
            for i, snap in enumerate(snaps):
                video.write(cv2.resize(snap, image_size))
            video.release()
        for (view, snaps) in zip(self._obs_cameras, self._rgb_snaps):
            if len(snaps) == 0:
                continue
            image_size = snaps[0].shape[:2]
            video = cv2.VideoWriter(
                f"{path}/{view}_rgb_obs.mp4",
                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                self._fps // self._obs_record_freq,
                tuple(image_size)
            )
            for i, snap in enumerate(snaps):
                video.write(snap[:, :, ::-1])
            video.release()

        self._pcd_snaps = [[] for _ in range(len(self._pcd_views))]
        self._rgb_snaps = [[] for _ in range(len(self._obs_cameras))]
        self._gt_keyframe_gripper_matrices = None
        self._pred_keyframe_gripper_matrices = None
        self._pred_location_heatmap = None
