# Start from https://github.com/MohitShridhar/YARR/blob/peract/yarr/utils/video_utils.py
import os
import shutil
import cv2
import torch
import open3d
import einops
import numpy as np
from typing import List
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from rlbench import Environment
from rlbench.backend.observation import Observation


def get_point_cloud_images(vis: List[open3d.visualization.Visualizer],
                           rgb_obs: np.array, pcd_obs: np.array,
                           keyframe_actions: np.array,
                           custom_cam_params: bool):
    num_cams = rgb_obs.shape[0]
    assert len(vis) == (num_cams + 1)  # Last visualizer is for aggregate

    def get_point_cloud_image(opcds, vis, custom_cam_params):
        if custom_cam_params:
            ctr = vis.get_view_control()
            window_name = vis.get_window_name()
            param_orig = ctr.convert_to_pinhole_camera_parameters()
        for opcd in opcds:
            vis.add_geometry(opcd)
            vis.update_geometry(opcd)
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

    # Add gripper keyframe actions to point clouds for visualization
    keyframe_actions_opcd = open3d.geometry.PointCloud()
    keyframe_actions_opcd.points = open3d.utility.Vector3dVector(keyframe_actions)
    keyframe_actions_color = np.zeros_like(keyframe_actions)
    keyframe_actions_color[:, -1] = 1
    keyframe_actions_opcd.colors = open3d.utility.Vector3dVector(keyframe_actions_color)

    opcds = [keyframe_actions_opcd]
    imgs = []

    for cam in range(num_cams):
        rgb = einops.rearrange(rgb_obs[cam, :3], "c h w -> (h w) c")
        pcd = einops.rearrange(pcd_obs[cam], "c h w -> (h w) c")
        opcd = open3d.geometry.PointCloud()
        opcd.points = open3d.utility.Vector3dVector(pcd)
        opcd.colors = open3d.utility.Vector3dVector(rgb)
        opcds.append(opcd)
        imgs.append(get_point_cloud_image([opcd, keyframe_actions_opcd], vis[cam], custom_cam_params))

    imgs.append(get_point_cloud_image(opcds, vis[-1], custom_cam_params))
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
                 fps=30, obs_record_freq=10, custom_cam_params=False):
        """
        Arguments:
            obs_cameras: observation camera view points
            env: environment that generates observations to record
            cam_motion: motion for 3rd person camera recording
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
        self._keyframe_actions = []

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

    def take_snap(self, obs: Observation, keyframe: bool = False):
        if keyframe:
            self._keyframe_actions.append(obs.gripper_pose[:3])

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
            pcd_imgs = get_point_cloud_images(self._open3d_pcd_vis, rgb_obs, pcd_obs,
                                              np.stack(self._keyframe_actions), self._custom_cam_params)
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
        for image in self._3d_person_snaps:
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
            for snap in snaps:
                video.write(snap[:, :, ::-1])
            video.release()

        self._pcd_snaps = [[] for _ in range(len(self._pcd_views))]
        self._rgb_snaps = [[] for _ in range(len(self._obs_cameras))]
        self._keyframe_actions = []
