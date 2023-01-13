# From https://github.com/MohitShridhar/YARR/blob/peract/yarr/utils/video_utils.py
import os
import shutil
import cv2
import open3d
import einops
import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from rlbench import Environment
from rlbench.backend.observation import Observation


def get_point_cloud_images(vis, rgb_obs: np.array, pcd_obs: np.array):
    def get_point_cloud_image(opcds, vis):
        for opcd in opcds:
            vis.add_geometry(opcd)
            vis.update_geometry(opcd)
        vis.poll_events()
        vis.update_renderer()
        img = (np.array(vis.capture_screen_float_buffer()) * 255).astype(np.uint8)
        for opcd in opcds:
            vis.remove_geometry(opcd)
        return img

    opcds = []
    imgs = []

    for cam in range(rgb_obs.shape[0]):
        rgb = einops.rearrange(rgb_obs[cam, :3], "c h w -> (h w) c")
        pcd = einops.rearrange(pcd_obs[cam], "c h w -> (h w) c")
        opcd = open3d.geometry.PointCloud()
        opcd.points = open3d.utility.Vector3dVector(pcd.astype(np.float32))
        opcd.colors = open3d.utility.Vector3dVector(rgb.astype(np.uint8))
        opcds.append(opcd)
        imgs.append(get_point_cloud_image([opcd], vis))

    imgs.append(get_point_cloud_image(opcds, vis))
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
                 fps=30, obs_record_freq=10):
        self._env = env
        self._cam_motion = cam_motion
        self._fps = fps
        self._obs_record_freq = obs_record_freq
        self._3d_person_snaps = []
        self._obs_cameras = obs_cameras
        self._pcd_views = [*self._obs_cameras, "aggregate"]
        self._pcd_snaps = [[] for _ in range(len(self._pcd_views))]
        self._rgb_snaps = [[] for _ in range(len(self._obs_cameras))]
        self._pcd_vis = open3d.visualization.Visualizer()
        self._pcd_vis.create_window()

    def take_snap(self, obs: Observation):
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
            # TODO Debug point cloud snaps
            # rgb_obs = einops.rearrange(rgb_obs, "n_cam h w c -> n_cam c h w")
            # pcd_obs = einops.rearrange(pcd_obs, "n_cam h w c -> n_cam c h w")
            # pcd_imgs = get_point_cloud_images(self._pcd_vis, rgb_obs, pcd_obs)
            # for i in range(len(self._pcd_snaps)):
            #     self._pcd_snaps[i].append(pcd_imgs[i])

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
        # TODO Debug point cloud snaps
        # for (view, snaps) in zip(self._pcd_views, self._pcd_snaps):
        #     if len(snaps) == 0:
        #         continue
        #     image_size = snaps[0].shape[:2]
        #     video = cv2.VideoWriter(
        #         f"{path}/{view}_pcd_obs.mp4",
        #         cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
        #         self._fps // self._obs_record_freq,
        #         tuple(image_size)
        #     )
        #     for i, snap in enumerate(snaps):
        #         # cv2.imwrite(f"{path}/{view}_pcd_obs_snap{i}.png", snap)  # DEBUG
        #         video.write(snap)
        #     video.release()
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
