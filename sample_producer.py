import sys
import os
import random
import math
from typing import List, Tuple, Any, Union
import pickle
from configparser import ConfigParser
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from tqdm import tqdm
from argparse import ArgumentParser
from utils import str2bool, read_settings, connect_and_draw_points, cotan, apply_radial_dist

FIXED_SEED_NUM = 35  # Seed number
GREEN_COLOR = (0, 255, 0)  # Green color code
ORANGE_COLOR = (255, 200, 0)  # Orange color code
RED_COLOR = (255, 0, 0)  # Red color code
WHITE_COLOR = (255, 255, 255)  # White color code
HALF_PIXEL_SIZE = 1 / 2
CAM_REGION_EXPAND_RATIO = 3.0  # Expanding ratio of original image. Odd integer number is advised.


def calculate_input_coord(bbox: List[Tuple[float]], proj_mid: List[np.ndarray]) -> Tuple[Union[float, Any], ndarray, Any]:
    """
    Calculates input and target data points to be used in the model.

    Parameters
    ----------
    bbox: List[Tuple[float]]
        Coordinates of the bounding box in the image.
    proj_mid: List[np.ndarray]
        Coordinates of the projection point in the image.

    Returns
    ----------
    input_coords: Tuple[Union[float, Any], ndarray, Any]
        Input coordinates to be fed to the projection point estimator.
    """
    bbox_bottom_center = ((np.array(bbox[2]) + np.array(bbox[3])) / 2)
    bbox_width_height = np.array((bbox[1][0] - bbox[0][0], bbox[2][1] - bbox[1][1]))
    proj_wrt_bbox_bottom_center = proj_mid[0] - bbox_bottom_center
    input_coords = bbox_bottom_center, bbox_width_height, proj_wrt_bbox_bottom_center
    return input_coords


class PointEstimatorProjection:
    """
    Point estimator projection class
    """
    def __init__(self, config: ConfigParser):
        """
        Initialize the Point estimator projection class.

        Parameters
        ----------
        config: ConfigParser
            Configuration object
        """
        self.initialize_config(config)
        if self.fixed_seed:
            os.environ['PYTHONHASHSEED'] = str(FIXED_SEED_NUM)
            random.seed(FIXED_SEED_NUM)
            np.random.seed(FIXED_SEED_NUM)

        self.cx = self.cam_pixel_width / 2  # horizontal center pixel loc.
        self.cy = self.cam_pixel_height / 2  # vertical center pixel loc.

        self.extended_height = int(CAM_REGION_EXPAND_RATIO * self.cam_pixel_height)
        self.extended_width = int(CAM_REGION_EXPAND_RATIO * self.cam_pixel_width)
        self.extended_y_start = -int((CAM_REGION_EXPAND_RATIO - 1) * self.cam_pixel_height / 2)
        self.extended_x_start = -int((CAM_REGION_EXPAND_RATIO - 1) * self.cam_pixel_width / 2)
        self.top_left_pixel_coord = np.array([self.extended_x_start, self.extended_y_start])
        self.pixel_world_coords = np.ones((self.extended_height, self.extended_width, 3), dtype=float) * sys.float_info.max

        self.calc_projection_matrix()
        if self.export:
            self.calc_pixel_world_coordinates()

    def initialize_config(self, config: ConfigParser):
        """
        Initialize and set the config fields.

        Parameters
        ----------
        config: ConfigParser
            Configuration object
        """
        self.fixed_seed = str2bool(config.get("sample_producer", "FIXED_SEED"))
        self.cam_pixel_width = int(float(config.get("sample_producer", "CAM_PIXEL_WIDTH")))
        self.cam_pixel_height = int(float(config.get("sample_producer", "CAM_PIXEL_HEIGHT")))
        self.cam_fov_hor = float(config.get("sample_producer", "CAM_FOV_HOR"))
        self.yaw_cam = float(config.get("sample_producer", "YAW_CAM"))
        self.pitch_cam = float(config.get("sample_producer", "PITCH_CAM"))
        self.roll_cam = float(config.get("sample_producer", "ROLL_CAM"))
        self.cam_x = float(config.get("sample_producer", "CAM_X"))
        self.cam_y = float(config.get("sample_producer", "CAM_Y"))
        self.cam_z = float(config.get("sample_producer", "CAM_Z"))
        self.obj_height = float(config.get("sample_producer", "OBJ_HEIGHT"))
        self.obj_width = float(config.get("sample_producer", "OBJ_WIDTH"))
        self.obj_length = float(config.get("sample_producer", "OBJ_LENGTH"))
        self.radial_dist_enabled = str2bool(config.get("sample_producer", "RADIAL_DIST_ENABLED"))
        self.random_deviation_enabled = str2bool(config.get("sample_producer", "RANDOM_DEVIATION_ENABLED"))
        self.deviation_sigma = float(config.get("sample_producer", "DEVIATON_SIGMA"))
        self.k_1 = float(config.get("sample_producer", "K_1"))
        self.k_2 = float(config.get("sample_producer", "K_2"))
        self.export = str2bool(config.get("sample_producer", "EXPORT"))

    def calc_camera_calibration_matrix(self) -> np.ndarray:
        """
        Calculate camera calibration matrix K. It is assumed that our pixels are square shaped and no skewed image
        center.

        Returns
        ----------
        K: np.ndarray
            Camera calibration matrix K
        """
        focal_length = cotan(math.radians(self.cam_fov_hor / 2)) * self.cx
        # Cameras intrinsic matrix [K]
        K = np.array(
            [[focal_length, 0., self.cx],
             [0., focal_length, self.cy],
             [0., 0., 1.]]
        )
        return K

    def get_rot_x(self, angle: float):
        """
        Rotation matrix around X-axis

        Parameters
        ----------
        angle: float
            Camera rotation angle around X-axis in radians

        Returns
        ----------
        R_x: np.ndarray
            Rotation matrix of camera around X-axis
        """

        R_x = np.zeros((3, 3))
        cos_ang = np.cos(angle)
        sin_ang = np.sin(angle)

        R_x[0, 0] = 1
        R_x[1, 1] = cos_ang
        R_x[1, 2] = -sin_ang
        R_x[2, 1] = sin_ang
        R_x[2, 2] = cos_ang
        return R_x

    def get_rot_y(self, angle: float):
        """
        Rotation matrix around Y-axis

        Parameters
        ----------
        angle: float
            Camera rotation angle around Y-axis in radians

        Returns
        ----------
        R_y: np.ndarray
            Rotation matrix of camera around Y-axis

        """
        R_y = np.zeros((3, 3))
        cos_ang = np.cos(angle)
        sin_ang = np.sin(angle)

        R_y[0, 0] = cos_ang
        R_y[0, 2] = -sin_ang
        R_y[1, 1] = 1
        R_y[2, 0] = sin_ang
        R_y[2, 2] = cos_ang
        return R_y

    def get_rot_z(self, angle: float) -> np.ndarray:
        """
        Rotation matrix around Z-axis

        Parameters
        ----------
        angle: float
            Camera rotation angle around Z-axis in radians

        Returns
        ----------
        R_z: np.ndarray
            Rotation matrix of camera around Z-axis
        """
        R_z = np.zeros((3, 3))
        cos_ang = np.cos(angle)
        sin_ang = np.sin(angle)

        R_z[0, 0] = cos_ang
        R_z[0, 1] = -sin_ang
        R_z[1, 0] = sin_ang
        R_z[1, 1] = cos_ang
        R_z[2, 2] = 1
        return R_z

    def calc_rotation_matrix(self, angles: List[float], order: List[str]) -> np.ndarray:
        """
        Calculates rotation matrix by considering angles in the given order
        Note: The rotation is calculated in clockwise direction (use right-hand-rule)

        Parameters
        ----------
        angles: List[float]
            Camera rotation angle values
        order: List[str]
            Camera rotation angle orders

        Returns
        ----------
        R: np.ndarray
            Rotation matrix of the camera
        """
        rot_map = {'pitch': self.get_rot_x, 'yaw': self.get_rot_y, 'roll': self.get_rot_z}
        R = np.eye(3)
        for angle, axis in list(zip(angles, order))[::-1]:
            R_axis = rot_map.get(axis)
            R = np.matmul(R, R_axis(-angle))
        return R

    def calc_projection_matrix(self):
        """
        Calculates projection matrix between world coordinate and camera pixel coordinates.
        """
        K = self.calc_camera_calibration_matrix()

        # Basic representation of world coordinate system and pixel coordinate system
        #   _____________
        #  |      ____x  |
        #  |      |      |
        #  |      |y     |
        #  |_____________|
        #    /Z
        #   /
        #  /
        # /________X
        # |
        # |
        # |
        # Y

        yaw = math.radians(self.yaw_cam)
        pitch = math.radians(self.pitch_cam)
        roll = math.radians(self.roll_cam)
        # Camera rotation matrix [R]
        R = self.calc_rotation_matrix([yaw, pitch, roll], ['yaw', 'pitch', 'roll'])

        # Coordinates of the camera centre (camera center is selected as the origin of the world coordinate system)
        cam_C = np.array([[0.0], [0.0], [0.0]])

        # P = KR[I | −C]
        self.P = np.matmul(np.matmul(K, R), np.concatenate((np.eye(3), -cam_C), axis=1))

        # P^+ = P^T(PP^T)^−1, for which PP^+ = I
        self.P_inv = np.matmul(np.transpose(self.P), np.linalg.inv(np.matmul(self.P, np.transpose(self.P))))

    def project_pixel_coord(self, relative_object_loc: List[List[float]]) -> np.ndarray:
        """
        Projects from the world coordinate system into pixel coordinate system.

        Parameters
        ----------
        relative_object_loc: List[List[float]]
            Relative location w.r.t camera center

        Returns
        ----------
        x_norm: np.ndarray
            Pixel coordinate of the object point
        """
        # Project into pixel coordinate system
        x_pixel = np.matmul(self.P, np.array(relative_object_loc))
        # Normalize by z to get real pixel coordinates
        x_norm = np.squeeze((x_pixel / x_pixel[2][0])[:-1])
        return x_norm

    def calc_pixel_world_coordinates(self):
        """
        Back projects from pixel coordinate into fixed elevation world coordinate system.
        """
        for ind_j, j in enumerate(range(self.extended_y_start, self.extended_height + self.extended_y_start)):
            for ind_i, i in enumerate(range(self.extended_x_start, self.extended_width + self.extended_x_start)):
                unit_ray_loc = np.matmul(self.P_inv, np.array([i+0.5, j+0.5, 1]))
                if unit_ray_loc[1] > 0:  # means ray is traveling in downward direction
                    unit_ray_multiplier = -self.cam_y / unit_ray_loc[1]  # Projected elevation is 0 which means -self.cam_y for camera centered coordinate system.
                    self.pixel_world_coords[ind_j, ind_i, :] = unit_ray_multiplier * unit_ray_loc[:3] + np.array([self.cam_x, self.cam_y, self.cam_z])
        print('Pixel ray coordinates are calculated.')

    def get_relative_world_coords(self, object_x: float, object_y: float, object_z: float, rotation_angle: float):
        """
        Gets the world coordinate system points of the object and projection of it w.r.t camera centered origin and object rotation.

        Parameters
        ----------
        object_x: float
            Location of the object in X-axis
        object_y: float
            Location of the object in Y-axis
        object_z: float
            Location of the object in Z-axis
        rotation_angle: float
            Rotation angle of the objects about Y-axis in radians

        """
        rotation_angle = math.radians(rotation_angle)
        rot_cos = math.cos(rotation_angle)
        rot_sin = math.sin(rotation_angle)

        rot_length_inc = (self.obj_length / 2) * rot_cos + (self.obj_width / 2) * rot_sin
        rot_length_dec = (self.obj_length / 2) * rot_cos - (self.obj_width / 2) * rot_sin

        rot_width_inc = (self.obj_width / 2) * rot_cos + (self.obj_length / 2) * rot_sin
        rot_width_dec = (self.obj_width / 2) * rot_cos - (self.obj_length / 2) * rot_sin

        object_rel_top = [object_y - self.obj_height / 2 - self.cam_y]
        object_rel_bottom = [object_y + self.obj_height / 2 - self.cam_y]

        self.object_tlf = [[object_x - rot_length_inc - self.cam_x], object_rel_top, [object_z - rot_width_dec - self.cam_z], [1.0]]
        self.object_tlb = [[object_x - rot_length_dec - self.cam_x], object_rel_top, [object_z + rot_width_inc - self.cam_z], [1.0]]
        self.object_trf = [[object_x + rot_length_dec - self.cam_x], object_rel_top, [object_z - rot_width_inc - self.cam_z], [1.0]]
        self.object_trb = [[object_x + rot_length_inc - self.cam_x], object_rel_top, [object_z + rot_width_dec - self.cam_z], [1.0]]

        self.object_blf = [[object_x - rot_length_inc - self.cam_x], object_rel_bottom, [object_z - rot_width_dec - self.cam_z], [1.0]]
        self.object_blb = [[object_x - rot_length_dec - self.cam_x], object_rel_bottom, [object_z + rot_width_inc - self.cam_z], [1.0]]
        self.object_brf = [[object_x + rot_length_dec - self.cam_x], object_rel_bottom, [object_z - rot_width_inc - self.cam_z], [1.0]]
        self.object_brb = [[object_x + rot_length_inc - self.cam_x], object_rel_bottom, [object_z + rot_width_dec - self.cam_z], [1.0]]

        self.proj_lf = [[object_x - rot_length_inc - self.cam_x], [0 - self.cam_y], [object_z - rot_width_dec - self.cam_z], [1.0]]
        self.proj_lb = [[object_x - rot_length_dec - self.cam_x], [0 - self.cam_y], [object_z + rot_width_inc - self.cam_z], [1.0]]
        self.proj_rf = [[object_x + rot_length_dec - self.cam_x], [0 - self.cam_y], [object_z - rot_width_inc - self.cam_z], [1.0]]
        self.proj_rb = [[object_x + rot_length_inc - self.cam_x], [0 - self.cam_y], [object_z + rot_width_dec - self.cam_z], [1.0]]
        self.proj_center = [[object_x - self.cam_x], [0 - self.cam_y], [object_z - self.cam_z], [1.0]]

    def get_pixel_points(self) -> Any:
        """
        Gets the pixel coordinate system points of the object and projection of it.

        Returns
        ----------
        pixel_points: Any:
            Pixel points of various locations
        """
        # Gets the pixel coordinate system points of object and projection
        point_o_tlf = self.project_pixel_coord(self.object_tlf)
        point_o_tlb = self.project_pixel_coord(self.object_tlb)
        point_o_trf = self.project_pixel_coord(self.object_trf)
        point_o_trb = self.project_pixel_coord(self.object_trb)
        point_o_blf = self.project_pixel_coord(self.object_blf)
        point_o_blb = self.project_pixel_coord(self.object_blb)
        point_o_brf = self.project_pixel_coord(self.object_brf)
        point_o_brb = self.project_pixel_coord(self.object_brb)

        point_p_lf = self.project_pixel_coord(self.proj_lf)
        point_p_lb = self.project_pixel_coord(self.proj_lb)
        point_p_rf = self.project_pixel_coord(self.proj_rf)
        point_p_rb = self.project_pixel_coord(self.proj_rb)
        point_p_cen = self.project_pixel_coord(self.proj_center)

        # Apply radial distortion
        if self.radial_dist_enabled:
            point_o_tlf = apply_radial_dist(point_o_tlf, self.cx, self.cy, self.k_1, self.k_2)
            point_o_tlb = apply_radial_dist(point_o_tlb, self.cx, self.cy, self.k_1, self.k_2)
            point_o_trf = apply_radial_dist(point_o_trf, self.cx, self.cy, self.k_1, self.k_2)
            point_o_trb = apply_radial_dist(point_o_trb, self.cx, self.cy, self.k_1, self.k_2)

            point_o_blf = apply_radial_dist(point_o_blf, self.cx, self.cy, self.k_1, self.k_2)
            point_o_blb = apply_radial_dist(point_o_blb, self.cx, self.cy, self.k_1, self.k_2)
            point_o_brf = apply_radial_dist(point_o_brf, self.cx, self.cy, self.k_1, self.k_2)
            point_o_brb = apply_radial_dist(point_o_brb, self.cx, self.cy, self.k_1, self.k_2)

            point_p_lf = apply_radial_dist(point_p_lf, self.cx, self.cy, self.k_1, self.k_2)
            point_p_lb = apply_radial_dist(point_p_lb, self.cx, self.cy, self.k_1, self.k_2)
            point_p_rf = apply_radial_dist(point_p_rf, self.cx, self.cy, self.k_1, self.k_2)
            point_p_rb = apply_radial_dist(point_p_rb, self.cx, self.cy, self.k_1, self.k_2)
            point_p_cen = apply_radial_dist(point_p_cen, self.cx, self.cy, self.k_1, self.k_2)

        # Find limits of the bbox
        # TODO not sure that in between points can get maximum value. If not we may need to prove it.
        bbox_bottom = max(point_o_blf[1], point_o_blb[1], point_o_brf[1], point_o_brb[1])
        bbox_top = min(point_o_tlf[1], point_o_tlb[1], point_o_trf[1], point_o_trb[1])
        bbox_right = max(point_o_tlf[0], point_o_tlb[0], point_o_trf[0], point_o_trb[0], point_o_blf[0], point_o_blb[0], point_o_brf[0], point_o_brb[0])
        bbox_left = min(point_o_tlf[0], point_o_tlb[0], point_o_trf[0], point_o_trb[0], point_o_blf[0], point_o_blb[0], point_o_brf[0], point_o_brb[0])

        bbox_tl = (bbox_left, bbox_top)
        bbox_tr = (bbox_right, bbox_top)
        bbox_br = (bbox_right, bbox_bottom)
        bbox_bl = (bbox_left, bbox_bottom)

        # Add random deviation to the location of each edge of the bounding box to simulate the error of object detector
        if self.random_deviation_enabled:
            top_dev = random.gauss(0, self.deviation_sigma)
            left_dev = random.gauss(0, self.deviation_sigma)
            bottom_dev = random.gauss(0, self.deviation_sigma)
            right_dev = random.gauss(0, self.deviation_sigma)

            bbox_tl = [bbox_tl[0] + left_dev, bbox_tl[1] + top_dev]
            bbox_tr = [bbox_tr[0] + right_dev, bbox_tr[1] + top_dev]
            bbox_br = [bbox_br[0] + right_dev, bbox_br[1] + bottom_dev]
            bbox_bl = [bbox_bl[0] + left_dev, bbox_bl[1] + bottom_dev]

        bbox_points = [bbox_tl, bbox_tr, bbox_br, bbox_bl]
        proj_points = [point_p_lb, point_p_rb, point_p_rf, point_p_lf]

        # Add random deviation to the location of center projection points to simulate the effect of marking error too.
        if self.random_deviation_enabled:
            proj_cent_dev_x = random.gauss(0, self.deviation_sigma)
            proj_cent_dev_y = random.gauss(0, self.deviation_sigma)
            point_p_cen = [point_p_cen[0] + proj_cent_dev_x, point_p_cen[1] + proj_cent_dev_y]

        proj_mid_point = [point_p_cen]

        # Gets the facade points for visual representation
        front_facade = [point_o_tlf, point_o_trf, point_o_brf, point_o_blf]
        back_facade = [point_o_tlb, point_o_trb, point_o_brb, point_o_blb]
        top_facade = [point_o_tlf, point_o_tlb, point_o_trb, point_o_trf]

        object_points = [point_o_tlf, point_o_tlb, point_o_trf, point_o_trb, point_o_blf, point_o_blb, point_o_brf, point_o_brb]
        return object_points, bbox_points, proj_points, proj_mid_point, front_facade, back_facade, top_facade


def produce_data(config: ConfigParser):
    """
    Produces data including the position of the overhead object and projection point.

    Parameters
    ----------
    config: ConfigParser
        Configuration object containing the settings.
    """
    demo_mode = str2bool(config.get("sample_producer", "DEMO_MODE"))
    draw_enabled = str2bool(config.get("sample_producer", "DRAW_ENABLED"))
    cam_pixel_width = int(float(config.get("sample_producer", "CAM_PIXEL_WIDTH")))
    cam_pixel_height = int(float(config.get("sample_producer", "CAM_PIXEL_HEIGHT")))
    search_distance_step = float(config.get("sample_producer", "SEARCH_DISTANCE_STEP"))
    z_search_min = float(config.get("sample_producer", "Z_SEARCH_MIN"))
    z_search_max = float(config.get("sample_producer", "Z_SEARCH_MAX"))
    z_search_count = int(float(config.get("sample_producer", "Z_SEARCH_COUNT")))
    x_search_min = float(config.get("sample_producer", "X_SEARCH_MIN"))
    x_search_max = float(config.get("sample_producer", "X_SEARCH_MAX"))
    y_search_min = float(config.get("sample_producer", "Y_SEARCH_MIN"))
    y_search_max = float(config.get("sample_producer", "Y_SEARCH_MAX"))
    obj_height = float(config.get("sample_producer", "OBJ_HEIGHT"))
    rotate_angle_min = float(config.get("sample_producer", "ROTATE_ANGLE_MIN"))
    rotate_angle_max = float(config.get("sample_producer", "ROTATE_ANGLE_MAX"))

    export = str2bool(config.get("sample_producer", "EXPORT"))
    path_to_output_folder = config.get("sample_producer", "PATH_TO_OUTPUT_FOLDER")
    auxiliary_data_path = config.get("sample_producer", "AUXILIARY_DATA_PATH")
    projection_data_path = config.get("sample_producer", "PROJECTION_DATA_PATH")

    inputs = []
    outputs = []
    cam_width_heights = []
    point_estimator = PointEstimatorProjection(config)

    PAUSE_FIG_TIME = 0.01  # Delay time applied during the consecutive drawings
    if demo_mode:  # For the demo mode delay time is increased for the sake of easy monitoring
        PAUSE_FIG_TIME = 0.50

    cam_width_height = np.array((cam_pixel_width, cam_pixel_height))
    print('Sample collection is started!')

    for z in tqdm(np.logspace(np.log10(z_search_min), np.log10(z_search_max), num=z_search_count)):
        # Logarithmic search procedure is applied so that distant observations in z-dimension do not dominate the sample set.
        for x in np.arange(x_search_min, x_search_max, search_distance_step):
            for y in np.arange(max(-y_search_min, obj_height / 2), -y_search_max, search_distance_step):
                rotate_angle = random.uniform(rotate_angle_min, rotate_angle_max)
                if demo_mode:
                    point_estimator.get_relative_world_coords(object_x=random.uniform(x_search_min, x_search_max),
                                                              object_y=random.uniform(y_search_min, y_search_max),
                                                              object_z=random.uniform(z_search_min, z_search_max),
                                                              rotation_angle=rotate_angle)
                else:
                    point_estimator.get_relative_world_coords(object_x=x, object_y=-y, object_z=z, rotation_angle=rotate_angle)

                obj, bbox, proj, proj_mid, front, back, top = point_estimator.get_pixel_points()
                mid_bottom_coord, bbox_width_height, proj_coord_offset = calculate_input_coord(bbox, proj_mid)
                # Filter the sample set in a such way that both object and center of the projection are completely visible
                # in the camera.
                if (0 <= (mid_bottom_coord[0] - bbox_width_height[0] / 2)) and (
                        (mid_bottom_coord[0] + bbox_width_height[0] / 2) < cam_pixel_width) and (
                        mid_bottom_coord[1] - bbox_width_height[1] >= 0) and (
                        mid_bottom_coord[1] < cam_pixel_height) and (
                        mid_bottom_coord[1] + proj_coord_offset[1] < cam_pixel_height):
                    if not demo_mode:  # For the visual demo mode the storing operation is skipped.
                        input_sample = np.append(mid_bottom_coord, bbox_width_height)
                        inputs.append(input_sample)
                        outputs.append(proj_coord_offset)
                        cam_width_heights.append(cam_width_height)

                    if demo_mode or draw_enabled:  # Printing of the results
                        image = np.ones((cam_pixel_height, cam_pixel_width, 3), dtype=np.uint8) * 90
                        connect_and_draw_points(image, bbox, GREEN_COLOR)

                        connect_and_draw_points(image, [obj[0]], RED_COLOR)
                        connect_and_draw_points(image, [obj[1]], RED_COLOR)
                        connect_and_draw_points(image, [obj[2]], RED_COLOR)
                        connect_and_draw_points(image, [obj[3]], RED_COLOR)
                        connect_and_draw_points(image, [obj[4]], RED_COLOR)
                        connect_and_draw_points(image, [obj[5]], RED_COLOR)
                        connect_and_draw_points(image, [obj[6]], RED_COLOR)
                        connect_and_draw_points(image, [obj[7]], RED_COLOR)

                        # connect_and_draw_points(image, proj, WHITE_COLOR)
                        connect_and_draw_points(image, proj_mid, RED_COLOR)
                        connect_and_draw_points(image, front, ORANGE_COLOR)
                        connect_and_draw_points(image, back, ORANGE_COLOR)
                        connect_and_draw_points(image, top, ORANGE_COLOR)
                        plt.imshow(image)
                        plt.pause(PAUSE_FIG_TIME)
                        plt.cla()

    if (not demo_mode) and export:
        export_data = np.hstack((inputs, outputs, cam_width_heights))
        dir_path = path_to_output_folder
        path_to_auxiliary_data = os.path.join(dir_path, auxiliary_data_path)
        path_to_projection_data = os.path.join(dir_path, projection_data_path)

        os.makedirs(dir_path, exist_ok=True)
        with open(path_to_auxiliary_data, 'wb') as handle:
            pickle.dump([point_estimator.pixel_world_coords, point_estimator.top_left_pixel_coord], handle, protocol=pickle.HIGHEST_PROTOCOL)
        np.savetxt(path_to_projection_data, export_data, header='x_coord_mid_bottom y_coord_mid_bottom bbox_width bbox_height proj_x_dist_to_mid_bottom proj_y_dist_to_mid_bottom cam_width cam_height', fmt='%1.6e')  # X is an array
        print('\nSample production is completed with {:6d} samples!'.format(export_data.shape[0]))


if __name__ == '__main__':
    parser = ArgumentParser(description="Script to produce data including the position of the overhead object and projection point.")
    parser.add_argument("--settings_path", help="Path to the settings file.", default=r"./settings.ini")

    args = parser.parse_args()
    config_ = read_settings(args.settings_path)
    produce_data(config=config_)
