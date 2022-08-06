import sys
import os
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

DEMO_MODE = False
FIXED_SEED = True
FIXED_SEED_NUM = 35
DRAW_ENABLED = True
LINE_THICKNESS = 1
GREEN_COLOR = (0, 255, 0)
YELLOW_COLOR = (255, 255, 0)
RED_COLOR = (255, 0, 0)
WHITE_COLOR = (255, 255, 255)
PAUSE_FIG_TIME = 0.01


CAM_FOV_HOR = 60  # in degrees
CAM_PIXEL_WIDTH = 1280  # number of horizontal pixel
CAM_PIXEL_HEIGHT = 720  # number of vertical pixel
HALF_PIXEL_SIZE = 1 / 2
CAM_REGION_EXPAND_RATIO = 3.0  # Odd integer number is advised.
RADIAL_DIST_ENABLED = True
K_1 = -0.05
K_2 = 0.0
RANDOM_DEVIATION_ENABLED = False
DEVIATON_SIGMA = 5
ROTATE_ANGLE_MIN = -10
ROTATE_ANGLE_MAX = 10

YAW_CAM = -15  # y dimension positive is clockwise (in degrees)
PITCH_CAM = -10  # x dimension positive is upward (in degrees)
ROLL_CAM = 0  # z dimension positive is clockwise (in degrees)

OBJ_HEIGHT = 2.5
OBJ_WIDTH = 2.0
OBJ_LENGTH = 1.0

Z_SEARCH_MIN = 20.0
Z_SEARCH_MAX = 50.0
Z_SEARCH_COUNT = 50

X_SEARCH_MIN = -40.0
X_SEARCH_MAX = 40.0

Y_SEARCH_MIN = -1.25
Y_SEARCH_MAX = -10.0
SEARCH_DISTANCE_STEP = 0.5

CAM_Z = 0
CAM_Y = -8.0
CAM_X = -5.0

EXPORT_TO_TXT: bool = True
PATH_TO_OUTPUT_FILE = '/home/poyraz/intenseye/input_outputs/overhead_object_projector/inputs_outputs_new.txt'


def connect_and_draw_points(image, points, color):
    if len(points) == 1:
        cv2.drawMarker(image, np.round(points[0]).astype(int), color=color, markerType=cv2.MARKER_STAR,
                       thickness=LINE_THICKNESS)
    elif len(points) > 1:
        for i in range(len(points)):
            if i == len(points) - 1:
                cv2.line(image, np.round(points[i]).astype(int), np.round(points[0]).astype(int), color=color,
                         thickness=LINE_THICKNESS)
            else:
                cv2.line(image, np.round(points[i]).astype(int), np.round(points[i + 1]).astype(int), color=color,
                         thickness=LINE_THICKNESS)
    else:
        print('Given point list is empty!')


def cotan(radians):
    return 1 / math.tan(radians)


def apply_radial_dist(point, cx, cy):
    norm_dist_x, norm_dist_y = (np.array(point) - np.array((cx, cy))) / np.array((cx, cy))
    r2 = norm_dist_x ** 2 + norm_dist_y ** 2
    distorted_point = np.array((norm_dist_x * (1 + K_1 * r2 + K_2 * r2**2), norm_dist_y * (1 + K_1 * r2 + K_2 * r2**2))) * np.array(
        (cx, cy)) + np.array((cx, cy))
    return distorted_point


def calculate_input_coord(bbox, proj_mid):
    bbox_bottom_center = ((np.array(bbox[2]) + np.array(bbox[3])) / 2)
    bbox_width_height = np.array((bbox[1][0] - bbox[0][0], bbox[2][1] - bbox[1][1]))
    proj_wrt_bbox_bottom_center = proj_mid[0] - bbox_bottom_center

    return bbox_bottom_center, bbox_width_height, proj_wrt_bbox_bottom_center


class PointEstimatorProjection:
    def __init__(self):

        if FIXED_SEED:
            os.environ['PYTHONHASHSEED'] = str(FIXED_SEED_NUM)
            random.seed(FIXED_SEED_NUM)
            np.random.seed(FIXED_SEED_NUM)

        self.cx = CAM_PIXEL_WIDTH / 2  # horizontal center pixel loc.                                                               (p_c^x in paper)
        self.cy = CAM_PIXEL_HEIGHT / 2  # vertical center pixel loc.

        self.extended_height = int(CAM_REGION_EXPAND_RATIO * CAM_PIXEL_HEIGHT)
        self.extended_width = int(CAM_REGION_EXPAND_RATIO * CAM_PIXEL_WIDTH)
        self.extended_y_start = -int((CAM_REGION_EXPAND_RATIO - 1) * CAM_PIXEL_HEIGHT / 2)
        self.extended_x_start = -int((CAM_REGION_EXPAND_RATIO - 1) * CAM_PIXEL_WIDTH / 2)
        self.top_left_pixel_coord = np.array([self.extended_x_start, self.extended_y_start])

        self.distance_along_axis = np.ones(self.extended_height, dtype=float) * sys.float_info.max  #                                   (d_a^j in paper)
        self.distance_perp_axis = np.ones((self.extended_height, self.extended_width), dtype=float) * sys.float_info.max

        self.P = self.calc_projection_matrix()

    def calc_camera_calibration_matrix(self):
        focal_length = cotan(math.radians(CAM_FOV_HOR / 2)) * self.cx
        # Cameras intrinsic matrix [K]
        K = np.array(
            [[focal_length, 0., self.cx],
             [0., focal_length, self.cy],
             [0., 0., 1.]]
        )
        return K

    def get_rot_x(self, angle):
        '''
        Rotation matrix around X axis
        '''
        R_x = np.zeros((3, 3))
        cos_ang = np.cos(angle)
        sin_ang = np.sin(angle)

        R_x[0, 0] = 1
        R_x[1, 1] = cos_ang
        R_x[1, 2] = -sin_ang
        R_x[2, 1] = sin_ang
        R_x[2, 2] = cos_ang
        return R_x

    def get_rot_y(self, angle):
        '''
        Rotation matrix around Y axis
        '''
        R_y = np.zeros((3, 3))
        cos_ang = np.cos(angle)
        sin_ang = np.sin(angle)

        R_y[0, 0] = cos_ang
        R_y[0, 2] = -sin_ang
        R_y[1, 1] = 1
        R_y[2, 0] = sin_ang
        R_y[2, 2] = cos_ang
        return R_y

    def get_rot_z(self, angle):
        '''
        Rotation matrix around Z axis
        '''
        R_z = np.zeros((3, 3))
        cos_ang = np.cos(angle)
        sin_ang = np.sin(angle)

        R_z[0, 0] = cos_ang
        R_z[0, 1] = -sin_ang
        R_z[1, 0] = sin_ang
        R_z[1, 1] = cos_ang
        R_z[2, 2] = 1
        return R_z

    def calc_rotation_matrix(self, angles, order):
        '''
        Calculates rotation martix by considering angles in the given order
        Note: The rotation is calculated in clockwise direction (use right-hand-rule)
        '''
        rot_map = {'pitch': self.get_rot_x, 'yaw': self.get_rot_y, 'roll': self.get_rot_z}
        R = np.eye(3)
        for angle, axis in list(zip(angles, order))[::-1]:
            R_axis = rot_map.get(axis)
            R = np.matmul(R, R_axis(-angle))
        return R

    def calc_projection_matrix(self):
        K = self.calc_camera_calibration_matrix()
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

        yaw = math.radians(YAW_CAM)
        pitch = math.radians(PITCH_CAM)
        roll = math.radians(ROLL_CAM)
        # Camera rotation matrix [R]
        R = self.calc_rotation_matrix([yaw, pitch, roll], ['yaw', 'pitch', 'roll'])

        # Coordinates of the camera centre
        camC = np.array([[0.0], [0.0], [0.0]])

        P = np.matmul(np.matmul(K, R), np.concatenate((np.eye(3), -camC), axis=1))
        return P

    def calc_pixel_coord(self, relative_object_loc):
        # Project (Xw, Yw, Zw, 0) into pixel coordinate system
        x_pixel = np.matmul(self.P, np.array(relative_object_loc))
        # Normalize by z to get (u,v,1)
        x_norm = np.squeeze((x_pixel / x_pixel[2][0])[:-1])
        return x_norm

    def calc_pixel_distances(self):
        for ind_j, j in enumerate(range(self.extended_y_start, self.extended_height + self.extended_y_start)):
            pixel_ver_angle = self.alpha_in_degrees_ver - (j - self.cy + HALF_PIXEL_SIZE) * self.IFOV  #                            (delta_j in paper)
            if pixel_ver_angle < 90:
                self.distance_along_axis[ind_j] = self.z_ver * (math.tan(math.radians(self.alpha_in_degrees_ver)) - math.tan(math.radians(pixel_ver_angle)))
                distance_to_center = self.z_ver / math.cos(math.radians(pixel_ver_angle))
                for ind_i, i in enumerate(range(self.extended_x_start, self.extended_width + self.extended_x_start)):
                    zeta_angle = (i - self.cx + HALF_PIXEL_SIZE) * self.IFOV  #                                                     (zeta_i in paper)
                    self.distance_perp_axis[ind_j, ind_i] = distance_to_center * math.tan(math.radians(zeta_angle))

    def get_relative_world_coords(self, object_x, object_y, object_z, rotate_angle=None):

        object_rel_left = [object_x - OBJ_LENGTH / 2 - CAM_X]
        object_rel_right = [object_x + OBJ_LENGTH / 2 - CAM_X]

        object_rel_top = [object_y - OBJ_HEIGHT / 2 - CAM_Y]
        object_rel_bottom = [object_y + OBJ_HEIGHT / 2 - CAM_Y]

        object_rel_front = [object_z - OBJ_WIDTH / 2 - CAM_Z]
        object_rel_back = [object_z + OBJ_WIDTH / 2 - CAM_Z]

        self.object_tlf = [object_rel_left, object_rel_top, object_rel_front, [1]]
        self.object_tlb = [object_rel_left, object_rel_top, object_rel_back, [1]]
        self.object_trf = [object_rel_right, object_rel_top, object_rel_front, [1]]
        self.object_trb = [object_rel_right, object_rel_top, object_rel_back, [1]]

        self.object_blf = [object_rel_left, object_rel_bottom, object_rel_front, [1]]
        self.object_blb = [object_rel_left, object_rel_bottom, object_rel_back, [1]]
        self.object_brf = [object_rel_right, object_rel_bottom, object_rel_front, [1]]
        self.object_brb = [object_rel_right, object_rel_bottom, object_rel_back, [1]]

        self.proj_lf = [object_rel_left, [0 - CAM_Y], object_rel_front, [1]]
        self.proj_lb = [object_rel_left, [0 - CAM_Y], object_rel_back, [1]]
        self.proj_rf = [object_rel_right, [0 - CAM_Y], object_rel_front, [1]]
        self.proj_rb = [object_rel_right, [0 - CAM_Y], object_rel_back, [1]]
        self.proj_center = [[object_x - CAM_X], [0 - CAM_Y], [object_z - CAM_Z], [1]]

    def get_pixel_points(self):
        point_o_tlf = self.calc_pixel_coord(self.object_tlf)
        point_o_tlb = self.calc_pixel_coord(self.object_tlb)
        point_o_trf = self.calc_pixel_coord(self.object_trf)
        point_o_trb = self.calc_pixel_coord(self.object_trb)
        point_o_blf = self.calc_pixel_coord(self.object_blf)
        point_o_blb = self.calc_pixel_coord(self.object_blb)
        point_o_brf = self.calc_pixel_coord(self.object_brf)
        point_o_brb = self.calc_pixel_coord(self.object_brb)

        point_p_lf = self.calc_pixel_coord(self.proj_lf)
        point_p_lb = self.calc_pixel_coord(self.proj_lb)
        point_p_rf = self.calc_pixel_coord(self.proj_rf)
        point_p_rb = self.calc_pixel_coord(self.proj_rb)
        point_p_cen = self.calc_pixel_coord(self.proj_center)

        if RADIAL_DIST_ENABLED:
            point_o_tlf = apply_radial_dist(point_o_tlf, self.cx, self.cy)
            point_o_tlb = apply_radial_dist(point_o_tlb, self.cx, self.cy)
            point_o_trf = apply_radial_dist(point_o_trf, self.cx, self.cy)
            point_o_trb = apply_radial_dist(point_o_trb, self.cx, self.cy)

            point_o_blf = apply_radial_dist(point_o_blf, self.cx, self.cy)
            point_o_blb = apply_radial_dist(point_o_blb, self.cx, self.cy)
            point_o_brf = apply_radial_dist(point_o_brf, self.cx, self.cy)
            point_o_brb = apply_radial_dist(point_o_brb, self.cx, self.cy)

            point_p_lf = apply_radial_dist(point_p_lf, self.cx, self.cy)
            point_p_lb = apply_radial_dist(point_p_lb, self.cx, self.cy)
            point_p_rf = apply_radial_dist(point_p_rf, self.cx, self.cy)
            point_p_rb = apply_radial_dist(point_p_rb, self.cx, self.cy)
            point_p_cen = apply_radial_dist(point_p_cen, self.cx, self.cy)

        bbox_bottom = max(point_o_blf[1], point_o_blb[1], point_o_brf[1], point_o_brb[1])  # bbox_bottom  (p_b^y in paper)
        bbox_top = min(point_o_tlf[1], point_o_tlb[1], point_o_trf[1], point_o_trb[1])  # bbox_top  (p_t^y in paper)
        bbox_right = max(point_o_tlf[0], point_o_tlb[0], point_o_trf[0], point_o_trb[0], point_o_blf[0], point_o_blb[0], point_o_brf[0], point_o_brb[0])  # bbox_bottom  (p_b^y in paper)
        bbox_left = min(point_o_tlf[0], point_o_tlb[0], point_o_trf[0], point_o_trb[0], point_o_blf[0], point_o_blb[0], point_o_brf[0], point_o_brb[0])

        bbox_tl = (bbox_left, bbox_top)
        bbox_tr = (bbox_right, bbox_top)
        bbox_br = (bbox_right, bbox_bottom)
        bbox_bl = (bbox_left, bbox_bottom)

        if RANDOM_DEVIATION_ENABLED:
            top_dev = random.gauss(0, DEVIATON_SIGMA)
            bottom_dev = random.gauss(0, DEVIATON_SIGMA)
            left_dev = random.gauss(0, DEVIATON_SIGMA)
            right_dev = random.gauss(0, DEVIATON_SIGMA)

            bbox_tl = [bbox_tl[0] + left_dev, bbox_tl[1] + top_dev]
            bbox_tr = [bbox_tr[0] + right_dev, bbox_tr[1] + top_dev]
            bbox_br = [bbox_br[0] + right_dev, bbox_br[1] + bottom_dev]
            bbox_bl = [bbox_bl[0] + left_dev, bbox_bl[1] + bottom_dev]

        bbox_points = [bbox_tl, bbox_tr, bbox_br, bbox_bl]
        proj_points = [point_p_lb, point_p_rb, point_p_rf, point_p_lf]
        proj_mid_point = [point_p_cen]

        front_facade = [point_o_tlf, point_o_trf, point_o_brf, point_o_blf]
        back_facade = [point_o_tlb, point_o_trb, point_o_brb, point_o_blb]
        top_facade = [point_o_tlf, point_o_tlb, point_o_trb, point_o_trf]

        return bbox_points, proj_points, proj_mid_point, front_facade, back_facade, top_facade


inputs = []
outputs = []
cam_width_heights = []
point_estimator = PointEstimatorProjection()

if DEMO_MODE:
    PAUSE_FIG_TIME = 0.50

cam_width_height = np.array((CAM_PIXEL_WIDTH, CAM_PIXEL_HEIGHT))
print('Sample collection is started!')

for z in tqdm(np.logspace(np.log10(Z_SEARCH_MIN), np.log10(Z_SEARCH_MAX), num=Z_SEARCH_COUNT)):
    for x in np.arange(X_SEARCH_MIN, X_SEARCH_MAX, SEARCH_DISTANCE_STEP):
        for y in np.arange(-Y_SEARCH_MIN, -Y_SEARCH_MAX, SEARCH_DISTANCE_STEP):
            rotate_angle = random.uniform(ROTATE_ANGLE_MIN, ROTATE_ANGLE_MAX)
            if DEMO_MODE:
                point_estimator.get_relative_world_coords(object_x=random.uniform(X_SEARCH_MIN, X_SEARCH_MAX),
                                                          object_y=random.uniform(Y_SEARCH_MIN, Y_SEARCH_MAX),
                                                          object_z=random.uniform(Z_SEARCH_MIN, Z_SEARCH_MAX),
                                                          rotate_angle=rotate_angle)
            else:
                point_estimator.get_relative_world_coords(object_x=x, object_y=-y, object_z=z, rotate_angle=rotate_angle)

            bbox, proj, proj_mid, front, back, top = point_estimator.get_pixel_points()
            mid_bottom_coord, bbox_width_height, proj_coord_offset = calculate_input_coord(bbox, proj_mid)
            if (0 <= (mid_bottom_coord[0] - bbox_width_height[0] / 2)) and (
                    (mid_bottom_coord[0] + bbox_width_height[0] / 2) < CAM_PIXEL_WIDTH) and (
                    mid_bottom_coord[1] - bbox_width_height[1] >= 0) and (
                    mid_bottom_coord[1] < CAM_PIXEL_HEIGHT) and (
                    mid_bottom_coord[1] + proj_coord_offset[1] < CAM_PIXEL_HEIGHT):
                if not DEMO_MODE:
                    input_sample = np.append(mid_bottom_coord, bbox_width_height)
                    inputs.append(input_sample)
                    outputs.append(proj_coord_offset)
                    cam_width_heights.append(cam_width_height)

                if DEMO_MODE or DRAW_ENABLED:
                    image = np.zeros((CAM_PIXEL_HEIGHT, CAM_PIXEL_WIDTH, 3), dtype=np.uint8)
                    connect_and_draw_points(image, bbox, GREEN_COLOR)
                    connect_and_draw_points(image, proj, YELLOW_COLOR)
                    connect_and_draw_points(image, proj_mid, RED_COLOR)
                    connect_and_draw_points(image, front, WHITE_COLOR)
                    connect_and_draw_points(image, back, WHITE_COLOR)
                    connect_and_draw_points(image, top, WHITE_COLOR)
                    plt.imshow(image)
                    plt.pause(PAUSE_FIG_TIME)
                    plt.cla()

if (not DEMO_MODE) and EXPORT_TO_TXT:
    export_data = np.hstack((inputs, outputs, cam_width_heights))
    dir_path = os.path.dirname(PATH_TO_OUTPUT_FILE)
    path_to_auxiliary_data = os.path.join(dir_path, 'auxiliary_data_w_roll_temp.pickle')
    os.makedirs(dir_path, exist_ok=True)
    with open(path_to_auxiliary_data, 'wb') as handle:
        pickle.dump([point_estimator.distance_along_axis, point_estimator.distance_perp_axis, point_estimator.top_left_pixel_coord], handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.savetxt(PATH_TO_OUTPUT_FILE, export_data, header='x_coord_mid_bottom y_coord_mid_bottom bbox_width bbox_height cam_width cam_height proj_x_dist_to_mid_bottom proj_y_dist_to_mid_bottom', fmt='%1.6e')  # X is an array
print('\nSample production is completed with {:6d} samples!'.format(export_data.shape[0]))
