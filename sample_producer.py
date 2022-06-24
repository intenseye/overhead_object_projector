import sys
import os
import math
import numpy as np
import cv2
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

FIXED_SEED = True
FIXED_SEED_NUM = 35
DEMO_MODE = False
CAM_FOV_HOR = 60  # in degrees
CAM_PIXEL_WIDTH = 1280  # number of horizontal pixel
CAM_PIXEL_HEIGHT = 720  # number of vertical pixel
HALF_PIXEL_SIZE = 1 / 2
CAM_REGION_EXPAND_RATIO = 3.0  # Odd integer number is advised.
CAM_ALTITUDE = 8.0
CAM_POS_WRT_REFERENCE = 5.0
CAM_PITCH_ANGLE = 80.0
CAM_YAW_ANGLE = 75.0
OBJ_HEIGHT = 2.5
OBJ_WIDTH = 2.0
OBJ_LENGTH = 1.0
LINE_THICKNESS = 1
DRAW_ENABLED = False
PAUSE_FIG_TIME = 0.01
ROLL_ENABLED = True
ROLL_ANGLE = 10  # in degrees (clockwise direction)
RADIAL_DIST_ENABLED = True
K_1 = -0.05
K_2 = 0.0
RANDOM_DEVIATION_ENABLED = False
DEVIATON_SIGMA = 5
ROTATE_ANGLE_MIN = -10
ROTATE_ANGLE_MAX = 10
X_SEARCH_MIN = 20.0
X_SEARCH_MAX = 50.0
X_SEARCH_COUNT = 50
Y_HOR_SEARCH_MIN = -40.0
Y_HOR_SEARCH_MAX = 40.0
Y_VER_SEARCH_MIN = 1.25
X_VER_SEARCH_MAX = 10.0
SEARCH_DISTANCE_STEP = 0.5
EXPORT_TO_TXT = True
PATH_TO_OUTPUT_FILE = '/home/poyraz/intenseye/input_outputs/overhead_object_projector/inputs_outputs_w_roll_temp.txt'
GREEN_COLOR = (0, 255, 0)
YELLOW_COLOR = (255, 255, 0)
RED_COLOR = (255, 0, 0)
WHITE_COLOR = (255, 255, 255)


def cotan(radians):
    return 1 / math.tan(radians)


def connect_and_draw_points(image, points, color):
    if len(points) == 1:
        cv2.drawMarker(image, np.round(proj_mid[0]).astype(int), color=color, markerType=cv2.MARKER_STAR,
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


def apply_radial_dist(point, cx, cy):
    norm_dist_x, norm_dist_y = (np.array(point) - np.array((cx, cy))) / np.array((cx, cy))
    r2 = norm_dist_x ** 2 + norm_dist_y ** 2
    distorted_point = np.array((norm_dist_x * (1 + K_1 * r2 + K_2 * r2**2), norm_dist_y * (1 + K_1 * r2 + K_2 * r2**2))) * np.array(
        (cx, cy)) + np.array((cx, cy))
    return distorted_point


def apply_roll(point, cx, cy):
    dist_x, dist_y = np.array(point) - np.array((cx, cy))
    r = math.sqrt(dist_x ** 2 + dist_y ** 2)
    # arctan of minus(-) dist_y is used to convert from pixel coordinates to cartesian coordinates.
    polar_angle = math.atan2(-dist_y, dist_x)
    rotated_polar_angle = polar_angle + math.radians(ROLL_ANGLE)
    # -math.sin(rotated_polar_angle)) is used to convert from cartesian coordinates to pixel coordinates.
    rolled_point = np.array((r * math.cos(rotated_polar_angle), r * -math.sin(rotated_polar_angle))) + np.array((cx, cy))
    return rolled_point


def calculate_input_coord(bbox, proj_mid):
    bbox_bottom_center = ((np.array(bbox[2]) + np.array(bbox[3])) / 2)
    bbox_width_height = np.array((bbox[1][0] - bbox[0][0], bbox[2][1] - bbox[1][1]))
    proj_wrt_bbox_bottom_center = proj_mid[0] - bbox_bottom_center

    return bbox_bottom_center, bbox_width_height, proj_wrt_bbox_bottom_center


class PointEstimator:
    def __init__(self):

        if FIXED_SEED:
            os.environ['PYTHONHASHSEED'] = str(FIXED_SEED_NUM)
            random.seed(FIXED_SEED_NUM)
            np.random.seed(FIXED_SEED_NUM)

        self.z_ver = CAM_ALTITUDE  # cam_altitude in meters                                     (z in paper)
        self.z_hor = CAM_POS_WRT_REFERENCE  # cam_point w.r.t reference point in meters         (z' in paper)
        self.alpha_in_degrees_ver = CAM_PITCH_ANGLE  # cam_pitch_angle in degrees (w.r.t the vertical plane not horizontal)        (alpha in paper)
        alpha_in_degrees_hor = CAM_YAW_ANGLE  # cam_yaw_angle in degrees                                                           (alpha' in paper)

        self.h = OBJ_HEIGHT  # obj_height in meters    (h in paper)
        self.w = OBJ_WIDTH  # obj_width in meters      (w in paper)
        self.t = OBJ_LENGTH  # obj_length in meters    (t in paper)

        self.cx = CAM_PIXEL_WIDTH / 2  # horizontal center pixel loc.                                                               (p_c^x in paper)
        self.cy = CAM_PIXEL_HEIGHT / 2  # vertical center pixel loc.                                                                (p_c^y in paper)

        self.IFOV = CAM_FOV_HOR / CAM_PIXEL_WIDTH  # It is assumed that IFOV is the same in both x and y dimensions
        self.alpha_ver = math.radians(self.alpha_in_degrees_ver)
        self.alpha_hor = math.radians(alpha_in_degrees_hor)

        self.y_ver = 0  # obj center point altitude in meters                                                                       (y in paper)
        self.y_hor = 0  # obj center point w.r.t the reference point in meters                                                      (y' in paper)
        self.x = 0  # cam_to_obj_nearest_point_hor_dist in meters                                                                   (x in paper)
        self.rotate_angle = 0

        self.W = np.zeros((2, 2), dtype=float)
        self.T = np.zeros((2, 2), dtype=float)

        self.w_prime = 0
        self.w_double_prime = 0
        self.t_prime = 0
        self.t_double_prime = 0

        self.extended_height = int(CAM_REGION_EXPAND_RATIO * CAM_PIXEL_HEIGHT)
        self.extended_width = int(CAM_REGION_EXPAND_RATIO * CAM_PIXEL_WIDTH)

        self.extended_y_start = -int((CAM_REGION_EXPAND_RATIO - 1) * CAM_PIXEL_HEIGHT / 2)
        self.extended_x_start = -int((CAM_REGION_EXPAND_RATIO - 1) * CAM_PIXEL_WIDTH / 2)
        self.top_left_pixel_coord = np.array([self.extended_x_start, self.extended_y_start])

        self.distance_along_axis = np.ones(self.extended_height, dtype=float) * sys.float_info.max  #                                   (d_a^j in paper)
        self.distance_perp_axis = np.ones((self.extended_height, self.extended_width), dtype=float) * sys.float_info.max  #                 (d_p^{j,i} in paper)

        self.calc_pixel_distances()

    def calc_pixel_distances(self):
        for ind_j, j in enumerate(range(self.extended_y_start, self.extended_height + self.extended_y_start)):
            pixel_ver_angle = self.alpha_in_degrees_ver - (j - self.cy + HALF_PIXEL_SIZE) * self.IFOV  #                            (delta_j in paper)
            if pixel_ver_angle < 90:
                self.distance_along_axis[ind_j] = self.z_ver * (math.tan(math.radians(self.alpha_in_degrees_ver)) - math.tan(math.radians(pixel_ver_angle)))
                distance_to_center = self.z_ver / math.cos(math.radians(pixel_ver_angle))
                for ind_i, i in enumerate(range(self.extended_x_start, self.extended_width + self.extended_x_start)):
                    zeta_angle = (i - self.cx + HALF_PIXEL_SIZE) * self.IFOV  #                                                     (zeta_i in paper)
                    self.distance_perp_axis[ind_j, ind_i] = distance_to_center * math.tan(math.radians(zeta_angle))

    def set_object_location(self, y_ver, y_hor, x, rotate_angle):
        self.y_ver = y_ver
        self.y_hor = y_hor
        self.x = x
        self.rotate_angle = rotate_angle

        self.w_prime = self.w * math.cos(math.radians(self.rotate_angle)) + self.t * math.sin(math.radians(self.rotate_angle))
        self.w_double_prime = self.w * math.cos(math.radians(self.rotate_angle)) - self.t * math.sin(math.radians(self.rotate_angle))
        self.t_prime = self.t * math.cos(math.radians(self.rotate_angle)) + self.w * math.sin(math.radians(self.rotate_angle))
        self.t_double_prime = self.t * math.cos(math.radians(self.rotate_angle)) - self.w * math.sin(math.radians(self.rotate_angle))

        self.W[0][0] = -1 * self.w_prime
        self.W[0][1] = -1 * self.w_double_prime
        self.W[1][0] = self.w_double_prime
        self.W[1][1] = self.w_prime

        self.T[0][0] = self.t_double_prime
        self.T[0][1] = -1 * self.t_prime
        self.T[1][0] = self.t_prime
        self.T[1][1] = -1 * self.t_double_prime

    def calc_vertical_point_loc(self):
        self.bottom_points = np.zeros((2, 2), dtype=float)
        for j in range(2):
            for k in range(2):
                u = (self.y_ver - self.z_ver + (self.x + self.W[j][k] / 2) * cotan(self.alpha_ver) - self.h / 2) * math.sin(
                    self.alpha_ver)
                denominator = (((self.x + self.W[j][k] / 2) / math.sin(self.alpha_ver)) - u * cotan(self.alpha_ver))
                beta_in_radians = math.atan2(u, denominator)
                beta = math.degrees(beta_in_radians)
                self.bottom_points[j][k] = self.cy - (beta / self.IFOV)

        self.top_points = np.zeros((2, 2), dtype=float)
        for j in range(2):
            for k in range(2):
                v = (self.y_ver - self.z_ver + (self.x + self.W[j][k] / 2) * cotan(self.alpha_ver) + self.h / 2) * math.sin(
                    self.alpha_ver)
                denominator = (((self.x + self.W[j][k] / 2) / math.sin(self.alpha_ver)) - v * cotan(self.alpha_ver))
                theta_in_radians = math.atan2(v, denominator)
                theta = math.degrees(theta_in_radians)
                self.top_points[j][k] = self.cy - (theta / self.IFOV)

        self.proj_y = np.zeros((2, 2), dtype=float)
        for j in range(2):
            for k in range(2):
                gamma_in_radians = self.alpha_ver - math.atan2((self.x + self.W[j][k] / 2), self.z_ver)
                gamma = math.degrees(gamma_in_radians)
                self.proj_y[j][k] = self.cy + (gamma / self.IFOV)

        gamma_in_radians = self.alpha_ver - math.atan2(self.x, self.z_ver)  #                (gamma in paper)
        gamma = math.degrees(gamma_in_radians)
        self.proj_mid_y = self.cy + (gamma / self.IFOV)  #                                   (p_{proj}^{y] in paper)

    def calc_horizontal_point_loc(self):
        self.proj_x = np.zeros((2, 2), dtype=float)

        self.right_points = np.zeros(2, dtype=float)
        for j in range(2):
            u = (self.y_hor - self.z_hor + (self.x + self.W[j][1] / 2) * cotan(self.alpha_hor) + self.T[j][1] / 2) * math.sin(self.alpha_hor)
            denominator = (((self.x + self.W[j][1] / 2) / math.sin(self.alpha_hor)) - u * cotan(self.alpha_hor))
            beta_in_radians = math.atan2(u, denominator)
            beta = math.degrees(beta_in_radians)
            self.right_points[j] = self.cx - (beta / self.IFOV)
            self.proj_x[j][1] = self.cx - (beta / self.IFOV)

        self.left_points = np.zeros(2, dtype=float)
        for j in range(2):
            v = (self.y_hor - self.z_hor + (self.x + self.W[j][0] / 2) * cotan(self.alpha_hor) + self.T[j][0] / 2) * math.sin(self.alpha_hor)
            denominator = (((self.x + self.W[j][0] / 2) / math.sin(self.alpha_hor)) - v * cotan(self.alpha_hor))
            theta_in_radians = math.atan2(v, denominator)
            theta = math.degrees(theta_in_radians)
            self.left_points[j] = self.cx - (theta / self.IFOV)
            self.proj_x[j][0] = self.cx - (theta / self.IFOV)

        c = (self.y_hor - self.z_hor + self.x * cotan(self.alpha_hor)) * math.sin(self.alpha_hor)
        denominator = (self.x / math.sin(self.alpha_hor)) - c * cotan(self.alpha_hor)
        gamma_prime_in_radians = math.atan2(c, denominator)
        gamma_prime = math.degrees(gamma_prime_in_radians)
        self.proj_mid_x = self.cx - (gamma_prime / self.IFOV)  #                                   (p_{proj}^{x] in paper)

    def get_bbox_proj_points(self):

        front_tl = (self.left_points[0], self.top_points[0][0])
        front_tr = (self.right_points[0], self.top_points[0][1])
        front_br = (self.right_points[0], self.bottom_points[0][1])
        front_bl = (self.left_points[0], self.bottom_points[0][0])

        back_tl = (self.left_points[1], self.top_points[1][0])
        back_tr = (self.right_points[1], self.top_points[1][1])
        back_br = (self.right_points[1], self.bottom_points[1][1])
        back_bl = (self.left_points[1], self.bottom_points[1][0])

        proj_near_l = (self.proj_x[0][0], self.proj_y[0][0])
        proj_near_r = (self.proj_x[0][1], self.proj_y[0][1])
        proj_far_l = (self.proj_x[1][0], self.proj_y[1][0])
        proj_far_r = (self.proj_x[1][1], self.proj_y[1][1])
        proj_mid_point = (self.proj_mid_x, self.proj_mid_y)

        if ROLL_ENABLED:
            front_tl = apply_roll(front_tl, self.cx, self.cy)
            front_tr = apply_roll(front_tr, self.cx, self.cy)
            front_br = apply_roll(front_br, self.cx, self.cy)
            front_bl = apply_roll(front_bl, self.cx, self.cy)

            back_tl = apply_roll(back_tl, self.cx, self.cy)
            back_tr = apply_roll(back_tr, self.cx, self.cy)
            back_br = apply_roll(back_br, self.cx, self.cy)
            back_bl = apply_roll(back_bl, self.cx, self.cy)

            proj_far_l = apply_roll(proj_far_l, self.cx, self.cy)
            proj_far_r = apply_roll(proj_far_r, self.cx, self.cy)
            proj_near_r = apply_roll(proj_near_r, self.cx, self.cy)
            proj_near_l = apply_roll(proj_near_l, self.cx, self.cy)
            proj_mid_point = apply_roll(proj_mid_point, self.cx, self.cy)

        if RADIAL_DIST_ENABLED:
            front_tl = apply_radial_dist(front_tl, self.cx, self.cy)
            front_tr = apply_radial_dist(front_tr, self.cx, self.cy)
            front_br = apply_radial_dist(front_br, self.cx, self.cy)
            front_bl = apply_radial_dist(front_bl, self.cx, self.cy)

            back_tl = apply_radial_dist(back_tl, self.cx, self.cy)
            back_tr = apply_radial_dist(back_tr, self.cx, self.cy)
            back_br = apply_radial_dist(back_br, self.cx, self.cy)
            back_bl = apply_radial_dist(back_bl, self.cx, self.cy)

            proj_far_l = apply_radial_dist(proj_far_l, self.cx, self.cy)
            proj_far_r = apply_radial_dist(proj_far_r, self.cx, self.cy)
            proj_near_r = apply_radial_dist(proj_near_r, self.cx, self.cy)
            proj_near_l = apply_radial_dist(proj_near_l, self.cx, self.cy)
            proj_mid_point = apply_radial_dist(proj_mid_point, self.cx, self.cy)

        bbox_bottom = max(front_br[1], front_bl[1], back_br[1], back_bl[1])  # bbox_bottom  (p_b^y in paper)
        bbox_top = min(front_tr[1], front_tl[1], back_tr[1], back_tl[1])  # bbox_top  (p_t^y in paper)
        bbox_right = max(front_bl[0], back_bl[0], front_tl[0], back_tl[0], front_br[0], back_br[0], front_tr[0], back_tr[0])  # bbox_bottom  (p_b^y in paper)
        bbox_left = min(front_bl[0], back_bl[0], front_tl[0], back_tl[0], front_br[0], back_br[0], front_tr[0], back_tr[0])  # bbox_top  (p_t^y in paper)

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
        proj_points = [proj_far_l, proj_far_r, proj_near_r, proj_near_l]
        proj_mid_point = [proj_mid_point]

        front_facade = [front_tl, front_tr, front_br, front_bl]
        back_facade = [back_tl, back_tr, back_br, back_bl]
        top_facade = [front_tl, back_tl, back_tr, front_tr]

        return bbox_points, proj_points, proj_mid_point, front_facade, back_facade, top_facade


inputs = []
outputs = []
cam_width_heights = []
point_estimator = PointEstimator()

if DEMO_MODE:
    PAUSE_FIG_TIME = 0.50

cam_width_height = np.array((CAM_PIXEL_WIDTH, CAM_PIXEL_HEIGHT))

print('Sample collection is started!')

for x in tqdm(np.logspace(np.log10(X_SEARCH_MIN), np.log10(X_SEARCH_MAX), num=X_SEARCH_COUNT)):
    for y_hor in np.arange(Y_HOR_SEARCH_MIN, Y_HOR_SEARCH_MAX, SEARCH_DISTANCE_STEP):
        for y_ver in np.arange(Y_VER_SEARCH_MIN, X_VER_SEARCH_MAX, SEARCH_DISTANCE_STEP):
            rotate_angle = random.uniform(ROTATE_ANGLE_MIN, ROTATE_ANGLE_MAX)
            if DEMO_MODE:
                point_estimator.set_object_location(y_ver=random.uniform(Y_VER_SEARCH_MIN, X_VER_SEARCH_MAX),
                                                    y_hor=random.uniform(Y_HOR_SEARCH_MIN, Y_HOR_SEARCH_MAX),
                                                    x=random.uniform(X_SEARCH_MIN, X_SEARCH_MAX),
                                                    rotate_angle=rotate_angle)
            else:
                point_estimator.set_object_location(y_ver=y_ver, y_hor=y_hor, x=x, rotate_angle=rotate_angle)
            point_estimator.calc_vertical_point_loc()
            point_estimator.calc_horizontal_point_loc()
            bbox, proj, proj_mid, front, back, top = point_estimator.get_bbox_proj_points()
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
