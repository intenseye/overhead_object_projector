import sys
import os
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

DEMO_MODE: bool = False  # Enables the demo mode. In demo mode the random located object are drawn along with projection.
DRAW_ENABLED: bool = False  # Enables the drawing mode
FIXED_SEED: bool = True  # Used to fix seed for deterministic results
FIXED_SEED_NUM: int = 35  # Seed number
LINE_THICKNESS: int = 1  # Line thickness used for the drawing operations.
GREEN_COLOR = (0, 255, 0)  # Green color code
YELLOW_COLOR = (255, 255, 0)  # Yellow color code
RED_COLOR = (255, 0, 0)  # Red color code
WHITE_COLOR = (255, 255, 255)  # White color code
PAUSE_FIG_TIME = 0.01  # Delay time applied during the consecutive drawings

CAM_FOV_HOR = 60.0  # Horizontal field-of-view (in degrees)
CAM_PIXEL_WIDTH: int = 1280  # number of horizontal pixel
CAM_PIXEL_HEIGHT: int = 720  # number of vertical pixel
HALF_PIXEL_SIZE = 1 / 2
CAM_REGION_EXPAND_RATIO = 3.0  # Expanding ratio of original image. Odd integer number is advised.
RADIAL_DIST_ENABLED: bool = True
K_1 = -0.05  # k1 parameter used for radial distribution
K_2 = 0.0  # k2 parameter used for radial distribution
RANDOM_DEVIATION_ENABLED: bool = True  # Enables random deviation for each edge od bounding box of the object.
DEVIATON_SIGMA = 1.0  # Enables random deviation for each edge od bounding box of the object.
ROTATE_ANGLE_MIN = -10.0  # Random rotation angle lower limit (in degrees)
ROTATE_ANGLE_MAX = 10.0  # Random rotation angle upper limit (in degrees)

YAW_CAM = -15.0  # rotation around y dimension (use rhr for positive dimension) (in degrees)
PITCH_CAM = -10.0  # rotation around x dimension (use rhr positive dimension) (in degrees)
ROLL_CAM = 0.0  # rotation around z-dimension (use rhr for positive dimension) (in degrees)

CAM_Z = 0.0  # position of the camera in z-dimension (in meters)
CAM_Y = -8.0  # position of the camera in y-dimension (in meters)
CAM_X = -5.0  # position of the camera in x-dimension (in meters)

OBJ_HEIGHT = 2.5  # height of the object (in y-dimension)
OBJ_WIDTH = 2.0  # width of the object (in z-dimension)
OBJ_LENGTH = 1.0  # length of the object (in x-dimension)

Z_SEARCH_MIN = 20.0  # lower limit for the position of the object center in z-dimension (in meters)
Z_SEARCH_MAX = 50.0  # upper limit for the position of the object center in z-dimension (in meters)
Z_SEARCH_COUNT = 50  # number of step points while searching the z dimension
X_SEARCH_MIN = -40.0  # lower limit for the position of the object center in x-dimension (in meters)
X_SEARCH_MAX = 40.0  # Upper limit for the position of the object center in x-dimension (in meters)
# In y dimension minus denotes above floor.
Y_SEARCH_MIN = -1.25  # lower limit for the position of the object center in x-dimension (in meters)
Y_SEARCH_MAX = -10.0  # lower limit for the position of the object center in x-dimension (in meters)
SEARCH_DISTANCE_STEP = 0.5  # lower limit for the position of the object center in x-dimension (in meters)

EXPORT: bool = True  # Enables data exporting
PATH_TO_OUTPUT_FOLDER = '/home/poyraz/intenseye/input_outputs/overhead_object_projector'  # path to output folder
PROJECTION_DATA_PATH = 'inputs_outputs_corrected.txt'  # relative path to projection data
AUXILIARY_DATA_PATH = 'auxiliary_data_corrected.pickle'  # relative path to auxiliary data


def connect_and_draw_points(input_image, points, color):
    '''
    Connects and draw points. If the points includes only one point then a marker placed to the point. Otherwise, the
    adjacent points are connected with line.
    '''
    if len(points) == 1:
        cv2.drawMarker(input_image, np.round(points[0]).astype(int), color=color, markerType=cv2.MARKER_STAR,
                       thickness=LINE_THICKNESS)
    elif len(points) > 1:
        for i in range(len(points)):
            if i == len(points) - 1:
                cv2.line(input_image, np.round(points[i]).astype(int), np.round(points[0]).astype(int), color=color,
                         thickness=LINE_THICKNESS)
            else:
                cv2.line(input_image, np.round(points[i]).astype(int), np.round(points[i + 1]).astype(int), color=color,
                         thickness=LINE_THICKNESS)
    else:
        print('Given point list is empty!')


def cotan(radians):
    '''
    Calculate cotangent of an angle given in radians.
    '''
    return 1 / math.tan(radians)


def apply_radial_dist(point, cx, cy):
    '''
    Transform a point in camera using radial distortion coefficients.
    '''
    norm_dist_x, norm_dist_y = (np.array(point) - np.array((cx, cy))) / np.array((cx, cy))
    r2 = norm_dist_x ** 2 + norm_dist_y ** 2
    distorted_point = np.array([-10000, -10000])
    distortion_coeff_x = 1 + K_1 * r2 + K_2 * r2**2
    distortion_coeff_y = 1 + K_1 * r2 + K_2 * r2 ** 2
    if distortion_coeff_x > 0 and distortion_coeff_y > 0:
        distorted_point = np.array((norm_dist_x * distortion_coeff_x, norm_dist_y * distortion_coeff_y)) * np.array(
            (cx, cy)) + np.array((cx, cy))
    return distorted_point


def calculate_input_coord(bbox, proj_mid):
    '''
    Calculates input and target data points to be used in the model.
    '''
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

        self.cx = CAM_PIXEL_WIDTH / 2  # horizontal center pixel loc.
        self.cy = CAM_PIXEL_HEIGHT / 2  # vertical center pixel loc.

        self.extended_height = int(CAM_REGION_EXPAND_RATIO * CAM_PIXEL_HEIGHT)
        self.extended_width = int(CAM_REGION_EXPAND_RATIO * CAM_PIXEL_WIDTH)
        self.extended_y_start = -int((CAM_REGION_EXPAND_RATIO - 1) * CAM_PIXEL_HEIGHT / 2)
        self.extended_x_start = -int((CAM_REGION_EXPAND_RATIO - 1) * CAM_PIXEL_WIDTH / 2)
        self.top_left_pixel_coord = np.array([self.extended_x_start, self.extended_y_start])
        self.pixel_world_coords = np.ones((self.extended_height, self.extended_width, 3), dtype=float) * sys.float_info.max

        self.calc_projection_matrix()
        self.calc_pixel_world_coordinates()

    def calc_camera_calibration_matrix(self):
        '''
        Calculate camera calibration matrix K
        '''
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
        '''
        Calculates projection matrix between world coordinate and camera pixel coordinates.
        '''
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

        yaw = math.radians(YAW_CAM)
        pitch = math.radians(PITCH_CAM)
        roll = math.radians(ROLL_CAM)
        # Camera rotation matrix [R]
        R = self.calc_rotation_matrix([yaw, pitch, roll], ['yaw', 'pitch', 'roll'])

        # Coordinates of the camera centre (camera center is selected as the origin of the world coordinate system)
        cam_C = np.array([[0.0], [0.0], [0.0]])

        # P = KR[I | −C]
        self.P = np.matmul(np.matmul(K, R), np.concatenate((np.eye(3), -cam_C), axis=1))

        # P^+ = P^T(PP^T)^−1, for which PP^+ = I
        self.P_inv = np.matmul(np.transpose(self.P), np.linalg.inv(np.matmul(self.P, np.transpose(self.P))))

    def project_pixel_coord(self, relative_object_loc):
        '''
        Projects from world coordinate system into pixel coordinate system.
        '''
        # Project into pixel coordinate system
        x_pixel = np.matmul(self.P, np.array(relative_object_loc))
        # Normalize by z to get real pixel coordinates
        x_norm = np.squeeze((x_pixel / x_pixel[2][0])[:-1])
        return x_norm

    def calc_pixel_world_coordinates(self):
        '''
        Back projects from pixel coordinate into fixed elevation world coordinate system.
        '''
        for ind_j, j in enumerate(range(self.extended_y_start, self.extended_height + self.extended_y_start)):
            for ind_i, i in enumerate(range(self.extended_x_start, self.extended_width + self.extended_x_start)):
                unit_ray_loc = np.matmul(self.P_inv, np.array([i, j, 1]))  # TODO: We may need to send ray from np.array([i+0.5, j+0.5, 1]
                if unit_ray_loc[1] > 0:  # means ray is traveling in downward direction
                    unit_ray_multiplier = -CAM_Y / unit_ray_loc[1]  # Projected elevation is 0 which means -CAM_Y for camera centered coordinate system.
                    self.pixel_world_coords[ind_j, ind_i, :] = unit_ray_multiplier * unit_ray_loc[:3] + np.array([CAM_X, CAM_Y, CAM_Z])

    def get_relative_world_coords(self, object_x, object_y, object_z, rotate_angle):
        '''
        Gets the world coordinate system points of the object and projection of it w.r.t camera centered origin and object rotation.
        '''
        rotate_angle = math.radians(rotate_angle)
        rot_cos = math.cos(rotate_angle)
        rot_sin = math.sin(rotate_angle)

        rot_length_inc = (OBJ_LENGTH / 2) * rot_cos + (OBJ_WIDTH / 2) * rot_sin
        rot_length_dec = (OBJ_LENGTH / 2) * rot_cos - (OBJ_WIDTH / 2) * rot_sin

        rot_width_inc = (OBJ_WIDTH / 2) * rot_cos + (OBJ_LENGTH / 2) * rot_sin
        rot_width_dec = (OBJ_WIDTH / 2) * rot_cos - (OBJ_LENGTH / 2) * rot_sin

        object_rel_top = [object_y - OBJ_HEIGHT / 2 - CAM_Y]
        object_rel_bottom = [object_y + OBJ_HEIGHT / 2 - CAM_Y]

        self.object_tlf = [[object_x - rot_length_inc - CAM_X], object_rel_top, [object_z - rot_width_dec - CAM_Z], [1]]
        self.object_tlb = [[object_x - rot_length_dec - CAM_X], object_rel_top, [object_z + rot_width_inc - CAM_Z], [1]]
        self.object_trf = [[object_x + rot_length_dec - CAM_X], object_rel_top, [object_z - rot_width_inc - CAM_Z], [1]]
        self.object_trb = [[object_x + rot_length_inc - CAM_X], object_rel_top, [object_z + rot_width_dec - CAM_Z], [1]]

        self.object_blf = [[object_x - rot_length_inc - CAM_X], object_rel_bottom, [object_z - rot_width_dec - CAM_Z], [1]]
        self.object_blb = [[object_x - rot_length_dec - CAM_X], object_rel_bottom, [object_z + rot_width_inc - CAM_Z], [1]]
        self.object_brf = [[object_x + rot_length_dec - CAM_X], object_rel_bottom, [object_z - rot_width_inc - CAM_Z], [1]]
        self.object_brb = [[object_x + rot_length_inc - CAM_X], object_rel_bottom, [object_z + rot_width_dec - CAM_Z], [1]]

        self.proj_lf = [[object_x - rot_length_inc - CAM_X], [0 - CAM_Y], [object_z - rot_width_dec - CAM_Z], [1]]
        self.proj_lb = [[object_x - rot_length_dec - CAM_X], [0 - CAM_Y], [object_z + rot_width_inc - CAM_Z], [1]]
        self.proj_rf = [[object_x + rot_length_dec - CAM_X], [0 - CAM_Y], [object_z - rot_width_inc - CAM_Z], [1]]
        self.proj_rb = [[object_x + rot_length_inc - CAM_X], [0 - CAM_Y], [object_z + rot_width_dec - CAM_Z], [1]]
        self.proj_center = [[object_x - CAM_X], [0 - CAM_Y], [object_z - CAM_Z], [1]]

    def get_pixel_points(self):
        '''
        Gets the pixel coordinate system points of the object and projection of it.
        '''
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

        # Find limits of the bbox
        bbox_bottom = max(point_o_blf[1], point_o_blb[1], point_o_brf[1], point_o_brb[1])  # TODO not sure that in between points can get maximum value. If not we may need to prove it.
        bbox_top = min(point_o_tlf[1], point_o_tlb[1], point_o_trf[1], point_o_trb[1])
        bbox_right = max(point_o_tlf[0], point_o_tlb[0], point_o_trf[0], point_o_trb[0], point_o_blf[0], point_o_blb[0], point_o_brf[0], point_o_brb[0])
        bbox_left = min(point_o_tlf[0], point_o_tlb[0], point_o_trf[0], point_o_trb[0], point_o_blf[0], point_o_blb[0], point_o_brf[0], point_o_brb[0])

        bbox_tl = (bbox_left, bbox_top)
        bbox_tr = (bbox_right, bbox_top)
        bbox_br = (bbox_right, bbox_bottom)
        bbox_bl = (bbox_left, bbox_bottom)

        # Add random deviation to the location of each edge of the bounding box to simulate the error of object detector
        if RANDOM_DEVIATION_ENABLED:
            top_dev = random.gauss(0, DEVIATON_SIGMA)
            bottom_dev = random.gauss(0, DEVIATON_SIGMA)
            left_dev = random.gauss(0, DEVIATON_SIGMA)
            right_dev = random.gauss(0, DEVIATON_SIGMA)

            bbox_tl = [bbox_tl[0] + left_dev, bbox_tl[1] + top_dev]
            bbox_tr = [bbox_tr[0] + right_dev, bbox_tr[1] + top_dev]
            bbox_br = [bbox_br[0] + right_dev, bbox_br[1] + bottom_dev]
            bbox_bl = [bbox_bl[0] + left_dev, bbox_bl[1] + bottom_dev]

        bbox_points = [bbox_tl, bbox_tr, bbox_br, bbox_bl]  # TODO: We may need to convert to integer to simulate object detector. (Is there any floating point bbox detector in literature?)

        proj_points = [point_p_lb, point_p_rb, point_p_rf, point_p_lf]
        proj_mid_point = [point_p_cen]  # TODO: Not sure that we need to convert to integer. I believe we are able to select a floating point GT using a tool.

        # Gets the facade points for visual representation
        front_facade = [point_o_tlf, point_o_trf, point_o_brf, point_o_blf]
        back_facade = [point_o_tlb, point_o_trb, point_o_brb, point_o_blb]
        top_facade = [point_o_tlf, point_o_tlb, point_o_trb, point_o_trf]

        return bbox_points, proj_points, proj_mid_point, front_facade, back_facade, top_facade


inputs = []
outputs = []
cam_width_heights = []
point_estimator = PointEstimatorProjection()

if DEMO_MODE:  # For the demo mode delay time is increased for the sake of easy monitoring
    PAUSE_FIG_TIME = 0.50

cam_width_height = np.array((CAM_PIXEL_WIDTH, CAM_PIXEL_HEIGHT))
print('Sample collection is started!')

for z in tqdm(np.logspace(np.log10(Z_SEARCH_MIN), np.log10(Z_SEARCH_MAX), num=Z_SEARCH_COUNT)):
    # Logarithmic search procedure is applied so that distant observations in z-dimension do not dominate the sample set.
    for x in np.arange(X_SEARCH_MIN, X_SEARCH_MAX, SEARCH_DISTANCE_STEP):
        for y in np.arange(max(-Y_SEARCH_MIN, OBJ_HEIGHT / 2), -Y_SEARCH_MAX, SEARCH_DISTANCE_STEP):
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
            # Filter the sample set in a such way that both object and center of the projection are completely visible
            # in the camera.
            if (0 <= (mid_bottom_coord[0] - bbox_width_height[0] / 2)) and (
                    (mid_bottom_coord[0] + bbox_width_height[0] / 2) < CAM_PIXEL_WIDTH) and (
                    mid_bottom_coord[1] - bbox_width_height[1] >= 0) and (
                    mid_bottom_coord[1] < CAM_PIXEL_HEIGHT) and (
                    mid_bottom_coord[1] + proj_coord_offset[1] < CAM_PIXEL_HEIGHT):
                if not DEMO_MODE:  # For the visual demo mode the storing operation is skipped.
                    input_sample = np.append(mid_bottom_coord, bbox_width_height)
                    inputs.append(input_sample)
                    outputs.append(proj_coord_offset)
                    cam_width_heights.append(cam_width_height)

                if DEMO_MODE or DRAW_ENABLED:  # Printing of the results
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

if (not DEMO_MODE) and EXPORT:
    export_data = np.hstack((inputs, outputs, cam_width_heights))
    dir_path = PATH_TO_OUTPUT_FOLDER
    path_to_auxiliary_data = os.path.join(dir_path, AUXILIARY_DATA_PATH)
    path_to_projection_data = os.path.join(dir_path, PROJECTION_DATA_PATH)

    os.makedirs(dir_path, exist_ok=True)
    with open(path_to_auxiliary_data, 'wb') as handle:
        pickle.dump([point_estimator.pixel_world_coords, point_estimator.top_left_pixel_coord], handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.savetxt(path_to_projection_data, export_data, header='x_coord_mid_bottom y_coord_mid_bottom bbox_width bbox_height proj_x_dist_to_mid_bottom proj_y_dist_to_mid_bottom cam_width cam_height', fmt='%1.6e')  # X is an array
    print('\nSample production is completed with {:6d} samples!'.format(export_data.shape[0]))
