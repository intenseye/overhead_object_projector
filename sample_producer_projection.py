import math
import numpy as np
import cv2
import matplotlib.pyplot as plt


LINE_THICKNESS = 1
GREEN_COLOR = (0, 255, 0)
YELLOW_COLOR = (255, 255, 0)
RED_COLOR = (255, 0, 0)
WHITE_COLOR = (255, 255, 255)
PAUSE_FIG_TIME = 0.01

CAM_PIXEL_WIDTH = 1280  # number of horizontal pixel
CAM_PIXEL_HEIGHT = 720  # number of vertical pixel
CAM_FOV_HOR = 110  # in degrees

YAW_CAM = -45  # y dimension positive is clockwise (in degrees)
PITCH_CAM = -10  # x dimension positive is upward (in degrees)
ROLL_CAM = 0  # z dimension positive is clockwise (in degrees)

OBJ_HEIGHT = 2.5
OBJ_WIDTH = 2.0
OBJ_LENGTH = 1.0

OBJECT_Z = 10
OBJECT_Y = 5
OBJECT_X = 0


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


def calc_camera_calibration_matrix():
    focal_length = cotan(math.radians(CAM_FOV_HOR / 2)) * (CAM_PIXEL_WIDTH / 2)
    # Cameras intrinsic matrix [K]
    K = np.array(
        [[focal_length, 0., CAM_PIXEL_WIDTH / 2],
         [0., focal_length, CAM_PIXEL_HEIGHT / 2],
         [0., 0., 1.]]
    )
    return K


def get_rot_x(angle):
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


def get_rot_y(angle):
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


def get_rot_z(angle):
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


def calc_rotation_matrix(angles, order):
    '''
    Calculates rotation martix by considering angles in the given order
    Note: The rotation is calculated in clockwise direction (use right-hand-rule)
    '''
    rot_map = {'pitch': get_rot_x, 'yaw': get_rot_y, 'roll': get_rot_z}
    R = np.eye(3)
    for angle, axis in list(zip(angles, order))[::-1]:
        R_axis = rot_map.get(axis)
        R = np.matmul(R, R_axis(-angle))
    return R


def calc_projection_matrix():
    K = calc_camera_calibration_matrix()
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
    R = calc_rotation_matrix([yaw, pitch, roll], ['yaw', 'pitch', 'roll'])

    # Coordinates of the camera centre
    camC = np.array([[0.0], [0.0], [0.0]])

    P = np.matmul(np.matmul(K, R), np.concatenate((np.eye(3), -camC), axis=1))
    return P


def calc_pixel_coord(relative_object_loc, projection_matrix):
    # Project (Xw, Yw, Zw, 0) into pixel coordinate system
    p_pixel = np.matmul(projection_matrix, np.array(relative_object_loc))
    # Normalize by z to get (u,v,1)
    p_norm = np.squeeze((p_pixel / p_pixel[2][0])[:-1])
    return p_norm

P = calc_projection_matrix()

object_left = [OBJECT_X - OBJ_LENGTH / 2]
object_right = [OBJECT_X + OBJ_LENGTH / 2]

object_top = [OBJECT_Y - OBJ_HEIGHT / 2]
object_bottom = [OBJECT_Y + OBJ_HEIGHT / 2]

object_front = [OBJECT_Z - OBJ_WIDTH / 2]
object_back = [OBJECT_Z + OBJ_WIDTH / 2]

object_tlf = [object_left, object_top, object_front, [1]]
object_tlb = [object_left, object_top, object_back, [1]]
object_trf = [object_right, object_top, object_front, [1]]
object_trb = [object_right, object_top, object_back, [1]]

object_blf = [object_left, object_bottom, object_front, [1]]
object_blb = [object_left, object_bottom, object_back, [1]]
object_brf = [object_right, object_bottom, object_front, [1]]
object_brb = [object_right, object_bottom, object_back, [1]]

proj_lf = [object_left, [15], object_front, [1]]
proj_lb = [object_left, [15], object_back, [1]]
proj_rf = [object_right, [15], object_front, [1]]
proj_rb = [object_right, [15], object_back, [1]]
proj_center = [[OBJECT_X], [15], [OBJECT_Z], [1]]

point_o_tlf = calc_pixel_coord(object_tlf, P)
point_o_tlb = calc_pixel_coord(object_tlb, P)
point_o_trf = calc_pixel_coord(object_trf, P)
point_o_trb = calc_pixel_coord(object_trb, P)
point_o_blf = calc_pixel_coord(object_blf, P)
point_o_blb = calc_pixel_coord(object_blb, P)
point_o_brf = calc_pixel_coord(object_brf, P)
point_o_brb = calc_pixel_coord(object_brb, P)

point_p_lf = calc_pixel_coord(proj_lf, P)
point_p_lb = calc_pixel_coord(proj_lb, P)
point_p_rf = calc_pixel_coord(proj_rf, P)
point_p_rb = calc_pixel_coord(proj_rb, P)
point_p_cen = calc_pixel_coord(proj_center, P)

proj_points = [point_p_lb, point_p_rb, point_p_rf, point_p_lf]
proj_mid_point = [point_p_cen]

front_facade = [point_o_tlf, point_o_trf, point_o_brf, point_o_blf]
back_facade = [point_o_tlb, point_o_trb, point_o_brb, point_o_blb]
top_facade = [point_o_tlf, point_o_tlb, point_o_trb, point_o_trf]

image = np.zeros((CAM_PIXEL_HEIGHT, CAM_PIXEL_WIDTH, 3), dtype=np.uint8)
connect_and_draw_points(image, proj_points, YELLOW_COLOR)
connect_and_draw_points(image, proj_mid_point, RED_COLOR)
connect_and_draw_points(image, front_facade, WHITE_COLOR)
connect_and_draw_points(image, back_facade, WHITE_COLOR)
connect_and_draw_points(image, top_facade, WHITE_COLOR)
plt.imshow(image)
plt.pause(PAUSE_FIG_TIME)
plt.cla()

print('Done')

