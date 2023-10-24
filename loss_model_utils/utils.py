from configparser import ConfigParser
from typing import List, Tuple
import math
import random
import cv2.cv2 as cv2
import numpy as np
import torch


LINE_THICKNESS: int = 1  # Line thickness used for the drawing operations.
INVALID_PIXEL_LOC = -10000
SQUARE_RADIUS_LIMIT = 4
DISTORTION_COEFF_LIMIT = 0.5


class ImageDimensions:
    """
    Represents the dimensions (width and height) of an image.
    """
    def __init__(self, width: int, height: int):
        """
        Initializes a new ImageDimensions object.

        Parameters
        ----------
        width : int
            The width of the image.
        height : int
            The height of the image.
        """
        self.width = width
        self.height = height

    def __eq__(self, other: 'ImageDimensions') -> bool:
        """
        Compare this ImageDimensions object with another for equality.

        Parameters
        ----------
        other : ImageDimensions
            Another ImageDimensions object to compare with.

        Returns
        -------
        bool
            True if the dimensions are equal, False otherwise.
        """
        if isinstance(other, ImageDimensions):
            return self.width == other.width and self.height == other.height
        return False


def str2bool(bool_string: str) -> bool:
    """
    Convert strings to booleans

    Parameters
    ----------
    bool_string: str
        Boolean string value

    Returns
    -------
    dataset_pairs: bool
        Boolean value
    """
    if bool_string.lower() == "true":
        return True
    elif bool_string.lower() == "false":
        return False
    else:
        raise ValueError


def read_settings(path: str) -> ConfigParser:
    """
    Read settings from an ini file.

    Parameters
    ----------
    path: str
        Path to the settings file.

    Returns
    -------
    config: ConfigParser
        Object including the parameters set in the settings file
    """
    config = ConfigParser()
    config.read(path)
    return config


def seed_worker(worker_id):
    """
    To preserve reproducibility seed worker is used. Taken from (https://pytorch.org/docs/stable/notes/randomness.html)
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def connect_and_draw_points(input_image: np.ndarray, points: List[Tuple[float]], color: Tuple[int, int, int]):
    """
    Connects and draw points. If the points includes only one point then a marker placed to the point. Otherwise, the
    adjacent points are connected with line.

    Parameters
    ----------
    input_image: np.ndarray
        Input image
    points: List[Tuple[float]]
        Point to be used to draw on image
    color: Tuple[int]
        Color used in the drawings.

    """
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


def cotan(radians: float) -> float:
    """
    Calculate cotangent of an angle given in radians.

    Parameters
    ----------
    radians: float
        Value of the given angle in radians

    Returns
    -------
    cotan_value: float
        Cotangent value of the given angle

    """
    cotan_value = 1 / math.tan(radians)
    return cotan_value


def apply_radial_dist(point: np.ndarray, cx: float, cy: float, k_1: float, k_2: float) -> np.ndarray:
    """
    Transform a point in camera using radial distortion coefficients.

    Parameters
    ----------
    point: np.ndarray
        Location of the point in image space before the distortion operation.
    cx: float
        Principal point location in the horizontal axis of the camera
    cy: float
        Principal point location in the vertical axis of the camera
    k_1: float
        k1 value of radial distortion modeling
    k_2: float
        k2 value of radial distortion modeling

    Returns
    -------
    distorted_point: np.ndarray
        Location of the point in image space after applying the distortion.
    """
    distorted_point = np.array([INVALID_PIXEL_LOC, INVALID_PIXEL_LOC])
    norm_dist_x, norm_dist_y = (np.array(point) - np.array((cx, cy))) / np.array((cx, cy))
    r2 = norm_dist_x ** 2 + norm_dist_y ** 2
    if r2 < SQUARE_RADIUS_LIMIT:
        distortion_coeff_x = 1 + k_1 * r2 + k_2 * r2 ** 2
        distortion_coeff_y = 1 + k_1 * r2 + k_2 * r2 ** 2
        if distortion_coeff_x > DISTORTION_COEFF_LIMIT and distortion_coeff_y > DISTORTION_COEFF_LIMIT:
            distorted_point = np.array((norm_dist_x * distortion_coeff_x, norm_dist_y * distortion_coeff_y)) * np.array(
                (cx, cy)) + np.array((cx, cy))
    return distorted_point
