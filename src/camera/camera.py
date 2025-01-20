import cv2 as cv
import logging
import numpy as np
from typing import List, Tuple

DEFAULT_ACCURACY_EPSILON = 0.001
DEFAULT_MAX_ITERATIONS = 30
DEFAULT_TERMINATION_CRITERIA = (
    cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_ACCURACY_EPSILON
)
DEFAULT_SEARCH_WINDOW = (5, 5)
DEFAULT_ZERO_ZONE = (-1, -1)

logger = logging.getLogger(__name__)


def chessboard_camera_calibration(
    height: int,
    width: int,
    image_fnames: List[str],
    criteria: Tuple[int, int, float] = DEFAULT_TERMINATION_CRITERIA,
    search_window: Tuple[int, int] = DEFAULT_SEARCH_WINDOW,
    zero_zone: Tuple[int, int] = DEFAULT_ZERO_ZONE,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    if len(image_fnames) == 0:
        raise Exception('No images provided for calibration')

    objp = np.zeros((height*width, 3), np.float32)
    objp[:,:2] = np.mgrid[:height,:width].T.reshape(-1,2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    img_size = None
    for fname in image_fnames:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # TODO: Verify the image size is consistent
        img_size = gray.shape[::-1]
        ret, corners = cv.findChessboardCorners(gray, (height, width), None)
        if not ret:
            raise Exception(f'Failed to find chessboard corners in image {fname}')
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, search_window, zero_zone, criteria)
        imgpoints.append(corners2)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    if not ret:
        raise Exception('Failed to calibrate camera')
    return mtx, dist, rvecs, tvecs
