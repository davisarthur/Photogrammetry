import numpy as np
from ..data import utils
from . import viewing_transforms
from typing import Tuple
from scipy.spatial.transform import Rotation

def dlt(world_file_name: str, view_file_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Implementation of the direct linear transform. Takes in one file defining world coordinates of a set of points,
    and one file containing pixel coordinates of the same points. Determines the orientation, position, and intrinsic
    characteristics of the camera.

    ref 1: https://www.ipb.uni-bonn.de/html/teaching/photo12-2021/2021-pho1-21-DLT.pptx.pdf
    ref 2: https://engineering.purdue.edu/CCE/Academics/Groups/Geomatics/DPRG/Courses_Materials/Photogrammetry_2019Fall/AKAM_DLT_CV_AKAM

    Params:
        world_file_name - json file containing world coordinates of a set of points
        view_file_name - json file containing pixel coordinates of a set of points

    Returns:
        x_0 - world position of camera
        K - intrinsic transform of camera
        R - rotation transform of camera
    '''
    # Read data, normalize pixel coords
    positions, pixels, width, height = utils.read_points(world_file_name, view_file_name)
    vp_pixels = viewing_transforms.inverse_viewport_transform(pixels, width, height)

    # Estimate projection matrix, L
    M = buildM(positions, vp_pixels)
    U, s, V = np.linalg.svd(M)
    L = np.array(V[-1]).reshape(3, 4)

    # Extract initial position from projection matrix
    H = L[:,:3]
    x_0 = -np.linalg.inv(H) @ L[:,3]

    # Extract extrensic (R) and intrinsic (K) transforms
    R_T, K_unnormalized_inv = np.linalg.qr(np.linalg.inv(H))
    pi_shift = Rotation.from_euler('xyz', (0, 0, np.pi)).as_matrix()
    R = pi_shift @ R_T.T
    K_unnormalized = np.linalg.inv(K_unnormalized_inv)
    K = K_unnormalized @ pi_shift / K_unnormalized[2,2]
    return x_0, K, R

def buildM(positions: np.array, pixels: np.array) -> np.array:
    assert positions.shape[0] == pixels.shape[0], 'pixel and points lists are of unequal length'
    n = positions.shape[0]
    output = np.zeros((n*2, 12))
    for i in range(n):
        pos = positions[i]
        pixel = pixels[i]
        output[i*2] = ax(pos, pixel[0])
        output[i*2+1] = ay(pos, pixel[1])
    return output 
        
def ax(pos: np.array, pixel_x: float | int) -> np.array:
    output = np.zeros(12)
    output[:3] = -pos
    output[3] = -1
    output[8:11] = pixel_x * pos
    output[11] = pixel_x
    return output

def ay(pos: np.array, pixel_y: float | int) -> np.array:
    output = np.zeros(12)
    output[4:7] = -pos
    output[7] = -1
    output[8:11] = pixel_y * pos
    output[11] = pixel_y
    return output
