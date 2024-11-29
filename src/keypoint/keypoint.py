import numpy as np
from typing import List
from scipy.ndimage import gaussian_filter, sobel, uniform_filter

def corner_finder(
    intensity: np.ndarray,
    smoothing_variance: int,
    k: float = 0.04,
) -> np.ndarray:
    '''
    Implementation of Harris & Stephens / Shi–Tomasi corner detection algorithm
    ref1: https://en.wikipedia.org/wiki/Corner_detection
    ref2: https://www.ipb.uni-bonn.de/html/teaching/photo12-2021/2021-pho1-10-features-keypoints.pptx.pdf

    Params:
        intensity - Grayscale intensity image
        smoothing_variance - variance of guassian filter to apply to image
        threshold - threshold for determining if a point is a corner
    Returns
        2d list of corner pixels
    '''
    smoothed_intensity = gaussian_filter(intensity, sigma=smoothing_variance)
    Jx = sobel(smoothed_intensity, axis=0)
    Jy = sobel(smoothed_intensity, axis=1)
    Jx2 = Jx * Jx
    Jxy = Jx * Jy
    Jy2 = Jy * Jy

    # TODO: investigate using a guassian filter
    Jx2_weighted_avg = uniform_filter(Jx2)
    Jxy_weighted_avg = uniform_filter(Jxy)
    Jy2_weighted_avg = uniform_filter(Jy2)

    height, width = intensity.shape
    corner_scores = np.zeros(intensity.shape)
    for i in range(height):
        for j in range(width):
            M = np.array([
                [Jx2_weighted_avg[i,j], Jxy_weighted_avg[i,j]],
                [Jxy_weighted_avg[i,j], Jy2_weighted_avg[i,j]],
            ])
            corner_scores[i,j] = np.linalg.det(M) - k * (np.trace(M)**2)
    return corner_scores
