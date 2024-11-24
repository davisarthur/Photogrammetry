import numpy as np

def sift(window: np.ndarray) -> np.ndarray:
    '''
    Scale invariate feature transform. Ref: https://en.wikipedia.org/wiki/Scale-invariant_feature_transform

    Params: 
        window - (18, 18) window from image, keypoint is located at (9,9).

    Retuns:
        feature vector (128 dimensions)
    '''
    up = window[:-2,1:-1]
    down = window[2:,1:-1]
    left = window[1:-1,:-2]
    right = window[1:-1,2:]
    guassian = uniform_2d_guassian(16, 16, 8)
    magnitudes = np.sqrt((up - down)**2 + (right - left)**2)
    weighted_magnitudes = guassian * magnitudes
    theta = np.arctan2(up - down, right - left)
    theta = theta - theta[9,9]

    # loop over each 4 x 4 section
    output = np.zeros(128)
    for i in range(4):
        for j in range(4):
            weights = weighted_magnitudes[i*4:(i+1)*4,j*4:(j+1)*4]
            angles = theta[i*4:(i+1)*4,j*4:(j+1)*4]
            bins = np.digitize(angles.flatten(), np.linspace(0, np.pi, 8))
            for weight, bin in zip(weights.flatten(), bins):
                output[(i*4+j)*8+bin-1] += weight
    return output

def uniform_2d_guassian(x_scale: int, y_scale: int, sigma: float):
    '''
    2D univariate radial guassian distribution
    '''
    output = np.zeros((y_scale, x_scale))
    mid = np.array([x_scale/2, y_scale/2])
    for j in range(y_scale):
        for i in range(x_scale):
            pos = np.array([j, i])
            dist = np.linalg.norm(pos - mid)
            output[j,i] = guassian_distribution(dist, sigma, 0)
    return output

def guassian_distribution(x: float, sigma: float, mean: float) -> float:
    '''
    Radial guassian distribution

    Params:
        x - point at which to evaluate distribution
        sigma - standard deviation
        mean

    Return:
        value of distribution at point x
    '''
    return np.exp(-(x - mean)**2 / (2 * sigma**2)) / (2 * np.pi * sigma**2.0)
