import numpy as np

def inverse_viewport_transform(pixels: np.array, width: int, height: int) -> np.ndarray:
    '''
    Normalizes pixel coordinates to fall within the range [-1, 1]

    pixels - (n, 2) array of pixels
    width - width of image in pixels
    height - height of image in pixels
    '''
    n = pixels.shape[0]
    transform = np.array([
        [2 / (width - 1), 0, -1],
        [0, 2 / (height - 1), -1],
        [0, 0, 1],
    ])
    output = np.zeros((n, 3))
    for i in range(n):
        output[i] = transform @ np.array([pixels[i][0], pixels[i][1], 1])
    return output[:,:2]