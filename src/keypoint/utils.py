import numpy as np
from typing import List
from PIL import Image

# RGB to grayscale factors as defined in CCIR 601
# ref: https://en.wikipedia.org/wiki/Luma_(video)#Rec._601_luma_versus_Rec._709_luma_coefficients
RED_FACTOR = 0.2989
GREEN_FACTOR = 0.5870
BLUE_FACTOR = 0.1140

def rgb_to_intensity(image_fname: str) -> np.ndarray:
    '''
    Convert RGB image to grayscale intensity matrix

    Params:
        image_fname - image filename
    Returns:
        I - intensity ndarray
    '''
    image = Image.open(image_fname)
    data = np.asarray(image, dtype="int32" )
    return RED_FACTOR * data[:,:,0] + GREEN_FACTOR * data[:,:,1] + BLUE_FACTOR * data[:,:,2]