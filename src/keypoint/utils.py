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

def clean_image(image: np.ndarray):
    n, m = image.shape
    visited = np.zeros(image.shape, dtype=int)
    cleaned_image = np.array(image)

    def dfs(x, y, members):
        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            if visited[y, x]:
                continue
            visited[y, x] = 1
            if image[y, x] != 0:
                members.append([x, y, image[y, x]])
            else:
                continue
            for new_x in range(x-1, x+2):
                for new_y in range(y-1, y+2):
                    if new_y < 0 or new_y >= n or new_x < 0 or new_x >= m or visited[new_y, new_x]:
                        continue
                    stack.append((new_x, new_y))

    for j in range(n):
        for i in range(m):
            if image[j, i] and (not visited[j, i]):
                members = []
                dfs(i, j, members)
                sorted_blob = sorted(members, key=lambda x: x[2])
                for i in range(len(members)-1):
                    cleaned_image[sorted_blob[i][1], sorted_blob[i][0]] = 0
    return cleaned_image
