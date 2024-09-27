import json
import pandas as pd
import numpy as np
from typing import Tuple

def read_dlt_points(world_file_name: str, view_file_name: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
    '''
    Reads world coordinates and pixel coordinates for a view of an object. Joins the two lists on point ID.
    Also extracts width and height of the image.

    Params:
        world_file_name - json file containing world coordinates of a set of points
        view_file_name - json file containing pixel coordinates of a set of points

    Returns:
        world_positions - list of points in world coordinates
        view_positions - list of points in pixel coordinates
        width - width of image
        height - height of image
    '''
    with open(world_file_name) as json_data:
        world_data = json.load(json_data)
        world_df = pd.DataFrame(world_data['points'])
    with open(view_file_name) as json_data:
        view_data = json.load(json_data)
        width = view_data['width']
        height = view_data['height']
        view_df = pd.DataFrame(view_data['points'])
    
    joined_df = world_df.join(view_df.set_index('name'), on='name')
    joined_df = joined_df[joined_df['coords'].notnull()]
    world_positions = np.array(joined_df['world_pos'].to_list())
    view_positions = np.array(joined_df['coords'].to_list())
    return world_positions, view_positions, width, height
