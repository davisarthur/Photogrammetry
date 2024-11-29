import streamlit as st
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import logging
import numpy as np
from keypoint import utils, keypoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if 'corner_scores' not in st.session_state:
    st.session_state.corner_scores = {}
if 'keypoints' not in st.session_state:
    st.session_state.keypoints = {}

logger.info('Starting the application')

# List all files in the images directory
image_dir = 'images'
valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith(valid_extensions)]

# Create a dropdown to select a file
selected_image = st.selectbox('Select an image file', image_files)

# Display the selected image
if selected_image:
    logger.info(f'Selected image: {selected_image}')
    image_path = os.path.join(image_dir, selected_image)
    img = mpimg.imread(image_path)

    # Get image dimensions
    height, width = img.shape[:2]

    # Compute corner scores
    if st.button('Compute corner scores'):
        logger.info(f'Computing keypoints for {selected_image}')
        intensity = utils.rgb_to_intensity(image_path)
        if image_path not in st.session_state.corner_scores:
            st.session_state.corner_scores[image_path] = keypoint.corner_finder(intensity, smoothing_variance=1.0)

    # Filter keypoints
    if st.session_state.corner_scores.get(image_path) is not None:
        threshold = st.slider('Set threshold', min_value=int(1E7), max_value=int(1E8), value=int(1E7), format="%d", step=1)
        logger.info(f'Filtering keypoints with threshold {threshold}')
        corner_scores = st.session_state.corner_scores[image_path]
        thresholded_corner_scores = np.where(corner_scores > threshold, 1, 0)
        cleaned_thresholded_corners = utils.clean_image(thresholded_corner_scores)
        st.session_state.keypoints[image_path] = np.argwhere(cleaned_thresholded_corners > 0)

    # Plot the image with a dot at the selected pixel
    fig, ax = plt.subplots()
    ax.imshow(img)
    if st.session_state.keypoints.get(image_path) is not None:
        keypoints = st.session_state.keypoints[image_path]
        ax.scatter(keypoints[:,1], keypoints[:,0], c='k', s=1)

    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove white border
    st.pyplot(fig)