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

# List all files in the images directory and its subdirectories
image_dir = 'images'
valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
image_files = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith(valid_extensions):
            image_files.append(os.path.join(root, file))

# Create a dropdown to select a file
selected_image = st.selectbox('Select an image file', image_files)

# Display the selected image
if selected_image:
    logger.info(f'Selected image: {selected_image}')
    image_path = selected_image
    img = mpimg.imread(image_path)

    # Get image dimensions
    height, width = img.shape[:2]

    # Compute corner scores
    if st.button('Compute corner scores'):
        logger.info(f'Retrieving corner scores for {selected_image}')
        intensity = utils.rgb_to_intensity(image_path)
        if image_path not in st.session_state.corner_scores:
            logger.info(f'Computing corner scores for {selected_image}')
            st.session_state.corner_scores[image_path] = keypoint.corner_finder(intensity, smoothing_variance=1.0)

    # Filter keypoints
    if st.session_state.corner_scores.get(image_path) is not None:
        corner_scores = st.session_state.corner_scores[image_path]
        corner_score_p95 = np.percentile(corner_scores, 95)
        corner_score_max = np.amax(corner_scores)

        # Allow user to select a filtering threshold based on log difference between max and 95th percentile
        logger.info(f'Corner score 95th percentile: {corner_score_p95}, Corner score max: {corner_score_max}')
        log_diff = np.log(corner_score_max - corner_score_p95)
        log_threshold = st.slider('Set filtering threshold', min_value=0, max_value=int(log_diff)+1, value=int(log_diff/2), step=1, format="%s")
        threshold = corner_score_p95 + np.exp(log_threshold)

        logger.info(f'Filtering keypoints with threshold {threshold}')
        thresholded_corner_scores = np.where(corner_scores > threshold, corner_scores, 0)
        cleaned_thresholded_corners = utils.clean_image(thresholded_corner_scores)
        st.session_state.keypoints[image_path] = np.argwhere(cleaned_thresholded_corners > 0)

    # Plot the image with keypoints
    fig, ax = plt.subplots()
    ax.imshow(img)
    if st.session_state.keypoints.get(image_path) is not None:
        keypoints = st.session_state.keypoints[image_path]
        ax.scatter(keypoints[:,1], keypoints[:,0], c='k', s=1)

    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove white border
    st.pyplot(fig)