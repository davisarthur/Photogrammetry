import streamlit as st
import os
import plotly.graph_objects as go
import matplotlib.image as mpimg
import logging
import numpy as np
from keypoint import utils, keypoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_session_state():
    corner_scores = st.session_state.get('corner_scores', {})
    keypoints = st.session_state.get('keypoints', {})
    selected_points = st.session_state.get('selected_points', [])
    st.session_state.corner_scores = corner_scores
    st.session_state.keypoints = keypoints
    st.session_state.selected_points = selected_points


logger.info('Starting the application')
init_session_state()

# List all files in the images directory and its subdirectories
image_dir = 'images'
valid_extensions = ('.png', '.jpg')
image_files = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith(valid_extensions):
            image_files.append(os.path.join(root, file))

# Create a dropdown to select a file
image_path = st.selectbox('Select an image file', image_files)

# Display the selected image
if image_path:
    logger.info(f'Selected image: {image_path}')
    img = mpimg.imread(image_path)

    fig = go.Figure()
    fig.add_trace(go.Image(z=img))

    # Compute corner scores
    if st.button('Compute corner scores'):
        logger.info(f'Retrieving corner scores for {image_path}')
        if image_path not in st.session_state.corner_scores:
            logger.info(f'Computing corner scores for {image_path}')
            intensity = utils.rgb_to_intensity(image_path)
            st.session_state.corner_scores[image_path] = keypoint.corner_finder(intensity, smoothing_variance=1.0)

    # Filter keypoints
    if st.session_state.corner_scores.get(image_path) is not None:
        # Allow user to select a filtering threshold based on log difference between max and 95th percentile corner scores
        corner_scores = st.session_state.corner_scores[image_path]
        corner_score_p95 = np.percentile(corner_scores, 95)
        corner_score_max = np.amax(corner_scores)
        logger.info(f'Corner score 95th percentile: {corner_score_p95}, Corner score max: {corner_score_max}')

        log_diff = np.log(corner_score_max - corner_score_p95)
        log_threshold = st.slider(
            'Set filtering threshold',
            min_value=0,
            max_value=int(log_diff)+1,
            value=int(log_diff/2),
            step=1,
            format="%s"
        )
        threshold = corner_score_p95 + np.exp(log_threshold)

        logger.info(f'Filtering keypoints with threshold {threshold}')
        thresholded_corner_scores = np.where(corner_scores > threshold, corner_scores, 0)
        cleaned_thresholded_corners = utils.clean_image(thresholded_corner_scores)
        keypoints = np.argwhere(cleaned_thresholded_corners > 0)
        logger.info(f'Number of keypoints: {len(keypoints)}')
        fig.add_trace(
            go.Scatter(
                x=keypoints[:,1],
                y=keypoints[:,0],
                mode='markers',
                marker=dict(color='red', size=2)
            )
        )

    fig.update_layout(
        width=800,
        height=800,
        xaxis=dict(range=[0, img.shape[1]]),
        yaxis=dict(range=[img.shape[0], 0]),
        dragmode='select',
    )

    event = st.plotly_chart(fig, on_select='rerun', selection_mode='box')
    if event and event['selection']:
        points = event['selection']['points']
        st.session_state.selected_points = np.array([[point['x'], point['y']] for point in points])
        logger.info(f'Selected {len(st.session_state.selected_points)} points')
