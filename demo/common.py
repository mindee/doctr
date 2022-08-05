# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import matplotlib.pyplot as plt
import streamlit as st

import cv2
import numpy as np

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from doctr.models.base import OCRPredictor
from doctr.utils.visualization import visualize_page

if os.environ.get("USE_TF").upper() == "YES":
    import tensorflow as tf
    if any(tf.config.experimental.list_physical_devices('gpu')):
        forward_device = tf.device("/gpu:0")
    else:
        forward_device = tf.device("/cpu:0")

elif os.environ.get("USE_TORCH").upper() == "YES":
    import torch
    forward_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

else:
    raise ValueError("Please set environment variables either USE_TF or USE_TORCH to 'YES'")


def load_predictor(det_arch: str, reco_arch: str, device) -> OCRPredictor:
    """
    Args:
        device is either tf.device or torch.device
    """
    if os.environ.get("USE_TF").upper() == "YES":
        with device:
            predictor = ocr_predictor(
                det_arch, reco_arch, pretrained=True,
                assume_straight_pages=("rotation" not in det_arch)
            )
    else:
        predictor = ocr_predictor(
            det_arch, reco_arch, pretrained=True,
            assume_straight_pages=("rotation" not in det_arch)
        ).to(device)
    return predictor


def forward_image(predictor: OCRPredictor, image: np.ndarray, device) -> np.ndarray:
    """
    Args:
        device is either tf.device or torch.device
    """
    if os.environ.get("USE_TF").upper() == "YES":
        with device:
            processed_batches = predictor.det_predictor.pre_processor([image])
            out = predictor.det_predictor.model(processed_batches[0], return_model_output=True)
            seg_map = out["out_map"]
 
        with tf.device("/cpu:0"):
            seg_map = tf.identity(seg_map).numpy()
    else:
        with torch.no_grad():
            processed_batches = predictor.det_predictor.pre_processor([image])
            out = predictor.det_predictor.model(processed_batches[0].to(device), return_model_output=True)
            seg_map = out["out_map"].to("cpu").numpy()

    return seg_map


def main(det_archs, reco_archs):
    """Build a streamlit layout"""

    # Wide mode
    st.set_page_config(layout="wide")

    # Designing the interface
    st.title("docTR: Document Text Recognition")
    # For newline
    st.write('\n')
    # Instructions
    st.markdown("*Hint: click on the top-right corner of an image to enlarge it!*")
    # Set the columns
    cols = st.columns((1, 1, 1, 1))
    cols[0].subheader("Input page")
    cols[1].subheader("Segmentation heatmap")
    cols[2].subheader("OCR output")
    cols[3].subheader("Page reconstitution")

    # Sidebar
    # File selection
    st.sidebar.title("Document selection")
    # Disabling warning
    st.set_option('deprecation.showfileUploaderEncoding', False)
    # Choose your own image
    uploaded_file = st.sidebar.file_uploader("Upload files", type=['pdf', 'png', 'jpeg', 'jpg'])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.pdf'):
            doc = DocumentFile.from_pdf(uploaded_file.read())
        else:
            doc = DocumentFile.from_images(uploaded_file.read())
        page_idx = st.sidebar.selectbox("Page selection", [idx + 1 for idx in range(len(doc))]) - 1
        page = doc[page_idx]
        cols[0].image(page)

    # Model selection
    st.sidebar.title("Model selection")
    det_arch = st.sidebar.selectbox("Text detection model", det_archs)
    reco_arch = st.sidebar.selectbox("Text recognition model", reco_archs)

    # For newline
    st.sidebar.write('\n')

    if st.sidebar.button("Analyze page"):

        if uploaded_file is None:
            st.sidebar.write("Please upload a document")

        else:
            with st.spinner('Loading model...'):
                predictor = load_predictor(det_arch, reco_arch, forward_device)

            with st.spinner('Analyzing...'):

                # Forward the image to the model
                seg_map = forward_image(predictor, page, forward_device)
                seg_map = np.squeeze(seg_map)
                seg_map = cv2.resize(seg_map, (page.shape[1], page.shape[0]),
                                     interpolation=cv2.INTER_LINEAR)

                # Plot the raw heatmap
                fig, ax = plt.subplots()
                ax.imshow(seg_map)
                ax.axis('off')
                cols[1].pyplot(fig)

                # Plot OCR output
                out = predictor([page])
                fig = visualize_page(out.pages[0].export(), page, interactive=False)
                cols[2].pyplot(fig)

                # Page reconsitution under input page
                page_export = out.pages[0].export()
                if "rotation" not in det_arch:
                    img = out.pages[0].synthesize()
                    cols[3].image(img, clamp=True)

                # Display JSON
                st.markdown("\nHere are your analysis results in JSON format:")
                st.json(page_export)


if __name__ == '__main__':
    main()
