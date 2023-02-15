# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from doctr.file_utils import is_tf_available
from doctr.io import DocumentFile
from doctr.utils.visualization import visualize_page

if is_tf_available():
    import tensorflow as tf

    from backend.tensorflow import DET_ARCHS, RECO_ARCHS, forward_image, load_predictor

    if any(tf.config.experimental.list_physical_devices("gpu")):
        forward_device = tf.device("/gpu:0")
    else:
        forward_device = tf.device("/cpu:0")

else:
    import torch

    from backend.pytorch import DET_ARCHS, RECO_ARCHS, forward_image, load_predictor

    forward_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(det_archs, reco_archs):
    """Build a streamlit layout"""

    # Wide mode
    st.set_page_config(layout="wide")

    # Designing the interface
    st.title("docTR: Document Text Recognition")
    # For newline
    st.write("\n")
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
    st.set_option("deprecation.showfileUploaderEncoding", False)
    # Choose your own image
    uploaded_file = st.sidebar.file_uploader("Upload files", type=["pdf", "png", "jpeg", "jpg"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            doc = DocumentFile.from_pdf(uploaded_file.read())
        else:
            doc = DocumentFile.from_images(uploaded_file.read())
        page_idx = st.sidebar.selectbox("Page selection", [idx + 1 for idx in range(len(doc))]) - 1
        page = doc[page_idx]
        cols[0].image(page)

    # Model selection
    st.sidebar.title("Model selection")
    st.sidebar.markdown("**Backend**: " + ("TensorFlow" if is_tf_available() else "PyTorch"))
    det_arch = st.sidebar.selectbox("Text detection model", det_archs)
    reco_arch = st.sidebar.selectbox("Text recognition model", reco_archs)

    # For newline
    st.sidebar.write("\n")

    if st.sidebar.button("Analyze page"):
        if uploaded_file is None:
            st.sidebar.write("Please upload a document")

        else:
            with st.spinner("Loading model..."):
                predictor = load_predictor(det_arch, reco_arch, forward_device)

            with st.spinner("Analyzing..."):
                # Forward the image to the model
                seg_map = forward_image(predictor, page, forward_device)
                seg_map = np.squeeze(seg_map)
                seg_map = cv2.resize(seg_map, (page.shape[1], page.shape[0]), interpolation=cv2.INTER_LINEAR)

                # Plot the raw heatmap
                fig, ax = plt.subplots()
                ax.imshow(seg_map)
                ax.axis("off")
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


if __name__ == "__main__":
    main(DET_ARCHS, RECO_ARCHS)
