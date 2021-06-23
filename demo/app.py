# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import streamlit as st
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import cv2

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if any(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from doctr.documents import DocumentFile
from doctr.models import ocr_predictor
from doctr.utils.visualization import draw_page, visualize_page

DET_ARCHS = ["db_resnet50"]
RECO_ARCHS = ["crnn_vgg16_bn", "crnn_resnet31", "sar_vgg16_bn", "sar_resnet31"]


def main():

    # Wide mode
    st.set_page_config(layout="wide")

    # Designing the interface
    st.title("DocTR: Document Text Recognition")
    # For newline
    st.write('\n')
    # Set the columns
    cols = st.beta_columns((1, 1))
    cols[0].subheader("Input document (first page)")
    cols[1].subheader("Raw heatmap (segmentation task)")

    # Sidebar
    # File selection
    st.sidebar.title("Document selection")
    # Disabling warning
    st.set_option('deprecation.showfileUploaderEncoding', False)
    # Choose your own image
    uploaded_file = st.sidebar.file_uploader("Upload files", type=['pdf', 'png', 'jpeg', 'jpg'])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.pdf'):
            doc = DocumentFile.from_pdf(uploaded_file.read()).as_images(output_size=(1024, 1024))
        else:
            doc = DocumentFile.from_images(uploaded_file.read())
        cols[0].image(doc[0], width=640)

    # Model selection
    st.sidebar.title("Model selection")
    det_arch = st.sidebar.selectbox("Text detection model", DET_ARCHS)
    reco_arch = st.sidebar.selectbox("Text recognition model", RECO_ARCHS)

    # For newline
    st.sidebar.write('\n')

    if st.sidebar.button("Analyze document"):

        if uploaded_file is None:
            st.sidebar.write("Please upload a document")

        else:
            with st.spinner('Loading model...'):
                predictor = ocr_predictor(det_arch, reco_arch, pretrained=True)

            with st.spinner('Analyzing...'):

                # Forward the image to the model
                processed_batches = predictor.det_predictor.pre_processor(doc)
                out = predictor.det_predictor.model(processed_batches[0], return_model_output=True, training=False)
                seg_map = out["out_map"]
                seg_map = tf.squeeze(seg_map[0, ...], axis=[2])
                seg_map = cv2.resize(seg_map.numpy(), (doc[0].shape[1], doc[0].shape[0]),
                                     interpolation=cv2.INTER_LINEAR)
                # Plot the raw heatmap
                fig, ax = plt.subplots()
                ax.imshow(seg_map)
                ax.axis('off')
                cols[1].pyplot(fig)

                # Plot OCR output
                out = predictor(doc, training=False)
                cols[1].subheader("OCR output")
                fig = visualize_page(out.pages[0].export(), doc[0], interactive=False)
                cols[1].pyplot(fig)

                # Page reconsitution under input page
                cols[0].subheader("Page reconstitution from OCR output")
                img = draw_page(out.pages[0].export())
                cols[0].image(img, clamp=True, width=640)


if __name__ == '__main__':
    main()
