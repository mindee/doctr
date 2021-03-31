# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import streamlit as st
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if any(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from doctr.documents import DocumentFile
from doctr.models import detection_predictor

DET_ARCHS = ["db_resnet50"]
RECO_ARCHS = ["crnn_vgg16_bn", "crnn_resnet31", "sar_vgg16_bn", "sar_resnet31"]


def main():

    # Designing the interface
    st.title("DocTR: Document Text Recognition")
    # For newline
    st.write('\n')

    # Sidebar
    st.sidebar.title("Model selection")
    det_arch = st.sidebar.selectbox("Text detection model", DET_ARCHS)

    predictor = detection_predictor(det_arch, pretrained=True)

    # reco_arch = st.sidebar.selectbox("Text recognition model", RECO_ARCHS)
    # predictor = detection_predictor(reco_arch, pretrained=True)

    st.sidebar.title("Document selection")
    # Disabling warning
    st.set_option('deprecation.showfileUploaderEncoding', False)
    # Choose your own image
    uploaded_file = st.sidebar.file_uploader("Upload files", type=['pdf', 'png', 'jpeg', 'jpg'])

    # Set the two columns
    col1, col2 = st.beta_columns((1, 1))
    col1.header("Input page")
    col2.header("Text detection output")

    if uploaded_file is not None:

        if uploaded_file.name.endswith('.pdf'):
            doc = DocumentFile.from_pdf(uploaded_file.read())
        else:
            doc = DocumentFile.from_images(uploaded_file.read())
        col1.image(doc[0], 'First page', use_column_width=True)

    # For newline
    st.sidebar.write('\n')

    if st.sidebar.button("Analyze document"):

        if uploaded_file is None:
            st.sidebar.write("Please upload a document")

        else:

            with st.spinner('Analyzing...'):

                # Forward the image to the model
                processed_batches = predictor.pre_processor(doc)
                seg_map = predictor.model(processed_batches[0])[0]

                # Plot the raw heatmap
                fig, ax = plt.subplots()
                ax.imshow(seg_map)
                ax.axis('off')
                col2.pyplot(fig, use_column_width=True)


if __name__ == '__main__':
    main()
