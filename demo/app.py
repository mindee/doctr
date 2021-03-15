# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import streamlit as st
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from doctr.documents import read_pdf_from_stream
from doctr.models import db_resnet50_predictor
from doctr.utils.visualization import visualize_page


predictor = db_resnet50_predictor(pretrained=True)


def main():

    # Designing the interface
    st.title("DocTR: Document Text Recognition")
    # For newline
    st.write('\n')

    # Sidebar
    st.sidebar.title("Document selection")
    # Disabling warning
    st.set_option('deprecation.showfileUploaderEncoding', False)
    # Choose your own image
    uploaded_file = st.sidebar.file_uploader(" ", type=['pdf'])

    # Set the two columns
    col1, col2 = st.beta_columns((1, 1))
    col1.header("Input page")
    col2.header("Text detection output")

    if uploaded_file is not None:

        doc = read_pdf_from_stream(uploaded_file.read())
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
