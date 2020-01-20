import streamlit as st
import pandas as pd

st.title("File uploader example")

st.write(
    """
This is an example of how to use a file uploader.
Here, we are simply going to upload a CSV file and display it.

It should serve as a minimal example
for you to jump off and do more complex things.
"""
)

st.header("Upload CSV")
csv_file = st.file_uploader(
    label="Upload a CSV file", type=["csv"], encoding="utf-8"
)

if csv_file is not None:
    data = pd.read_csv(csv_file)
    st.dataframe(data)

st.header("Upload Images")

st.write(
    """
Below is another example, where we upload an image and display it.
"""
)

image_file = st.file_uploader(
    label="Upload an image", type=["png", "jpg", "tiff"], encoding=None
)

if image_file is not None:
    st.image(image_file)
