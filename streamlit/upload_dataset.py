import streamlit as st
import pandas as pd
import os

# Define the directory where files will be saved
UPLOAD_DIRECTORY = "../datasets/"

# Create the directory if it doesn't exist
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

def save_uploaded_file(uploaded_file):
    """Save the uploaded file to the defined directory."""
    try:
        # Construct the full file path
        file_path = os.path.join(UPLOAD_DIRECTORY, uploaded_file.name)
        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def st_dataset_upload():
    st.title("CSV File Upload")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Show the contents of the uploaded file
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(data)

        # Save the file
        saved_path = save_uploaded_file(uploaded_file)
        if saved_path:
            st.success(f"File saved successfully: {saved_path}")

if __name__ == "__main__":
    st_dataset_upload()