import subprocess
import streamlit as st

def open_folder(button_text, path):
    """
    Create a button to open the folder in the file explorer.
    """
    if st.button(button_text):
        print(f"Opening folder: {path}")
        subprocess.Popen(r'explorer /select,"'+ path+ '"')