import streamlit as st
import runpy
from pathlib import Path
import os

st.set_page_config(page_title="Test Report Generator", page_icon="🧾", layout="wide")

# 👇 Correct path to your generator
GEN_PATH = Path(__file__).resolve().parents[1] / "TestReportGenerator" / "app_streamlit4.py"
GEN_DIR = GEN_PATH.parent

st.title("🧾 Test Report Generator")
st.caption(f"Launching from: {GEN_PATH}")

if GEN_PATH.exists():
    cwd = os.getcwd()
    try:
        # temporarily change directory so relative files (logo, Excel templates, etc.) work
        os.chdir(GEN_DIR)
        runpy.run_path(str(GEN_PATH), run_name="__main__")
    finally:
        os.chdir(cwd)
else:
    st.error(f"Generator file not found: {GEN_PATH}")
