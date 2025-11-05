# ============================================================
# 🌍 External Test Reports (Nextcloud Integration)
# Author: Dhanush BK | Flow Oil Pumps Pvt. Ltd.
# Description: Upload, list, and preview external test reports
# Storage: Nextcloud via WebDAV (from st.secrets["nextcloud"])
# ============================================================

import streamlit as st
import pandas as pd
import io
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime

# ------------------------------------------------------------
# 🔐 Load Nextcloud Config from secrets.toml
# ------------------------------------------------------------
config = st.secrets["nextcloud"]
BASE_URL = config["url"]
USERNAME = config["user"]
PASSWORD = config["password"]
BASE_FOLDER = config["base_folder"]

# Full WebDAV path for the External Reports subfolder
EXTERNAL_FOLDER = f"{BASE_URL}/{BASE_FOLDER}/External Reports"

# ------------------------------------------------------------
# ⚙️ Helper Functions
# ------------------------------------------------------------
def ensure_folder_exists(folder_url):
    """Check if folder exists, create if missing."""
    res = requests.request("PROPFIND", folder_url, auth=HTTPBasicAuth(USERNAME, PASSWORD))
    if res.status_code == 404:
        res = requests.request("MKCOL", folder_url, auth=HTTPBasicAuth(USERNAME, PASSWORD))
        if res.status_code not in (201, 405):
            st.error(f"❌ Could not create folder: {res.text}")
        else:
            st.success("📂 Created 'External Reports' folder in Nextcloud")

def upload_to_nextcloud(file_obj, filename):
    """Upload file to Nextcloud folder."""
    upload_url = f"{EXTERNAL_FOLDER}/{filename}"
    res = requests.put(upload_url, data=file_obj.read(), auth=HTTPBasicAuth(USERNAME, PASSWORD))
    if res.status_code in (201, 204):
        st.success(f"✅ Uploaded successfully as '{filename}'")
    else:
        st.error(f"⚠️ Upload failed ({res.status_code}): {res.text}")

def list_files_in_nextcloud(folder_url):
    """List files in a given Nextcloud folder."""
    res = requests.request("PROPFIND", folder_url, auth=HTTPBasicAuth(USERNAME, PASSWORD))
    if res.status_code != 207:
        st.warning("⚠️ Could not list files. Check credentials or folder path.")
        return []
    from xml.etree import ElementTree as ET
    tree = ET.fromstring(res.content)
    files = []
    for resp in tree.findall("{DAV:}response"):
        href = resp.find("{DAV:}href").text
        if href.endswith("/"):  # skip folders
            continue
        name = href.split("/")[-1]
        files.append(name)
    return sorted(files)

# ------------------------------------------------------------
# 🖥️ Streamlit UI
# ------------------------------------------------------------
st.title("📁 External Test Reports")
st.caption("Upload and view external test reports linked to pump models (stored in Nextcloud)")

st.divider()

# Ensure folder exists before operations
ensure_folder_exists(EXTERNAL_FOLDER)

# ------------------------------------------------------------
# 📘 Load Pump Model List (Optional - from Excel in same folder)
# ------------------------------------------------------------
MODEL_FILE_NAME = "List of models for page8.xlsx"
model_file_url = f"{BASE_URL}/{BASE_FOLDER}/{MODEL_FILE_NAME}"
res = requests.get(model_file_url, auth=HTTPBasicAuth(USERNAME, PASSWORD))

if res.status_code == 200:
    df = pd.read_excel(io.BytesIO(res.content))
    model_list = df.iloc[1:, 0].dropna().tolist()
else:
    st.warning(f"⚠️ Could not load model list file ({res.status_code}). Check '{MODEL_FILE_NAME}' in Nextcloud.")
    model_list = []

# ------------------------------------------------------------
# 1️⃣ Upload Section
# ------------------------------------------------------------
selected_model = st.selectbox("Select Pump Model", model_list if model_list else ["-- No Models Found --"])
col1, col2, col3 = st.columns(3)
impeller_dia = col1.text_input("Impeller Dia (mm)")
flow = col2.text_input("Flow (LPM)")
head = col3.text_input("Head (m)")
remarks = st.text_area("Remarks (optional)")

st.subheader("⬆ Upload External Test Report")
uploaded_file = st.file_uploader("Choose file", type=["pdf", "jpg", "jpeg"])

if st.button("Upload to Nextcloud"):
    if uploaded_file and selected_model != "-- No Models Found --":
        ext = uploaded_file.name.split(".")[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"{selected_model}_{timestamp}.{ext}"
        upload_to_nextcloud(uploaded_file, filename)
    else:
        st.warning("Please select a model and upload a file before submitting.")

st.divider()

# ------------------------------------------------------------
# 2️⃣ View Uploaded Files
# ------------------------------------------------------------
st.subheader("📚 View Uploaded External Reports")
files = list_files_in_nextcloud(EXTERNAL_FOLDER)

if not files:
    st.info("No external reports found in Nextcloud.")
else:
    for file in files:
        file_url = f"{EXTERNAL_FOLDER}/{file}"
        if file.lower().endswith(".pdf"):
            st.markdown(f"**📄 {file}**")
            st.markdown(f'<iframe src="{file_url}" width="100%" height="500"></iframe>', unsafe_allow_html=True)
        elif file.lower().endswith((".jpg", ".jpeg", ".png")):
            st.image(file_url, caption=file, use_container_width=True)
        st.markdown(f"[🔗 Open in Nextcloud]({file_url})")
        st.divider()
