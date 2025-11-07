# ============================================================
# 🧾 Internal Test Reports Page (Service Account + Google Drive)
# Author: Dhanush BK | Flow Oil Pumps Pvt. Ltd.
# Purpose: Upload & view internal test reports (Google Drive API)
# Storage: Google Drive (Testing-app-uploads/Internal Reports)
# Updated: Streamlit Cloud compatible (no credentials.json needed)
# ============================================================

import streamlit as st
import pandas as pd
import io
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google.oauth2 import service_account

# ------------------------------------------------------------
# 🔐 Google Drive Authentication using Streamlit Secrets
# ------------------------------------------------------------
SCOPES = ["https://www.googleapis.com/auth/drive"]

def get_drive_service():
    """Authenticate to Google Drive using service account credentials."""
    SERVICE_ACCOUNT_INFO = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(
        SERVICE_ACCOUNT_INFO, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)

drive_service = get_drive_service()

# ------------------------------------------------------------
# 🔍 Folder References
# ------------------------------------------------------------
PARENT_FOLDER_ID = st.secrets["gdrive"]["folder_id"]  # main shared folder
SUBFOLDER_NAME = "Internal Reports"
MODEL_FILE_NAME = "List of models for page8.xlsx"

# ------------------------------------------------------------
# 🧰 Utility Functions
# ------------------------------------------------------------
def get_subfolder_id(parent_id, subfolder_name):
    """Return subfolder ID if exists under given parent."""
    query = (
        f"name='{subfolder_name}' and "
        f"'{parent_id}' in parents and "
        f"mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    results = drive_service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
    ).execute()
    files = results.get("files", [])
    if not files:
        st.error(f"🚫 Folder '{subfolder_name}' not found inside shared Drive folder.")
        return None
    return files[0]["id"]

internal_reports_id = get_subfolder_id(PARENT_FOLDER_ID, SUBFOLDER_NAME)

# ------------------------------------------------------------
# 📘 Load Pump Model List
# ------------------------------------------------------------
def get_model_list():
    """Fetch Excel model list from shared folder."""
    query = f"name='{MODEL_FILE_NAME}' and '{PARENT_FOLDER_ID}' in parents and trashed=false"
    results = drive_service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
    ).execute()
    files = results.get("files", [])
    if not files:
        st.error(f"📂 '{MODEL_FILE_NAME}' not found in shared folder.")
        return []

    file_id = files[0]["id"]
    request = drive_service.files().get_media(fileId=file_id)
    content = io.BytesIO(request.execute())
    df = pd.read_excel(content)
    return df.iloc[1:, 0].dropna().tolist()

# ------------------------------------------------------------
# 📤 Upload File to Drive
# ------------------------------------------------------------
def upload_file_to_drive(uploaded_file, model_name):
    """Upload internal report to Drive."""
    if not internal_reports_id:
        st.error("❌ Internal Reports folder missing — cannot upload.")
        return

    ext = uploaded_file.name.split(".")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    new_name = f"{model_name}_{timestamp}.{ext}"

    file_metadata = {"name": new_name, "parents": [internal_reports_id]}
    media = MediaIoBaseUpload(io.BytesIO(uploaded_file.read()), mimetype=uploaded_file.type)

    drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id",
        supportsAllDrives=True,
    ).execute()

    st.success(f"✅ Uploaded successfully as '{new_name}'")

# ------------------------------------------------------------
# 📄 List Uploaded Reports
# ------------------------------------------------------------
def list_uploaded_reports():
    """List all uploaded test reports."""
    if not internal_reports_id:
        return []
    results = drive_service.files().list(
        q=f"'{internal_reports_id}' in parents and trashed=false",
        fields="files(id, name, mimeType, modifiedTime)",
        orderBy="modifiedTime desc",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
    ).execute()
    return results.get("files", [])

# ------------------------------------------------------------
# 🖥️ Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="Internal Test Reports", page_icon="📄", layout="wide")
st.title("📄 Internal Test Reports")
st.caption("Upload and view internal test reports linked to pump models")

st.divider()

# ------------------------------------------------------------
# Model List Section
# ------------------------------------------------------------
models = get_model_list()
if not models:
    st.warning("⚠️ Unable to fetch pump models. Please check file name or Drive access.")
else:
    selected_model = st.selectbox("Select Pump Model", models)
    c1, c2, c3 = st.columns(3)
    impeller_dia = c1.text_input("Impeller Dia (mm)")
    flow = c2.text_input("Flow (LPM)")
    head = c3.text_input("Head (m)")
    remarks = st.text_area("Remarks (optional)")

    st.subheader("⬆ Upload Internal Test Report")
    uploaded_file = st.file_uploader("Choose file", type=["pdf", "jpg", "jpeg"])

    if st.button("Upload to Google Drive"):
        if uploaded_file and selected_model:
            upload_file_to_drive(uploaded_file, selected_model)
        else:
            st.warning("Please select a model and choose a file to upload.")

    st.divider()
    st.subheader("📚 View Uploaded Reports")

    files = list_uploaded_reports()
    if not files:
        st.info("No files uploaded yet.")
    else:
        for f in files:
            st.markdown(f"**📄 {f['name']}**  \n*Modified:* {f['modifiedTime'][:10]}")
            if "pdf" in f["mimeType"]:
                st.markdown(
                    f'<iframe src="https://drive.google.com/file/d/{f["id"]}/preview" width="100%" height="500"></iframe>',
                    unsafe_allow_html=True,
                )
            elif "image" in f["mimeType"]:
                st.image(f"https://drive.google.com/uc?id={f['id']}", use_container_width=True)
            st.markdown(f"[🔗 Open in Drive](https://drive.google.com/file/d/{f['id']}/view?usp=sharing)")
            st.divider()
