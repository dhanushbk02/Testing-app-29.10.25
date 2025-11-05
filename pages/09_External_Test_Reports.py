# ============================================================
# 🧾 External Test Reports Page (Service Account - Deployable)
# Author: Dhanush BK | Flow Oil Pumps Pvt. Ltd.
# Purpose: Upload & view external test reports
# Storage: Google Drive (Testing-app-uploads/External Reports)
# Deployable: Yes (Streamlit Cloud Ready, Uses secrets.toml)
# ============================================================

import streamlit as st
import pandas as pd
import io
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# ------------------------------------------------------------
# 🔐 Google Drive Setup (From secrets.toml)
# ------------------------------------------------------------
try:
    SERVICE_ACCOUNT = st.secrets["gcp_service_account"]
    GDRIVE = st.secrets["gdrive"]
    PARENT_FOLDER_ID = GDRIVE.get("folder_id")
except Exception as e:
    st.error("⚠️ Google Drive configuration not found in secrets.toml.")
    st.stop()

SCOPES = ["https://www.googleapis.com/auth/drive"]

creds = service_account.Credentials.from_service_account_info(
    SERVICE_ACCOUNT,
    scopes=SCOPES
)
drive_service = build("drive", "v3", credentials=creds)

# ------------------------------------------------------------
# 📁 Folder Configuration
# ------------------------------------------------------------
EXTERNAL_REPORTS_NAME = "External Reports"
MODEL_FILE_NAME = "List of models for page8.xlsx"

# ------------------------------------------------------------
# 🧰 Helper Functions
# ------------------------------------------------------------
def get_folder_id(folder_name, parent_id):
    """Find the folder ID under the given parent folder."""
    query = (
        f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' "
        f"and '{parent_id}' in parents and trashed=false"
    )
    results = drive_service.files().list(
        q=query,
        fields="files(id, name)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True
    ).execute()
    files = results.get("files", [])
    if not files:
        st.error(f"🚫 Folder '{folder_name}' not found in Google Drive.")
        return None
    return files[0]["id"]

external_reports_id = get_folder_id(EXTERNAL_REPORTS_NAME, PARENT_FOLDER_ID)

# ------------------------------------------------------------
# 📘 Load Pump Model List
# ------------------------------------------------------------
def get_model_list():
    """Read Excel file containing pump model list from Drive."""
    query = f"name='{MODEL_FILE_NAME}' and '{PARENT_FOLDER_ID}' in parents and trashed=false"
    results = drive_service.files().list(
        q=query,
        fields="files(id)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True
    ).execute()
    files = results.get("files", [])
    if not files:
        st.error(f"'{MODEL_FILE_NAME}' not found in Google Drive.")
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
    """Upload a test report file to the External Reports folder."""
    if not uploaded_file:
        st.warning("Please select a file before uploading.")
        return

    ext = uploaded_file.name.split(".")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    new_filename = f"{model_name}_{timestamp}.{ext}"

    file_metadata = {
        "name": new_filename,
        "parents": [external_reports_id],
    }

    media = MediaIoBaseUpload(
        io.BytesIO(uploaded_file.read()),
        mimetype=uploaded_file.type,
        resumable=True
    )

    drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id",
        supportsAllDrives=True
    ).execute()

    st.success(f"✅ File '{new_filename}' uploaded successfully to Google Drive!")

# ------------------------------------------------------------
# 📄 List Uploaded Reports
# ------------------------------------------------------------
def list_uploaded_reports():
    """List all uploaded external test reports."""
    results = drive_service.files().list(
        q=f"'{external_reports_id}' in parents and trashed=false",
        fields="files(id, name, mimeType, modifiedTime)",
        orderBy="modifiedTime desc",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True
    ).execute()
    return results.get("files", [])

# ------------------------------------------------------------
# 🖥️ Streamlit UI
# ------------------------------------------------------------
st.title("📄 External Test Reports")
st.caption("Upload and view external test reports linked to pump models")

st.divider()

# Load Model List
models = get_model_list()
if not models:
    st.warning("⚠️ Unable to fetch pump models. Please check Google Drive or Excel file name.")
else:
    selected_model = st.selectbox("Select Pump Model", models)
    c1, c2, c3 = st.columns(3)
    impeller_dia = c1.text_input("Impeller Dia (mm)")
    flow = c2.text_input("Flow (LPM)")
    head = c3.text_input("Head (m)")
    remarks = st.text_area("Remarks (optional)")

    st.subheader("⬆ Upload External Test Report")
    uploaded_file = st.file_uploader("Choose file", type=["pdf", "jpg", "jpeg"])

    if st.button("Upload to Google Drive"):
        if selected_model and uploaded_file:
            upload_file_to_drive(uploaded_file, selected_model)
        else:
            st.warning("Please select a model and upload a file.")

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
                    unsafe_allow_html=True)
            elif "image" in f["mimeType"]:
                st.image(f"https://drive.google.com/uc?id={f['id']}", use_container_width=True)
            st.markdown(f"[🔗 Open in Drive](https://drive.google.com/file/d/{f['id']}/view?usp=sharing)")
            st.divider()
