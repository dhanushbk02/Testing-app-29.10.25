import streamlit as st
import pandas as pd
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
import gspread
from datetime import datetime

# ------------------------------------------------------------
# 🔐 Google Auth Setup
# ------------------------------------------------------------
creds = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=[
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets",
    ],
)

# Initialize clients
drive_service = build("drive", "v3", credentials=creds)
gc = gspread.authorize(creds)

# ------------------------------------------------------------
# 📂 CONFIGURATION
# ------------------------------------------------------------
MAIN_FOLDER_ID = st.secrets["gdrive"]["folder_id"]  # "Testing-app-uploads"
CALIB_FOLDER_NAME = "cal_certificates"
SHEET_ID = "your_google_sheet_id_here"  # 🔹 Replace with your Instrument List Google Sheet ID
SHEET_NAME = "Sheet1"

st.title("🧰 Instrument Details")
st.caption("Manage and view instrument details with calibration reports stored in Google Drive")

st.divider()

# ------------------------------------------------------------
# 📊 Load Instrument List from Google Sheet
# ------------------------------------------------------------
try:
    sh = gc.open_by_key(SHEET_ID)
    worksheet = sh.worksheet(SHEET_NAME)
    data = worksheet.get_all_records()
    df = pd.DataFrame(data)

    st.subheader("📋 Calibration Instrument Register")
    st.dataframe(df, use_container_width=True)
except Exception as e:
    st.error(f"⚠️ Could not load Google Sheet: {e}")
    df = pd.DataFrame()

st.divider()

# ------------------------------------------------------------
# ➕ Add or Update Instrument
# ------------------------------------------------------------
st.subheader("➕ Add New Instrument")

with st.form("add_instrument"):
    c1, c2, c3 = st.columns(3)
    inst_id = c1.text_input("Instrument ID (Unique)")
    name = c2.text_input("Instrument Name")
    make = c3.text_input("Make")

    c4, c5, c6 = st.columns(3)
    model = c4.text_input("Model")
    sl_no = c5.text_input("Serial Number")
    loc = c6.text_input("Location")

    c7, c8, c9 = st.columns(3)
    cal_date = c7.date_input("Calibration Date")
    due_date = c8.date_input("Due Date")
    status = c9.selectbox("Status", ["Calibrated", "Pending", "Upcoming"])

    submitted = st.form_submit_button("💾 Save to Google Sheet")

    if submitted:
        try:
            new_row = [
                inst_id,
                name,
                make,
                model,
                sl_no,
                loc,
                str(cal_date),
                str(due_date),
                status,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ]
            worksheet.append_row(new_row)
            st.success(f"✅ '{name}' added successfully!")
        except Exception as e:
            st.error(f"❌ Failed to update sheet: {e}")

st.divider()

# ------------------------------------------------------------
# 📁 View Calibration Reports
# ------------------------------------------------------------
st.subheader("📁 Calibration Certificates")

def get_folder_id(folder_name, parent_id):
    """Find subfolder ID by name inside parent folder"""
    query = f"name='{folder_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = (
        drive_service.files()
        .list(q=query, fields="files(id, name)", supportsAllDrives=True, includeItemsFromAllDrives=True)
        .execute()
    )
    files = results.get("files", [])
    return files[0]["id"] if files else None

# Find the 'cal_certificates' folder
cal_folder_id = get_folder_id(CALIB_FOLDER_NAME, MAIN_FOLDER_ID)

if cal_folder_id:
    try:
        query = f"'{cal_folder_id}' in parents and mimeType='application/pdf' and trashed=false"
        results = (
            drive_service.files()
            .list(
                q=query,
                fields="files(id, name, modifiedTime)",
                orderBy="modifiedTime desc",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        files = results.get("files", [])

        if files:
            selected_file = st.selectbox("🔍 Select Calibration Report", [f["name"] for f in files])
            file = next((f for f in files if f["name"] == selected_file), None)

            if file:
                st.markdown(f"**📄 {file['name']}**  — *Last Modified: {file['modifiedTime'][:10]}*")
                st.markdown(
                    f'<iframe src="https://drive.google.com/file/d/{file["id"]}/preview" width="100%" height="550"></iframe>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"[🔗 Open in Drive](https://drive.google.com/file/d/{file['id']}/view?usp=sharing)"
                )
        else:
            st.info("No calibration PDFs found in 'cal_certificates'.")

    except Exception as e:
        st.error(f"⚠️ Failed to load calibration reports: {e}")
else:
    st.warning("⚠️ 'cal_certificates' folder not found in Google Drive.")
