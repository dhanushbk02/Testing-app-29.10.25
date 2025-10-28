# pages/Instruments.py
import streamlit as st
import pandas as pd
import sqlite3
import os
import base64
import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta
from werkzeug.utils import secure_filename
import mimetypes
import streamlit.components.v1 as components
from typing import List, Optional
from datetime import date, timedelta
import pandas as pd

def is_valid_date(d):
    """Return a clean datetime.date if valid, else None."""
    if pd.isna(d):
        return None
    if isinstance(d, pd.Timestamp):
        d = d.date()
    if not isinstance(d, date):
        return None
    return d
from datetime import date, timedelta
import pandas as pd

def safe_date(d):
    """Convert NaT/Timestamp to plain date, return None if invalid."""
    if pd.isna(d):
        return None
    if isinstance(d, pd.Timestamp):
        d = d.date()
    if isinstance(d, date):
        return d
    return None

# =========================
# Page config
# =========================
st.set_page_config(page_title="Instrument Details", page_icon="🧰", layout="wide")

# =========================
# Paths (project-root relative)
# =========================
APP_ROOT = Path(__file__).resolve().parents[1]   # folder that contains /pages
DB_PATH = APP_ROOT / "database" / "instruments.db"
UPLOAD_ROOT = APP_ROOT / "uploads" / "cal_certificates"
IMPORTS_DIR = APP_ROOT / "database" / "imports"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
IMPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Exact Excel (your file lives in database/imports)
EXCEL_FILE = IMPORTS_DIR / "Calibration_Calender_20251013_205701.xlsx"

TODAY = date.today()

# =========================
# DB helpers
# =========================
def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def discover_calibration_pdfs(root: Path):
    """
    Scan a folder for calibration reports (PDF).
    Returns a DataFrame: [instrument_id, filename, filepath, modified_at].
    - If PDFs are stored inside subfolders named after instrument IDs -> auto-detect IDs.
    - If all PDFs are in one flat folder, ID = filename stem (before first underscore/dash).
    """
    records = []
    if not root.exists():
        return pd.DataFrame(columns=["instrument_id", "filename", "filepath", "modified_at"])

    for path in root.rglob("*.pdf"):
        # derive instrument id
        inst_id = path.parent.name  # folder name
        if inst_id == "cal_certificates":
            # flat folder mode
            inst_id = path.stem.split("_")[0].split("-")[0]
        records.append({
            "instrument_id": inst_id,
            "filename": path.name,
            "filepath": str(path),
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime)
        })

    return pd.DataFrame(records)

def _col_exists(cur, table: str, col: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    return any(r[1] == col for r in cur.fetchall())

def init_db():
    with get_connection() as conn:
        cur = conn.cursor()
        # Original table + new fields for the requested form
        cur.execute("""
        CREATE TABLE IF NOT EXISTS instruments (
            instrument_id TEXT PRIMARY KEY,
            instrument_name TEXT,
            location TEXT,
            calibration_due DATE,
            last_calibrated DATE,
            status TEXT,
            remarks TEXT
        )
        """)
        for new_col, col_type in [
            ("make", "TEXT"),
            ("model", "TEXT"),
            ("serial_no", "TEXT"),
            ("range_spec", "TEXT"),
            ("calibration_date", "DATE"),
            ("used_for", "TEXT"),
        ]:
            if not _col_exists(cur, "instruments", new_col):
                cur.execute(f"ALTER TABLE instruments ADD COLUMN {new_col} {col_type}")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS certificates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instrument_id TEXT,
            filename TEXT,
            filepath TEXT,
            uploaded_at TIMESTAMP,
            FOREIGN KEY (instrument_id) REFERENCES instruments(instrument_id) ON DELETE CASCADE
        )
        """)
        conn.commit()

def load_instruments_df():
    with get_connection() as conn:
        try:
            df = pd.read_sql_query(
                "SELECT * FROM instruments",
                conn,
                parse_dates=["calibration_due", "last_calibrated", "calibration_date"]
            )
        except Exception:
            cols = ["instrument_id","instrument_name","location","calibration_due","last_calibrated",
                    "status","remarks","make","model","serial_no","range_spec","calibration_date","used_for"]
            return pd.DataFrame(columns=cols)

    if df.empty:
        cols = ["instrument_id","instrument_name","location","calibration_due","last_calibrated",
                "status","remarks","make","model","serial_no","range_spec","calibration_date","used_for"]
        return pd.DataFrame(columns=cols)

    for c in ["calibration_due","last_calibrated","calibration_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    return df

def upsert_instrument(row: dict):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO instruments (instrument_id, instrument_name, make, model, serial_no, range_spec,
                                 calibration_date, location, used_for,
                                 calibration_due, last_calibrated, status, remarks)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(instrument_id) DO UPDATE SET
          instrument_name=excluded.instrument_name,
          make=excluded.make,
          model=excluded.model,
          serial_no=excluded.serial_no,
          range_spec=excluded.range_spec,
          calibration_date=excluded.calibration_date,
          location=excluded.location,
          used_for=excluded.used_for,
          calibration_due=excluded.calibration_due,
          last_calibrated=excluded.last_calibrated,
          status=excluded.status,
          remarks=excluded.remarks
        """, (
            row.get("instrument_id"),
            row.get("instrument_name"),
            row.get("make"),
            row.get("model"),
            row.get("serial_no"),
            row.get("range_spec"),
            row.get("calibration_date"),
            row.get("location"),
            row.get("used_for"),
            row.get("calibration_due"),
            row.get("last_calibrated"),
            row.get("status"),
            row.get("remarks"),
        ))
        conn.commit()

def save_certificate_record(instrument_id, filename, filepath):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO certificates (instrument_id, filename, filepath, uploaded_at)
        VALUES (?, ?, ?, ?)
        """, (instrument_id, filename, str(filepath), datetime.now()))
        conn.commit()

def list_certificates(instrument_id):
    with get_connection() as conn:
        try:
            df = pd.read_sql_query(
                "SELECT id, instrument_id, filename, filepath, uploaded_at FROM certificates WHERE instrument_id = ? ORDER BY uploaded_at DESC",
                conn, params=(instrument_id,)
            )
        except Exception:
            df = pd.DataFrame(columns=["id","instrument_id","filename","filepath","uploaded_at"])
    return df
def discover_calibration_pdfs(root: Path):
    """
    Scan a folder for calibration reports (PDF).
    Returns a DataFrame: [instrument_id, filename, filepath, modified_at].
    - If PDFs are stored inside subfolders named after instrument IDs -> auto-detect IDs.
    - If all PDFs are in one flat folder, ID = filename stem (before first underscore/dash).
    """
    records = []
    if not root.exists():
        return pd.DataFrame(columns=["instrument_id", "filename", "filepath", "modified_at"])

    for path in root.rglob("*.pdf"):
        # derive instrument id
        inst_id = path.parent.name  # folder name (default)
        if inst_id == "cal_certificates":
            # flat folder mode: extract ID from filename
            inst_id = path.stem.split("_")[0].split("-")[0]
        records.append({
            "instrument_id": inst_id,
            "filename": path.name,
            "filepath": str(path),
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime)
        })

    return pd.DataFrame(records)

# =========================
# Utilities (Excel handling)
# =========================
def _normalize(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    return " ".join(s.strip().split()).lower()

from datetime import datetime, date, timedelta
import pandas as pd
from pathlib import Path

def _load_excel_as_is(xl_path: Path):
    """
    Load calibration Excel (header at A3:F3).
    Cleans, renames, and parses 'Calibration Due Date'.
    Handles missing, duplicate, or NaT values safely.
    """

    if not xl_path or not xl_path.exists():
        return pd.DataFrame(), None, pd.Series(dtype="datetime64[ns]")

    # --- Read Excel (header starts from row 3) ---
    df = pd.read_excel(xl_path, header=2)
    df.columns = [str(c).strip() for c in df.columns]

    # --- Rename columns to consistent readable names ---
    rename_map = {
        df.columns[0]: "Instrument ID",
        df.columns[1]: "Instrument Name",
        df.columns[2]: "Make/Model",
        df.columns[3]: "Serial Number",
    }

    # Find calibration column dynamically
    cal_col = None
    for c in df.columns:
        c_norm = c.strip().lower().replace(".", "").replace(" ", "")
        if "calibration" in c_norm and "due" in c_norm:
            cal_col = c
            break
    if cal_col is None:
        possible_date_cols = [
            c for c in df.columns if any(k in c.lower() for k in ["date", "due", "cal"])
        ]
        cal_col = possible_date_cols[-1] if possible_date_cols else df.columns[-1]

    rename_map[cal_col] = "Calibration Due Date"
    df = df.rename(columns=rename_map)

    # --- Clean ---
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df = df.dropna(subset=["Instrument ID"], how="any")
    df["Instrument ID"] = df["Instrument ID"].astype(str).str.strip().str.upper()
    df = df.drop_duplicates(subset=["Instrument ID"], keep="first").reset_index(drop=True)

    # --- Parse Calibration Due Date ---
    def parse_date_safe(val):
        if pd.isna(val):
            return None
        if isinstance(val, (pd.Timestamp, datetime)):
            return val.date()
        if isinstance(val, (float, int)):
            try:
                return pd.to_datetime(val, origin="1899-12-30", unit="D").date()
            except Exception:
                return None
        if isinstance(val, str):
            for fmt in ("%d.%m.%Y", "%d-%m-%Y", "%Y-%m-%d"):
                try:
                    return datetime.strptime(val.strip(), fmt).date()
                except ValueError:
                    continue
        return None

    df["Calibration Due Date"] = df["Calibration Due Date"].apply(parse_date_safe)

    return df, "Calibration Due Date", df["Calibration Due Date"]



def save_uploaded_file(uploaded_file, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = uploaded_file.name
    path = dest_dir / fname
    i = 1
    while path.exists():
        path = dest_dir / f"{path.stem}_{i}{path.suffix}"
        i += 1
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path, path.name

# =========================
# Init
# =========================
init_db()

# =========================
# Header
# =========================
st.title("🧰 Instrument Details")
st.caption("Excel is read directly from /database/imports with header at row 2. The table renders exactly like your sheet.")

# Diagnostics
st.caption(f"Looking for Excel at: `{EXCEL_FILE}`")
if not EXCEL_FILE.exists():
    files_seen = sorted(p.name for p in IMPORTS_DIR.glob("*.xlsx"))
    st.warning("Excel file not found in /database/imports. Files there:")
    st.code("\n".join(files_seen) or "(no .xlsx files)")
    # Fallback: use latest Calibration_Calender_*.xlsx if available
    candidates = sorted(IMPORTS_DIR.glob("Calibration_Calender_*.xlsx"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True)
    if candidates:
        EXCEL_FILE = candidates[0]
        st.info(f"Using fallback file: {EXCEL_FILE.name}")

# Load Excel AS-IS (header row 2)
excel_df, cal_due_col, cal_due_parsed = _load_excel_as_is(EXCEL_FILE)

# =========================
# Dashboard (NaT-safe, from 'Calibration due date')
# =========================
from datetime import date, timedelta
import pandas as pd

def safe_date(d):
    """Convert NaT/Timestamp/string to datetime.date or None."""
    if pd.isna(d) or d in ["", " ", None]:
        return None
    if isinstance(d, pd.Timestamp):
        d = d.date()
    if isinstance(d, str):
        try:
            d = pd.to_datetime(d, errors="coerce").date()
        except Exception:
            return None
    if isinstance(d, date):
        return d
    return None


TODAY = date.today()

total_instruments = len(excel_df) if not excel_df.empty else 0
pending_count = 0
upcoming_count = 0

if not excel_df.empty and cal_due_col:
    parsed_list = [safe_date(d) for d in cal_due_parsed]

    mask_pending = pd.Series(
        [d and d < TODAY for d in parsed_list],
        index=excel_df.index
    )
    mask_upcoming = pd.Series(
        [d and TODAY <= d <= TODAY + timedelta(days=30) for d in parsed_list],
        index=excel_df.index
    )

    pending_count = int(mask_pending.sum())
    upcoming_count = int(mask_upcoming.sum())

m1, m2, m3 = st.columns(3)
m1.metric("Total instruments", total_instruments)
m2.metric("Calibration Pending", pending_count)
m3.metric("Upcoming calibration instruments", upcoming_count)

st.markdown("---")

left_col, right_col = st.columns([2, 1])

# =========================
# LEFT: Excel list AS-IS + Form
# =========================
with left_col:
    st.subheader("📋 Calibration Register (Excel view)")
    if excel_df.empty:
        st.info("No data to display. Ensure the Excel file exists.")
    else:
        with st.expander("Filter (optional)", expanded=False):
            q = st.text_input("Search text (matches any column)")
            only_pending = st.checkbox("Show only: Calibration Pending")
            only_upcoming = st.checkbox("Show only: Upcoming (≤ 30 days)")

        df_view = excel_df.copy()
        if q:
            qs = q.strip().lower()
            df_view = df_view[df_view.apply(lambda r: any(qs in str(v).lower() for v in r.values), axis=1)]

        if cal_due_col:
            safe_dates = [safe_date(d) for d in cal_due_parsed]

            mask_pending = pd.Series(
                [(d is None) or (d and d < TODAY) for d in safe_dates],
                index=excel_df.index
            )
            mask_upcoming = pd.Series(
                [d and TODAY <= d <= TODAY + timedelta(days=30) for d in safe_dates],
                index=excel_df.index
            )

            if only_pending:
                df_view = df_view[mask_pending]
            if only_upcoming:
                df_view = df_view[mask_upcoming]

        st.dataframe(df_view, use_container_width=True, height=420)


    st.subheader("➕ Add / Update Instrument")
    st.caption("Form saves to the database (for certificates & details). Excel view remains unchanged.")
    with st.form("instrument_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            inst_id = st.text_input("ID (unique)", key="inst_id")
            inst_name = st.text_input("Name of the instrument")
            make = st.text_input("Make")
        with c2:
            model = st.text_input("Model")
            serial_no = st.text_input("Sl. no.")
            range_spec = st.text_input("Range")
        with c3:
            calib_date = st.date_input("Calibration date", value=None, format="YYYY-MM-DD")
            location = st.text_input("Location")
            used_for = st.text_input("Instrument used for")

        # Legacy fields (kept for compatibility; not shown in Excel)
        calibration_due = None
        last_calibrated = None
        status = None
        remarks = None

        submitted = st.form_submit_button("Save Instrument")
        if submitted:
            if not inst_id or not inst_name:
                st.error("ID and Name of the instrument are required.")
            else:
                row = {
                    "instrument_id": inst_id.strip(),
                    "instrument_name": inst_name.strip(),
                    "make": make.strip() if make else None,
                    "model": model.strip() if model else None,
                    "serial_no": serial_no.strip() if serial_no else None,
                    "range_spec": range_spec.strip() if range_spec else None,
                    "calibration_date": calib_date.isoformat() if isinstance(calib_date, date) else None,
                    "location": location.strip() if location else None,
                    "used_for": used_for.strip() if used_for else None,
                    "calibration_due": calibration_due,
                    "last_calibrated": last_calibrated,
                    "status": status,
                    "remarks": remarks,
                }
                upsert_instrument(row)
                st.success(f"Saved instrument {inst_id}.")
                st.rerun()

# =========================
# RIGHT: Calibration Report Viewer (with PDF preview + download)
# =========================
with right_col:
    st.subheader("📁 Calibration Reports")

    df_db = load_instruments_df()

    if "instrument_id" not in df_db.columns:
        st.error("⚠️ 'instrument_id' column not found in database.")
    else:
        instrument_ids = (
            df_db["instrument_id"]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )

        instr_choice = st.selectbox("🔎 View by Instrument ID", options=["-- Select --"] + instrument_ids)

        if instr_choice and instr_choice != "-- Select --":
            inst_row = df_db[df_db["instrument_id"].astype(str) == instr_choice]
            if not inst_row.empty:
                r = inst_row.iloc[0]
                st.markdown(f"**Instrument ID:** {r.get('instrument_id', '')}")
                cal_date = r.get("calibration_date", "")
                if pd.notna(cal_date) and str(cal_date).lower() != "nat":
                    st.markdown(f"**Calibration Date:** {cal_date}")

            # --- Auto-detect calibration PDFs ---
            pdf_df = discover_calibration_pdfs(UPLOAD_ROOT)
            if pdf_df.empty:
                st.info("No calibration reports found in uploads/cal_certificates.")
            else:
                def normalize_id(s):
                    if not isinstance(s, str):
                        return ""
                    return s.lower().replace("-", "").replace("_", "").strip()

                match_df = pdf_df[pdf_df["instrument_id"].apply(normalize_id) == normalize_id(instr_choice)]
                if match_df.empty:
                    match_df = pdf_df[pdf_df["filename"].str.lower().str.contains(normalize_id(instr_choice))]

                if match_df.empty:
                    st.warning(f"No calibration report found for Instrument ID **{instr_choice}**.")
                else:
                    st.success(f"Found {len(match_df)} calibration report(s):")

                    # --- Show PDF previews + download buttons ---
                    for _, row in match_df.iterrows():
                        pdf_path = os.path.join(UPLOAD_ROOT, row["filename"])
                        with open(pdf_path, "rb") as f:
                            pdf_bytes = f.read()

                        st.markdown(
                            f"**📄 {row['filename']}**  "
                            f"*(last modified: {row['modified_at'].strftime('%d-%b-%Y %H:%M')})*"
                        )
                        # PDF preview (full width)
                        st.components.v1.html(
                            f"""
                            <iframe src="data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode()}" 
                                    width="100%" height="600px" 
                                    style="border: 1px solid #ccc; border-radius: 8px;">
                            </iframe>
                            """,
                            height=620,
                        )

                        # Download button
                        st.download_button(
                            label=f"⬇ Download {row['filename']}",
                            data=pdf_bytes,
                            file_name=row["filename"],
                            mime="application/pdf",
                            use_container_width=True,
                        )
                        st.divider()
        else:
            st.info("Select an Instrument ID to view its auto-detected calibration reports.")

# =========================
# Footer
# =========================
st.markdown("---")
st.caption("Header row is 2 (A2). Key date column: ‘Calibration due date’. The register mirrors the Excel exactly.")
