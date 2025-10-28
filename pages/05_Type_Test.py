# pages/TypeTest.py
import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
from werkzeug.utils import secure_filename
import mimetypes
import base64
import io
import streamlit.components.v1 as components

# ======================================================
# Page config
# ======================================================
st.set_page_config(page_title="Pump Type Test", page_icon="🧪", layout="wide")

# ======================================================
# Paths / constants
# ======================================================
BASE_DIR = Path.cwd()
DB_PATH = BASE_DIR / "database" / "instruments.db"
UPLOAD_ROOT = BASE_DIR / "uploads" / "type_tests"
IMPORTS_DIR = BASE_DIR / "database" / "imports"  # <- as requested
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
IMPORTS_DIR.mkdir(parents=True, exist_ok=True)

FILE_INTERNAL = IMPORTS_DIR / "Pump Type Test data internal.xlsx"
FILE_EXTERNAL = IMPORTS_DIR / "Pump Type Test data external.xlsx"

TODAY = date.today()

# ======================================================
# DB helpers
# ======================================================
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def _column_exists(cur, table: str, col: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    return any(r[1] == col for r in cur.fetchall())

def init_db():
    with get_connection() as conn:
        cur = conn.cursor()
        # Main table (add test_type)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS type_tests (
            model TEXT PRIMARY KEY,
            model_desc TEXT,
            type_test_date DATE,
            expiry_date DATE,
            engineer TEXT,
            status TEXT,
            notes TEXT
        )
        """)
        # Migration: add test_type if missing
        if not _column_exists(cur, "type_tests", "test_type"):
            cur.execute("ALTER TABLE type_tests ADD COLUMN test_type TEXT")
        # Files table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS type_test_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT,
            filename TEXT,
            filepath TEXT,
            uploaded_at TIMESTAMP,
            FOREIGN KEY (model) REFERENCES type_tests(model)
        )
        """)
        conn.commit()

@st.cache_data(show_spinner=False)
def load_type_tests_df_cached() -> pd.DataFrame:
    with get_connection() as conn:
        try:
            df = pd.read_sql_query(
                "SELECT * FROM type_tests",
                conn,
                parse_dates=["type_test_date", "expiry_date"]
            )
        except Exception:
            df = pd.DataFrame(columns=["model","model_desc","type_test_date","expiry_date","engineer","status","notes","test_type"])

    if df.empty:
        return df

    for c in ["type_test_date","expiry_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date

    df["status"] = df.apply(lambda r: compute_status(r.get("type_test_date"), r.get("expiry_date")), axis=1)
    if "test_type" not in df.columns:
        df["test_type"] = None
    return df

def invalidate_main_cache():
    load_type_tests_df_cached.clear()

def upsert_type_test(row: dict):
    # compute status if not provided
    if not row.get("status") or row.get("status") == "Auto":
        row["status"] = compute_status(row.get("type_test_date"), row.get("expiry_date"))

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO type_tests (model, model_desc, type_test_date, expiry_date, engineer, status, notes, test_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(model) DO UPDATE SET
          model_desc=excluded.model_desc,
          type_test_date=excluded.type_test_date,
          expiry_date=excluded.expiry_date,
          engineer=excluded.engineer,
          status=excluded.status,
          notes=excluded.notes,
          test_type=excluded.test_type
        """, (
            row.get("model"),
            row.get("model_desc"),
            row.get("type_test_date"),
            row.get("expiry_date"),
            row.get("engineer"),
            row.get("status"),
            row.get("notes"),
            row.get("test_type"),
        ))
        conn.commit()
    invalidate_main_cache()

def delete_type_test(model: str):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM type_test_files WHERE model=?", (model,))
        cur.execute("DELETE FROM type_tests WHERE model=?", (model,))
        conn.commit()
    invalidate_main_cache()

def save_type_file(model: str, filename: str, filepath: Path):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO type_test_files (model, filename, filepath, uploaded_at)
        VALUES (?, ?, ?, ?)
        """, (model, filename, str(filepath), datetime.now()))
        conn.commit()

@st.cache_data(show_spinner=False)
def list_type_files_cached(model: str) -> pd.DataFrame:
    with get_connection() as conn:
        try:
            df = pd.read_sql_query(
                "SELECT id, model, filename, filepath, uploaded_at FROM type_test_files WHERE model=? ORDER BY uploaded_at DESC",
                conn, params=(model,)
            )
        except Exception:
            df = pd.DataFrame(columns=["id","model","filename","filepath","uploaded_at"])
    return df

def invalidate_files_cache():
    list_type_files_cached.clear()

# ======================================================
# Utilities
# ======================================================
def compute_status(tt_date: Optional[date], exp_date: Optional[date]) -> str:
    if tt_date is None or pd.isna(tt_date):
        return "Pending"
    if exp_date is None or pd.isna(exp_date):
        return "Active"
    try:
        exp = exp_date if isinstance(exp_date, date) else pd.to_datetime(exp_date, errors="coerce").date()
    except Exception:
        return "Active"
    return "Active" if exp >= TODAY else "Expired"

def coerce_date(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        d = pd.to_datetime(val, errors="coerce")
        return None if pd.isna(d) else d.date()
    except Exception:
        return None

def secure_save(uploaded_file, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = secure_filename(uploaded_file.name)
    path = dest_dir / fname
    counter = 1
    while path.exists():
        path = dest_dir / f"{Path(fname).stem}_{counter}{Path(fname).suffix}"
        counter += 1
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path, path.name

def pdf_iframe_from_path(path: Path, height: int = 420) -> None:
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        src = f"data:application/pdf;base64,{b64}"
        components.html(f'<iframe src="{src}" width="100%" height="{height}px"></iframe>', height=height + 20)
    except Exception:
        st.info("PDF preview not available.")

# ---------- import helpers (silent mapping to DB) ----------
def normalize_header(h: str) -> str:
    if not isinstance(h, str):
        h = "" if h is None else str(h)
    return " ".join(h.strip().split()).lower()

def guess_mapping(cols_norm: List[str]) -> Dict[str, Optional[str]]:
    synonyms = {
        "model": ["model","pump model","model no","model number","pump code","code"],
        "model_desc": ["description","model desc","model description","variant","pump description"],
        "type_test_date": ["type test date","typetest date","type_test_date","tested on","test date"],
        "expiry_date": ["expiry date","validity","valid till","valid upto","valid up to","expiry","validity end","expiration"],
        "engineer": ["engineer","responsible","owner","tested by","approved by"],
        "status": ["status"],
        "notes": ["notes","remarks","comment","comments","note"],
    }
    mapping = {}
    for key, cands in synonyms.items():
        hit = next((c for c in cands if c in cols_norm), None)
        mapping[key] = hit
    return mapping

def _row_to_payload(row: pd.Series, mapping: Dict[str, Optional[str]], test_type: str) -> Optional[dict]:
    if not mapping.get("model"):
        return None
    model_raw = row.get(mapping["model"])
    if model_raw is None or (isinstance(model_raw, float) and pd.isna(model_raw)):
        return None
    model = str(model_raw).strip()
    if not model:
        return None
    payload = {
        "model": model,
        "model_desc": (str(row.get(mapping["model_desc"])).strip()
                       if mapping.get("model_desc") and not pd.isna(row.get(mapping["model_desc"])) else None),
        "type_test_date": (coerce_date(row.get(mapping["type_test_date"])) if mapping.get("type_test_date") else None),
        "expiry_date": (coerce_date(row.get(mapping["expiry_date"])) if mapping.get("expiry_date") else None),
        "engineer": (str(row.get(mapping["engineer"])).strip()
                     if mapping.get("engineer") and not pd.isna(row.get(mapping["engineer"])) else None),
        "status": (str(row.get(mapping["status"])).strip()
                   if mapping.get("status") and not pd.isna(row.get(mapping["status"])) else None),
        "notes": (str(row.get(mapping["notes"])).strip()
                  if mapping.get("notes") and not pd.isna(row.get(mapping["notes"])) else None),
        "test_type": test_type
    }
    return payload

def silent_sync_excel_to_db(df: Optional[pd.DataFrame], test_type: str) -> int:
    """Try best-effort import so certificates panel keeps working.
       Does not alter the display format (register still shows Excel 'as-is')."""
    if df is None or df.empty:
        return 0
    cols_norm = [normalize_header(c) for c in df.columns]
    mapping = guess_mapping(cols_norm)
    count = 0
    for _, r in df.iterrows():
        payload = _row_to_payload(r, mapping, test_type)
        if payload:
            upsert_type_test(payload)
            count += 1
    return count

# ======================================================
# Init
# ======================================================
init_db()

# ======================================================
# Header / KPIs (from DB)
# ======================================================
st.title("🧪 Pump Type Test")
st.caption("Internal & External type tests. Excel files are auto-loaded from database/imports and listed below exactly as provided.")

# Try to load Excel files (as-is for display)
df_internal_raw, df_external_raw = None, None
internal_info, external_info = "", ""

if FILE_INTERNAL.exists():
    try:
        df_internal_raw = pd.read_excel(FILE_INTERNAL, dtype=str)
        internal_info = f"Loaded **{FILE_INTERNAL.name}** ({len(df_internal_raw)} rows, {len(df_internal_raw.columns)} cols)"
    except Exception as e:
        st.error(f"Could not read '{FILE_INTERNAL.name}': {e}")

if FILE_EXTERNAL.exists():
    try:
        df_external_raw = pd.read_excel(FILE_EXTERNAL, dtype=str)
        external_info = f"Loaded **{FILE_EXTERNAL.name}** ({len(df_external_raw)} rows, {len(df_external_raw.columns)} cols)"
    except Exception as e:
        st.error(f"Could not read '{FILE_EXTERNAL.name}': {e}")

# Silent DB sync so the rest of the app (certificates) works
synced_counts = []
if df_internal_raw is not None:
    synced_counts.append(("Internal", silent_sync_excel_to_db(df_internal_raw, "Internal")))
if df_external_raw is not None:
    synced_counts.append(("External", silent_sync_excel_to_db(df_external_raw, "External")))

# KPIs (from DB after sync)
df_all = load_type_tests_df_cached()
active = int((df_all["status"] == "Active").sum()) if not df_all.empty else 0
expired = int((df_all["status"] == "Expired").sum()) if not df_all.empty else 0
pending = int((df_all["status"] == "Pending").sum()) if not df_all.empty else 0
due_30 = 0
if not df_all.empty and ("expiry_date" in df_all.columns):
    due_30 = int(df_all["expiry_date"].apply(lambda d: (isinstance(d, date) and (TODAY <= d <= TODAY + timedelta(days=30)))).sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Active", active)
m2.metric("Expired", expired)
m3.metric("Pending", pending)
m4.metric("Expiring ≤ 30 days", due_30)
if synced_counts:
    st.caption("Auto-synced to DB: " + ", ".join([f"{t}: {n}" for t, n in synced_counts]))

st.markdown("---")

# ======================================================
# LEFT/RIGHT layout
# ======================================================
left, right = st.columns([2, 1])

# ======================================================
# LEFT: Auto Import status + Type Test Register (Excel format preserved)
# ======================================================
with left:
    st.subheader("📥 Import Type Test List (Auto)")
    colA, colB = st.columns(2)
    with colA:
        if internal_info:
            st.success(internal_info)
        else:
            st.info("Internal file not found: **Pump Type Test data internal.xlsx**")
    with colB:
        if external_info:
            st.success(external_info)
        else:
            st.info("External file not found: **Pump Type Test data external.xlsx**")

    st.markdown("— The register below mirrors the Excel files **exactly** (columns/order). No manual mapping needed.")
    st.markdown("")

    st.subheader("📋 Type Test Register (Excel view)")
    tabs = st.tabs(["Type Test (Internal)", "Type Test (External)"])

    with tabs[0]:
        if df_internal_raw is None or df_internal_raw.empty:
            st.info("No Internal list available.")
        else:
            st.dataframe(df_internal_raw, use_container_width=True, height=420)

    with tabs[1]:
        if df_external_raw is None or df_external_raw.empty:
            st.info("No External list available.")
        else:
            st.dataframe(df_external_raw, use_container_width=True, height=420)

    st.markdown("---")
    st.subheader("➕ Add / Update Model (DB)")
    st.caption("Optional: Maintain specific fields for certificates/expiry tracking. Register view above still follows Excel.")
    with st.form("tt_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            model = st.text_input("Model (unique key)", key="tt_model")
            model_desc = st.text_input("Description / Variant")
        with c2:
            type_test_date = st.date_input("Type Test Date", value=None, format="YYYY-MM-DD")
            engineer = st.text_input("Engineer / Owner")
        with c3:
            expiry_date = st.date_input("Expiry Date", value=None, format="YYYY-MM-DD")
            status_manual = st.selectbox("Status (leave 'Auto' to compute)", ["Auto","Active","Expired","Pending"], index=0)

        test_type = st.selectbox("Type Test", ["Internal", "External"], index=0)
        notes = st.text_area("Notes")

        cA, cB = st.columns([1,1])
        with cA:
            submit = st.form_submit_button("💾 Save Model")
        with cB:
            del_click = st.form_submit_button("🗑️ Delete Model")

        if submit:
            if not model:
                st.error("Model is required.")
            else:
                payload = {
                    "model": model.strip(),
                    "model_desc": (model_desc.strip() if model_desc else None),
                    "type_test_date": (type_test_date.isoformat() if isinstance(type_test_date, date) else None),
                    "expiry_date": (expiry_date.isoformat() if isinstance(expiry_date, date) else None),
                    "engineer": (engineer.strip() if engineer else None),
                    "status": None if status_manual == "Auto" else status_manual,
                    "notes": notes.strip() if notes else None,
                    "test_type": test_type
                }
                upsert_type_test(payload)
                st.success(f"Saved model {model}.")
                st.rerun()

        if del_click:
            if not model:
                st.error("Enter the Model you want to delete, then click Delete.")
            else:
                delete_type_test(model.strip())
                st.success(f"Deleted model {model}.")
                st.rerun()

# ======================================================
# RIGHT: Certificates / Reports (uses DB for models)
# ======================================================
with right:
    st.subheader("📑 Certificates / Reports")

    models = load_type_tests_df_cached()["model"].astype(str).tolist() if not df_all.empty else []
    chosen = st.selectbox("Select model", options=["-- Select --"] + models)

    if chosen and chosen != "-- Select --":
        # quick details
        row = load_type_tests_df_cached()
        row = row[row["model"] == chosen]
        if not row.empty:
            rec = row.iloc[0]
            st.markdown(f"**Model:** {rec.get('model') or ''}")
            st.markdown(f"**Description:** {rec.get('model_desc') or ''}")
            st.markdown(f"**Type:** {rec.get('test_type') or ''}")
            st.markdown(f"**Type Test Date:** {rec.get('type_test_date') or ''}")
            st.markdown(f"**Expiry Date:** {rec.get('expiry_date') or ''}")
            st.markdown(f"**Status:** {rec.get('status') or ''}")
            st.markdown(f"**Engineer:** {rec.get('engineer') or ''}")
            if rec.get('notes'):
                st.markdown(f"**Notes:** {rec.get('notes')}")
        else:
            st.info("Model details not found.")

        # existing files
        files_df = list_type_files_cached(chosen)
        st.markdown("**Uploaded files**")
        if files_df.empty:
            st.info("No files uploaded for this model.")
        else:
            for _, r in files_df.iterrows():
                file_id = int(r["id"])
                fname = r["filename"]
                fpath = Path(r["filepath"])
                st.write(f"{fname}  •  uploaded: {r['uploaded_at']}")
                mime = mimetypes.guess_type(str(fpath))[0]

                # Previews
                if mime and mime.startswith("image"):
                    try:
                        st.image(str(fpath), use_column_width=True)
                    except Exception:
                        st.write("Image preview not available.")
                elif mime == "application/pdf":
                    pdf_iframe_from_path(fpath, height=420)

                # Actions
                colD1, colD2 = st.columns([1,1])
                with colD1:
                    try:
                        with open(fpath, "rb") as f:
                            st.download_button("⬇️ Download", data=f.read(), file_name=fname, key=f"dl_{file_id}")
                    except Exception:
                        st.warning("File missing on disk.")
                with colD2:
                    if st.button("🗑️ Delete file", key=f"del_{file_id}"):
                        # delete immediately without stale cache
                        with get_connection() as conn:
                            cur = conn.cursor()
                            cur.execute("SELECT filepath FROM type_test_files WHERE id=?", (file_id,))
                            fr = cur.fetchone()
                            if fr:
                                p = Path(fr[0])
                                try:
                                    if p.exists():
                                        p.unlink()
                                except Exception:
                                    pass
                            cur.execute("DELETE FROM type_test_files WHERE id=?", (file_id,))
                            conn.commit()
                        invalidate_files_cache()
                        st.success("Deleted file.")
                        st.rerun()

        st.markdown("---")
        st.markdown("**Upload certificate / report**")
        upl = st.file_uploader("Upload (pdf / jpg / png)", type=["pdf","jpg","jpeg","png"], key=f"upl_{chosen}")
        if upl:
            dest = UPLOAD_ROOT / chosen
            path, saved_name = secure_save(upl, dest)
            save_type_file(chosen, saved_name, path)
            invalidate_files_cache()
            st.success(f"Saved {saved_name} for {chosen}")
            st.rerun()
    else:
        st.info("Select a model to view/upload certificates.")
