# pages/3_NonConformance.py
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import date, datetime
import io
import re
import os

# ---------------------------
# Paths & constants
# ---------------------------
BASE_DIR = Path.cwd()
DB_XLSX = BASE_DIR / "database" / "non_conformance.xlsx"
MODELS_XLSX = BASE_DIR / "List of models.xlsm"   # expects file at project root
UPLOADS_ROOT = BASE_DIR / "uploads" / "nc"
EXPORTS_DIR = BASE_DIR / "exports"
for p in [DB_XLSX.parent, UPLOADS_ROOT, EXPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

SHEET_MAIN = "nc_records"
SHEET_LOOKUPS = "lookups"

DEFAULT_TYPES = [
    "Winding Resistance", "IR", "HV Failures", "No load failure",
    "Locked rotor failure", "Performance related failure", "Others"
]
DEFAULT_STAGES = ["Testing", "Production", "Incoming", "Assembly", "Others"]
DEFAULT_CLOSURE = ["Pending", "Resolved", "Rejected", "On Hold"]

NC_COLUMNS = [
    "NC_ID","Model","Date","Component_No_or_Sl_No","Type_of_NC",
    "Failure_Description","Test_Stage","Action_Plan_Advised","Action_Taken",
    "Responsible_Person","Target_Closure_Date","Closure_Status",
    "Closure_Date","Attachments","Created_At","Updated_At"
]

# ---------------------------
# Small helpers (no extra deps)
# ---------------------------
def secure_filename(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9._-]", "", name)
    return name or "file"
def _safe_date(val, default=None):
    """Return a Python date or None from mixed inputs (str/NaT/NaN/Timestamp/date)."""
    if val is None:
        return default
    s = str(val).strip()
    if s in ("", "NaT", "nat", "NaN", "nan", "None", "null"):
        return default
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        return default
    if isinstance(ts, pd.Timestamp):
        return ts.date()
    if isinstance(ts, datetime):
        return ts.date()
    if isinstance(ts, date):
        return ts
    return default
def _safe_date(val, default=None):
    """Return a Python date or None from mixed inputs (str/NaT/NaN/Timestamp/date)."""
    if val is None:
        return default
    s = str(val).strip()
    if s in ("", "NaT", "nat", "NaN", "nan", "None", "null"):
        return default
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        return default
    if isinstance(ts, pd.Timestamp):
        return ts.date()
    if isinstance(ts, datetime):
        return ts.date()
    if isinstance(ts, date):
        return ts
    return default


def load_models_from_excel(path: Path = MODELS_XLSX) -> list[str]:
    """
    Read models from Column A starting A2. If that yields nothing, try a few
    fallbacks (named columns, no header). Returns a sorted unique list.
    """
    try:
        if not path.exists():
            return []
        # Primary: first sheet, col A, A1 is header, take A2↓
        df = pd.read_excel(path, sheet_name=0, usecols=[0], header=0)  # requires openpyxl
        vals = df.iloc[1:, 0].dropna().astype(str).str.strip()

        # Fallback 1: header=None; skip first row explicitly
        if vals.empty:
            df2 = pd.read_excel(path, sheet_name=0, usecols=[0], header=None)
            vals = df2.iloc[1:, 0].dropna().astype(str).str.strip()

        # Fallback 2: named columns
        if vals.empty:
            df3 = pd.read_excel(path, sheet_name=0)
            for col in ["Model", "Models", "MODEL", "model"]:
                if col in df3.columns:
                    vals = df3[col].dropna().astype(str).str.strip()
                    break

        models = sorted({v for v in vals if v})
        return models
    except Exception:
        return []

def today_str():
    return date.today().strftime("%Y-%m-%d")

def ensure_workbook():
    """Create workbook with sheets & headers if missing."""
    if not DB_XLSX.exists():
        with pd.ExcelWriter(DB_XLSX, engine="openpyxl") as xw:
            pd.DataFrame(columns=NC_COLUMNS).to_excel(xw, index=False, sheet_name=SHEET_MAIN)
            lookups = {
                "Type_of_NC": DEFAULT_TYPES,
                "Test_Stage": DEFAULT_STAGES,
                "Closure_Status": DEFAULT_CLOSURE,
            }
            pd.DataFrame(dict([(k, pd.Series(v)) for k, v in lookups.items()])).to_excel(
                xw, index=False, sheet_name=SHEET_LOOKUPS
            )

def load_lookups():
    ensure_workbook()
    try:
        df = pd.read_excel(DB_XLSX, sheet_name=SHEET_LOOKUPS)
        def col_list(col, fallback):
            return [x for x in df.get(col, pd.Series(dtype=object)).dropna().astype(str).tolist()] or fallback
        return {
            "Type_of_NC": col_list("Type_of_NC", DEFAULT_TYPES),
            "Test_Stage": col_list("Test_Stage", DEFAULT_STAGES),
            "Closure_Status": col_list("Closure_Status", DEFAULT_CLOSURE),
        }
    except Exception:
        return {"Type_of_NC": DEFAULT_TYPES, "Test_Stage": DEFAULT_STAGES, "Closure_Status": DEFAULT_CLOSURE}

def load_records() -> pd.DataFrame:
    ensure_workbook()
    try:
        df = pd.read_excel(DB_XLSX, sheet_name=SHEET_MAIN, dtype=str)
    except Exception:
        df = pd.DataFrame(columns=NC_COLUMNS)
    # normalize columns
    for c in NC_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[NC_COLUMNS]
    return df

def write_records(df: pd.DataFrame):
    # Keep lookups sheet intact
    lookups_df = pd.read_excel(DB_XLSX, sheet_name=SHEET_LOOKUPS)
    with pd.ExcelWriter(DB_XLSX, engine="openpyxl", mode="w") as xw:
        df.to_excel(xw, index=False, sheet_name=SHEET_MAIN)
        lookups_df.to_excel(xw, index=False, sheet_name=SHEET_LOOKUPS)

def next_nc_id(df: pd.DataFrame) -> str:
    """Generate NC-YYYY-#### unique id."""
    year = datetime.now().year
    pattern = re.compile(rf"^NC-{year}-(\d{{4}})$")
    seq = 0
    for x in df["NC_ID"].astype(str).tolist():
        m = pattern.match(x)
        if m:
            seq = max(seq, int(m.group(1)))
    return f"NC-{year}-{seq+1:04d}"

def df_bool(val):
    return str(val).lower() in ("true", "1", "yes")

# ---------------------------
# Page title & top controls
# ---------------------------
st.title("🚩 Non-Conformance (NC)")

# Model selection (dropdown from Excel + manual entry fallback)
if "recent_models" not in st.session_state:
    st.session_state.recent_models = []

left_top, right_top = st.columns([2, 2])

with left_top:
    models_from_xlsx = load_models_from_excel()

    # Debug/helpful status + one-time fallback upload when list is empty
    st.caption(f"Looking for: '{MODELS_XLSX.name}' at: {MODELS_XLSX}  |  exists: {MODELS_XLSX.exists()}")
    st.caption(f"Loaded {len(models_from_xlsx)} models from Excel")
    if len(models_from_xlsx) == 0:
        st.warning("Couldn’t load models from the file. You can upload the list here temporarily (Column A, header in A1, data from A2↓).")
        mdl_upl = st.file_uploader("Upload 'List of models.xlsx' (one-time)", type=["xlsx"], key="mdl_upl_fallback")
        if mdl_upl is not None:
            try:
                df_tmp = pd.read_excel(mdl_upl, sheet_name=0, usecols=[0], header=0)
                vals = df_tmp.iloc[1:, 0].dropna().astype(str).str.strip()
                if vals.empty:
                    df_tmp2 = pd.read_excel(mdl_upl, sheet_name=0, header=None, usecols=[0])
                    vals = df_tmp2.iloc[1:, 0].dropna().astype(str).str.strip()
                models_from_xlsx = sorted({v for v in vals if v})
                st.success(f"Loaded {len(models_from_xlsx)} models from uploaded file")
            except Exception as e:
                st.error(f"Couldn’t read uploaded file: {e}")

    default_model = st.selectbox(
        "Model (from 'List of models.xlsx')",
        options=["-- Select --"] + models_from_xlsx,
        index=0
    )
    if default_model != "-- Select --":
        model_input = default_model
    else:
        model_input = st.text_input("Or enter a model manually", placeholder="Type or paste model name…")

    if model_input and model_input not in st.session_state.recent_models:
        st.session_state.recent_models = [model_input] + st.session_state.recent_models[:9]
    recent = st.session_state.recent_models
    if recent:
        st.caption("Recent:")
        st.write(", ".join(recent))

# Load data & KPIs
df_all = load_records()
total_n = len(df_all)
resolved_n = (df_all["Closure_Status"] == "Resolved").sum()
pending_n = (df_all["Closure_Status"].isin(["Pending","On Hold"])).sum()
resolved_pct = f"{(resolved_n/total_n*100):.0f}%" if total_n else "—"

with right_top:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total NCs", total_n)
    k2.metric("Resolved", resolved_n)
    k3.metric("Pending", pending_n)
    k4.metric("Resolved %", resolved_pct)

st.divider()

# ---------------------------
# Layout: left form / right table
# ---------------------------
left, right = st.columns([2, 2])

lookups = load_lookups()

# keep current edit NC_ID in session
if "edit_nc_id" not in st.session_state:
    st.session_state.edit_nc_id = None

# -------- LEFT: create/edit form --------
with left:
    st.subheader("Create / Edit NC")

    # Prefill if editing
editing = st.session_state.edit_nc_id is not None
if editing and not df_all.empty:
    mask = df_all["NC_ID"] == st.session_state.edit_nc_id
    editing_rec = df_all.loc[mask].iloc[0] if mask.any() else None
else:
    editing_rec = None

def prefill(col, default=""):
    return ("" if editing_rec is None else str(editing_rec[col])) or default

model = model_input or prefill("Model")
col1, col2 = st.columns(2)
with col1:
    date_val = st.date_input(
        "Date",
        value=(date.today() if editing_rec is None else pd.to_datetime(prefill("Date") or today_str()).date())
    )
    comp_no = st.text_input("Component number or Sl. No.", value=prefill("Component_No_or_Sl_No"))
    type_nc = st.selectbox(
        "Type of NC",
        options=lookups["Type_of_NC"],
        index=0 if editing_rec is None else max(
            0,
            lookups["Type_of_NC"].index(prefill("Type_of_NC")) if prefill("Type_of_NC") in lookups["Type_of_NC"] else 0
        )
    )
    test_stage = st.selectbox(
        "Test stage",
        options=lookups["Test_Stage"],
        index=0 if editing_rec is None else max(
            0,
            lookups["Test_Stage"].index(prefill("Test_Stage")) if prefill("Test_Stage") in lookups["Test_Stage"] else 0
        )
    )
    target_close = st.date_input(
        "Target closure date",
        value=(None if (editing_rec is None or not prefill("Target_Closure_Date"))
               else pd.to_datetime(prefill("Target_Closure_Date")).date())
    )
with col2:
    resp = st.text_input("Responsible person", value=prefill("Responsible_Person"))
    closure_status = st.selectbox(
        "Closure status",
        options=lookups["Closure_Status"],
        index=(lookups["Closure_Status"].index(prefill("Closure_Status"))
               if prefill("Closure_Status") in lookups["Closure_Status"] else 0)
    )
    closure_date_enabled = closure_status in ["Resolved", "Rejected"]
    closure_date = st.date_input(
        "Closure date",
        value=(None if (editing_rec is None or not prefill("Closure_Date"))
               else pd.to_datetime(prefill("Closure_Date")).date()),
        disabled=not closure_date_enabled
    )

    fail_desc = st.text_area("Failure description", value=prefill("Failure_Description"), height=120, placeholder="Describe the failure or observation…")
    action_plan = st.text_area("Action plan advised", value=prefill("Action_Plan_Advised"), height=100)
    action_taken = st.text_area("Action taken", value=prefill("Action_Taken"), height=100)

    st.markdown("**Attachments** (PDF/JPG/PNG)")
    # --- Upload OR capture from camera ---
    col_upl, col_cam = st.columns([2, 1])
    with col_upl:
        files = st.file_uploader("Upload one or more files", type=["pdf","jpg","jpeg","png"], accept_multiple_files=True)
    with col_cam:
        cam_img = st.camera_input("Capture photo")

    # Merge uploaded files + camera capture into a single list
    all_files = list(files) if files else []
    if cam_img is not None:
        all_files.append(cam_img)

    cbtn1, cbtn2 = st.columns(2)
    with cbtn1:
        if st.button("💾 Save / Update NC", use_container_width=True):
            # validations
            if not model:
                st.error("Model is required.")
            elif not date_val:
                st.error("Date is required.")
            elif not type_nc:
                st.error("Type of NC is required.")
            elif not fail_desc:
                st.error("Failure description is required.")
            elif closure_status in ["Resolved","Rejected"] and not closure_date:
                st.error("Closure date required for Resolved/Rejected.")
            else:
                df = load_records()
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if editing_rec is None:
                    nc_id = next_nc_id(df)
                    new_row = {
                        "NC_ID": nc_id,
                        "Model": model,
                        "Date": str(date_val),
                        "Component_No_or_Sl_No": comp_no,
                        "Type_of_NC": type_nc,
                        "Failure_Description": fail_desc,
                        "Test_Stage": test_stage,
                        "Action_Plan_Advised": action_plan,
                        "Action_Taken": action_taken,
                        "Responsible_Person": resp,
                        "Target_Closure_Date": (str(target_close) if target_close else ""),
                        "Closure_Status": closure_status,
                        "Closure_Date": (str(closure_date) if closure_date_enabled and closure_date else ""),
                        "Attachments": "",
                        "Created_At": now,
                        "Updated_At": now,
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    st.success(f"Created {nc_id}")
                else:
                    nc_id = editing_rec["NC_ID"]
                    idx = df[df["NC_ID"] == nc_id].index[0]
                    df.loc[idx, :] = [
                        nc_id,
                        model,
                        str(date_val),
                        comp_no,
                        type_nc,
                        fail_desc,
                        test_stage,
                        action_plan,
                        action_taken,
                        resp,
                        (str(target_close) if target_close else ""),
                        closure_status,
                        (str(closure_date) if closure_date_enabled and closure_date else ""),
                        df.loc[idx, "Attachments"],  # keep existing; will update below
                        df.loc[idx, "Created_At"],
                        now,
                    ]
                    st.success(f"Updated {nc_id}")

                # Handle attachments (upload + camera)
                if all_files:
                    attach_dir = UPLOADS_ROOT / nc_id
                    attach_dir.mkdir(parents=True, exist_ok=True)
                    saved = []
                    for f in all_files:
                        fname = secure_filename(f.name)
                        target = attach_dir / fname
                        # ensure unique
                        counter = 1
                        while target.exists():
                            target = attach_dir / f"{Path(fname).stem}_{counter}{Path(fname).suffix}"
                            counter += 1
                        with open(target, "wb") as out:
                            out.write(f.getbuffer())
                        saved.append(str(target))
                    # merge with existing
                    ix = df[df["NC_ID"] == nc_id].index[0]
                    existing = [p for p in str(df.loc[ix, "Attachments"]).split(",") if p.strip()]
                    df.loc[ix, "Attachments"] = ",".join(existing + saved)

                # finally write workbook
                try:
                    write_records(df)
                    # stay on record
                    st.session_state.edit_nc_id = nc_id
                    st.rerun()
                except PermissionError:
                    st.error("Excel file is open in another program. Please close it and try again.")
                except Exception as e:
                    st.error(f"Failed to write Excel: {e}")

    with cbtn2:
        if st.button("🧹 Reset Form", use_container_width=True):
            st.session_state.edit_nc_id = None
            st.rerun()

# -------- RIGHT: filters + table + exports --------
with right:
    st.subheader("Browse & Filter")

    df = load_records()

    # Filters
    f1, f2 = st.columns(2)
    with f1:
        date_from = st.date_input("From date", value=None)
    with f2:
        date_to = st.date_input("To date", value=None)

    f3, f4 = st.columns(2)
    with f3:
        type_filter = st.multiselect("Type of NC", options=lookups["Type_of_NC"], default=lookups["Type_of_NC"])
    with f4:
        status_filter = st.multiselect("Closure status", options=lookups["Closure_Status"], default=lookups["Closure_Status"])

    search = st.text_input("Search (Model / Description / Component / Responsible)")

    df_view = df.copy()

    # apply filters
    if date_from:
        df_view = df_view[pd.to_datetime(df_view["Date"], errors="coerce") >= pd.to_datetime(date_from)]
    if date_to:
        df_view = df_view[pd.to_datetime(df_view["Date"], errors="coerce") <= pd.to_datetime(date_to)]
    if type_filter:
        df_view = df_view[df_view["Type_of_NC"].isin(type_filter)]
    if status_filter:
        df_view = df_view[df_view["Closure_Status"].isin(status_filter)]
    if search:
        s = search.lower()
        df_view = df_view[df_view.apply(lambda r:
            s in str(r["Model"]).lower()
            or s in str(r["Failure_Description"]).lower()
            or s in str(r["Component_No_or_Sl_No"]).lower()
            or s in str(r["Responsible_Person"]).lower()
        , axis=1)]

    st.caption(f"Showing {len(df_view)} of {len(df)} records")
    st.dataframe(
        df_view[["NC_ID","Model","Date","Type_of_NC","Closure_Status","Responsible_Person","Target_Closure_Date"]]
        .sort_values(by=["Date","NC_ID"], ascending=False),
        use_container_width=True
    )

    st.markdown("**Row actions**")
    colA, colB, colC = st.columns(3)
    with colA:
        edit_id = st.text_input("Enter NC_ID to Edit", placeholder="e.g., NC-2025-0001")
    with colB:
        view_id = st.text_input("Enter NC_ID to View Attachments", placeholder="e.g., NC-2025-0001")
    with colC:
        st.write("")

    act1, act2, act3 = st.columns(3)
    with act1:
        if st.button("✏️ Load for Edit"):
            if edit_id and edit_id in df["NC_ID"].values:
                st.session_state.edit_nc_id = edit_id
                st.rerun()
            else:
                st.warning("NC_ID not found.")
    with act2:
        if st.button("📂 Show Attachments"):
            if view_id and view_id in df["NC_ID"].values:
                rec = df[df["NC_ID"] == view_id].iloc[0]
                paths = [p for p in str(rec["Attachments"]).split(",") if p.strip()]
                if not paths:
                    st.info("No attachments saved.")
                else:
                    for p in paths:
                        pth = Path(p)
                        st.write(pth.name)
                        try:
                            with open(pth, "rb") as f:
                                st.download_button("Download", data=f.read(), file_name=pth.name, key=f"dl_{pth.name}_{view_id}")
                        except Exception:
                            st.warning(f"Missing file: {pth}")
            else:
                st.warning("NC_ID not found.")
    with act3:
        # Export current view
        if st.button("⬇️ Export filtered (Excel & CSV)"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            xlsx_path = EXPORTS_DIR / f"nc_export_{ts}.xlsx"
            csv_path = EXPORTS_DIR / f"nc_export_{ts}.csv"
            try:
                # write Excel
                with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
                    df_view.to_excel(xw, index=False, sheet_name="nc_filtered")
                # write CSV
                df_view.to_csv(csv_path, index=False)
                # provide downloads
                with open(xlsx_path, "rb") as fx:
                    st.download_button("Download Excel", data=fx.read(), file_name=xlsx_path.name, key="exp_xlsx")
                with open(csv_path, "rb") as fc:
                    st.download_button("Download CSV", data=fc.read(), file_name=csv_path.name, key="exp_csv")
                st.success("Exported successfully.")
            except PermissionError:
                st.error("Cannot write export files (permission denied). Close any open files and try again.")
            except Exception as e:
                st.error(f"Export failed: {e}")
