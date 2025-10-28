import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Inspection Module", page_icon="🧾", layout="wide")

st.title("🧾 Inspection & Quality Check Module")

st.markdown("""
This section allows entering inspection reports and tracking component quality.
""")

col1, col2 = st.columns(2)
with col1:
    component_name = st.text_input("Component Name")
    batch_no = st.text_input("Batch / Serial No.")
with col2:
    inspector = st.text_input("Inspector Name")
    date = st.date_input("Inspection Date", datetime.today())

status = st.selectbox("Status", ["Pending", "Passed", "Rejected"])
remarks = st.text_area("Remarks / Observations")

if st.button("✅ Save Inspection Record"):
    new_data = {
        "Date": date,
        "Component": component_name,
        "Batch No": batch_no,
        "Inspector": inspector,
        "Status": status,
        "Remarks": remarks,
    }
    st.success("Inspection record saved successfully.")
    st.json(new_data)
