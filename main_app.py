import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px

# ------------------------------------------------------
# Page Configuration
# ------------------------------------------------------
st.set_page_config(
    page_title="FPL Testing Dashboard",
    page_icon="🧪",
    layout="wide"
)

# ------------------------------------------------------
# Header Section
# ------------------------------------------------------
st.title("🧪 Flow Oil Pumps Pvt. Ltd. — Testing Department Dashboard")
st.markdown("""
Welcome to the **Testing Department Dashboard**.  
Here you can monitor weekly testing activity, report generation, equipment calibration, and staff availability.
""")

st.divider()

# ------------------------------------------------------
# Generate Mock Weekly Data (to be replaced with DB data later)
# ------------------------------------------------------
today = datetime.now()
dates = [today - timedelta(days=i) for i in range(6, -1, -1)]  # last 7 days

# Non-conformance data (per day)
nc_data = [2, 3, 1, 4, 0, 2, 1]
# Test report generation count
report_data = [5, 8, 6, 9, 10, 7, 12]
# Type test status (summary)
type_test_status = {"Completed": 28, "Ongoing": 5, "Planned": 7}
# Instruments due (by calibration month)
instrument_due = {"Due Soon": 3, "Up-to-date": 47, "Overdue": 2}
# Staff availability
staff_today = {
    "Assistant Manager": ["Dhanush BK"],
    "Test Engineers": ["Ravi Kumar", "Anand", "Sandeep"],
    "Testing Assistants": ["Manoj", "Pradeep"]
}

# ------------------------------------------------------
# Summary Metrics Section
# ------------------------------------------------------
st.subheader("📊 Weekly Summary")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🚫 Non-Conformances (This Week)", sum(nc_data), delta="-2 vs Last Week")
with col2:
    st.metric("📄 Reports Generated (This Week)", sum(report_data), delta="+5 vs Last Week")
with col3:
    st.metric("⚙️ Type Tests (Completed)", type_test_status["Completed"])

st.divider()

# ------------------------------------------------------
# Charts Section
# ------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["📅 Weekly Trends", "📈 Test & Type Summary", "👷 Staff Availability"])

# --- Tab 1: Weekly Trends ---
with tab1:
    colA, colB = st.columns(2)

    # Non-Conformance trend
    with colA:
        df_nc = pd.DataFrame({"Date": [d.strftime("%b %d") for d in dates], "Count": nc_data})
        fig_nc = px.bar(
            df_nc,
            x="Date",
            y="Count",
            title="Weekly Non-Conformance Trend",
            color="Count",
            color_continuous_scale="Reds"
        )
        fig_nc.update_layout(xaxis_title="", yaxis_title="No. of Issues", template="plotly_white")
        st.plotly_chart(fig_nc, use_container_width=True)

    # Report generation trend
    with colB:
        df_rep = pd.DataFrame({"Date": [d.strftime("%b %d") for d in dates], "Reports": report_data})
        fig_rep = px.line(
            df_rep,
            x="Date",
            y="Reports",
            title="Daily Test Report Generation",
            markers=True,
            line_shape="spline",
        )
        fig_rep.update_layout(xaxis_title="", yaxis_title="Reports", template="plotly_white")
        st.plotly_chart(fig_rep, use_container_width=True)

# --- Tab 2: Type Test & Instrument Overview ---
with tab2:
    colC, colD = st.columns(2)

    with colC:
        # Pie chart for type tests
        fig_type = px.pie(
            names=list(type_test_status.keys()),
            values=list(type_test_status.values()),
            title="Type Test Progress Summary",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig_type, use_container_width=True)

    with colD:
        # Pie chart for instruments
        fig_inst = px.pie(
            names=list(instrument_due.keys()),
            values=list(instrument_due.values()),
            title="Instrument Calibration Status",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_inst, use_container_width=True)

import streamlit as st

st.subheader("👷‍♂️ Staff Availability Today")

# Fixed Assistant Manager (always available)
st.markdown("**Assistant Manager:** Dhanush ✅ *(General Shift: 8:30 AM – 5:00 PM)*")

st.markdown("---")

# Other staff list
staff_members = [
    "Kiran (Test Engineer)",
    "Suresh (Test Engineer)",
    "Ravi (Testing Assistant)",
    "Vikas (Testing Assistant)",
    "Anand (Technician)",
]

shifts = ["General (8:30 AM – 5:00 PM)", "1st Shift (6 AM – 2 PM)", "2nd Shift (2 PM – 10 PM)"]

availability = {}

for staff in staff_members:
    col1, col2 = st.columns([2, 1.5])
    with col1:
        available = st.checkbox(f"{staff}", value=False, key=f"avail_{staff}")
    with col2:
        if available:
            shift = st.selectbox(f"Shift for {staff.split('(')[0].strip()}", shifts, key=f"shift_{staff}")
            availability[staff] = shift

# Summary Display
if any(availability):
    st.markdown("### ✅ Available Staff Summary:")
    for name, shift in availability.items():
        st.markdown(f"- **{name}** — {shift}")
else:
    st.info("No staff marked available yet.")


# ------------------------------------------------------
# Footer
# ------------------------------------------------------
st.caption(f"© {datetime.now().year} Flow Oil Pumps Pvt. Ltd. | Testing Department | Developed by Dhanush BK")
