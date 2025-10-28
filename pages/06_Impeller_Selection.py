import streamlit as st
import pandas as pd

st.set_page_config(page_title="Impeller Selection", page_icon="⚙️", layout="wide")

st.title("⚙️ Impeller Selection Module")

st.markdown("""
This tool helps you select the right **impeller size and trim** based on performance requirements.
Provide the following parameters to calculate or filter recommended impeller sizes.
""")

col1, col2, col3 = st.columns(3)

with col1:
    flow = st.number_input("Flow Rate (LPM)", min_value=0.0, value=100.0)
with col2:
    head = st.number_input("Head (m)", min_value=0.0, value=10.0)
with col3:
    motor_speed = st.number_input("Motor Speed (RPM)", min_value=0.0, value=2900.0)

st.markdown("---")

if st.button("🔍 Find Suitable Impeller"):
    # Placeholder formula or database lookup
    dia = (flow * head / motor_speed) ** 0.5 * 10
    st.success(f"Recommended Impeller Diameter: **{dia:.2f} mm**")
    st.info("This recommendation is based on nominal calculation. Please verify with engineering data.")

st.markdown("---")

uploaded_file = st.file_uploader("📤 Upload Pump Curve Data (optional, CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
    st.write("Uploaded pump curve data loaded successfully.")
