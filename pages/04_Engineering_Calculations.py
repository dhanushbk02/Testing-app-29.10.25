# pages/4_Engineering_Calculations.py
import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="Engineering Calculations", page_icon="🧮", layout="wide")

st.title("🧮 Engineering Calculations Workbench")
st.markdown("A set of quick engineering calculation tools for testing engineers. Enter values and results appear instantly.")

st.divider()

# -------------------------
# Helper converters & constants
# -------------------------
g = 9.81  # gravity m/s^2

def lpm_to_m3s(lpm: float) -> float:
    """Convert LPM (liters per minute) to m^3/s."""
    return float(lpm) / 1000.0 / 60.0

def kg_to_newton(kg: float) -> float:
    return float(kg) * 9.81

def mm_to_m(mm: float) -> float:
    return float(mm) / 1000.0

# -------------------------
# Section 1: Efficiency Calculation
# -------------------------
with st.expander("1. Efficiency Calculation", expanded=False):
    st.markdown("Compute pump efficiency using the formula:\n\n"
                "`Pump Efficiency (%) = ((0.0001409 * Flow(LPM) * Head(m)) / InputPower(kW)) * 100`")
    col1, col2, col3 = st.columns(3)
    with col1:
        flow_lpm = st.number_input("Flow (LPM)", min_value=0.0, value=100.0, step=1.0, key="eff_flow")
    with col2:
        head_m = st.number_input("Head (m)", min_value=0.0, value=10.0, step=0.1, key="eff_head")
    with col3:
        input_kw = st.number_input("Input Power (kW)", min_value=0.0001, value=1.0, step=0.01, key="eff_kw")

    try:
        eff = ((0.0001409 * flow_lpm * head_m) / input_kw) * 100.0
        st.success(f"Pump Efficiency: **{eff:.2f} %**")
    except Exception as e:
        st.error(f"Calculation error: {e}")

st.divider()

# -------------------------
# Section 2: Impeller Calculation (Affinity Law)
# -------------------------
with st.expander("2. Impeller Calculation (Affinity Law)", expanded=False):
    st.markdown("Affinity laws: \n- Q ∝ D * N\n- H ∝ (D * N)^2\n- P ∝ (D * N)^3\nYou can change impeller diameter or speed to predict new performance.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Known (base case)")
        D1_mm = st.number_input("Impeller Dia D1 (mm)", value=200.0, min_value=1.0, key="imp_d1")
        Q1 = st.number_input("Flow Q1 (LPM)", value=500.0, min_value=0.0, key="imp_q1")
        H1 = st.number_input("Head H1 (m)", value=40.0, min_value=0.0, key="imp_h1")
        RPM1 = st.number_input("Speed N1 (RPM)", value=1450.0, min_value=1.0, key="imp_n1")
        P1_kw = st.number_input("Power P1 (kW) [optional]", value=0.0, min_value=0.0, key="imp_p1")
    with c2:
        st.subheader("Target (change diameter or speed)")
        D2_mm = st.number_input("Impeller Dia D2 (mm)", value=180.0, min_value=1.0, key="imp_d2")
        RPM2 = st.number_input("Speed N2 (RPM)", value=RPM1, min_value=1.0, key="imp_n2")
        compute_by = st.radio("Predict by:", ["Diameter change (D2)", "Speed change (N2)", "Both"], index=0)

    try:
        # Using affinity laws: scale factors
        kD = (D2_mm / D1_mm)
        kN = (RPM2 / RPM1)
        scale = kD * kN  # if both changed Q ∝ D*N
        # But typically Q scales as (D2/D1)*(N2/N1)
        Q2 = Q1 * (kD * kN)
        H2 = H1 * (kD * kN) ** 2
        if P1_kw and P1_kw > 0:
            P2 = P1_kw * (kD * kN) ** 3
        else:
            P2 = None

        st.write(f"Predicted Flow Q2: **{Q2:.2f} LPM**")
        st.write(f"Predicted Head H2: **{H2:.2f} m**")
        if P2 is not None:
            st.write(f"Predicted Power P2: **{P2:.3f} kW**")
        else:
            st.info("Provide P1 (kW) to estimate P2 by affinity law (P ∝ (D*N)^3).")
    except Exception as e:
        st.error(f"Error in affinity calculation: {e}")

st.divider()

# -------------------------
# Section 3: Resistance at Different Temperatures
# -------------------------
with st.expander("3. Resistance at Different Temperatures", expanded=False):
    st.markdown("Calculate changed resistance with temperature using: `R2 = R1 * (1 + α*(T2 - T1))`")
    r1 = st.number_input("Cold Resistance R1 (Ω)", value=0.5, min_value=0.0, key="res_r1")
    t1 = st.number_input("Cold Temperature T1 (°C)", value=25.0, key="res_t1")
    t2 = st.number_input("Target Temperature T2 (°C)", value=75.0, key="res_t2")
    alpha = st.number_input("Temperature coefficient α (per °C)", value=0.0040, format="%.6f", key="res_alpha")

    try:
        r2 = r1 * (1 + alpha * (t2 - t1))
        st.success(f"Resistance at {t2}°C → R2 = **{r2:.6f} Ω**")
    except Exception as e:
        st.error(f"Calculation error: {e}")

st.divider()

# -------------------------
# Section 4: Temperature Rise Calculation
# -------------------------
with st.expander("4. Temperature Rise (from resistance)", expanded=False):
    st.markdown("From R1 at T1 and R2 at T2, ΔT = (R2/R1 - 1) / α. Then T_hot = T1 + ΔT.")
    r1_tr = st.number_input("Cold Resistance R1 (Ω)", value=0.5, key="tr_r1")
    t1_tr = st.number_input("Cold Temperature T1 (°C)", value=25.0, key="tr_t1")
    r2_tr = st.number_input("Hot Resistance R2 (Ω)", value=0.6, key="tr_r2")
    alpha_tr = st.number_input("Temperature coefficient α (per °C)", value=0.0040, key="tr_alpha")

    try:
        if r1_tr <= 0:
            st.error("R1 must be > 0")
        else:
            delta_t = (r2_tr / r1_tr - 1.0) / alpha_tr
            t_hot = t1_tr + delta_t
            st.success(f"Temperature Rise ΔT = **{delta_t:.2f} °C** · Estimated Hot Temp = **{t_hot:.2f} °C**")
    except Exception as e:
        st.error(f"Calculation error: {e}")

st.divider()

# -------------------------
# Section 5: Pressure Drop Calculation (ΔP)
# -------------------------
with st.expander("5. Pressure Drop Calculation (ΔP)", expanded=False):
    st.markdown("Compute pressure drop using Darcy-Weisbach. Steps: convert flow, compute velocity, Reynolds, friction factor, head loss, then ΔP = ρ*g*h_f.")
    col1, col2, col3 = st.columns(3)
    with col1:
        flow_lph = st.number_input("Flow (Lph)", value=12000.0, min_value=0.0, key="pd_flow_lph")
        diameter_mm = st.number_input("Pipe Dia (mm)", value=80.0, min_value=1.0, key="pd_d_mm")
    with col2:
        length_m = st.number_input("Pipe Length (m)", value=200.0, key="pd_len")
        density = st.number_input("Density (kg/m³)", value=850.0, key="pd_rho")
    with col3:
        viscosity = st.number_input("Dynamic viscosity μ (Pa·s)", value=0.02, key="pd_mu")

    try:
        Q_m3s = float(flow_lph) / 1000.0 / 3600.0  # Lph -> m3/s
        D_m = mm_to_m(diameter_mm)
        A = math.pi * (D_m ** 2) / 4.0
        V = Q_m3s / A if A > 0 else 0.0
        Re = (density * V * D_m) / viscosity if viscosity > 0 else 0.0

        # friction factor
        if Re == 0:
            f = 0
        elif Re < 2000:
            f = 64.0 / Re  # laminar
        else:
            # Blasius approx for smooth turbulent: valid for Re up to ~1e5
            f = 0.0791 * (Re ** -0.25)

        h_f = f * (length_m / D_m) * (V ** 2) / (2 * g)
        delta_p = density * g * h_f  # Pa
        delta_p_bar = delta_p / 1e5

        st.write(f"- Cross-sectional area A = **{A:.6f} m²**")
        st.write(f"- Flow velocity V = **{V:.4f} m/s**")
        st.write(f"- Reynolds number Re = **{Re:.0f}**")
        st.write(f"- Friction factor f ≈ **{f:.5f}**")
        st.success(f"Pressure Drop ΔP = **{delta_p:.2f} Pa**  (≈ **{delta_p_bar:.6f} bar**)")
    except Exception as e:
        st.error(f"Calculation error: {e}")

st.divider()

# -------------------------
# Section 6: Rotor Balancing (IS 1940:2003) - Single plane example
# -------------------------
with st.expander("6. Rotor Balancing (IS 1940:2003) - Single Plane", expanded=False):
    st.markdown("Calculate permissible residual unbalance (Uper) and allocate to bearing planes for single-plane balancing.")
    col1, col2, col3 = st.columns(3)
    with col1:
        rpm = st.number_input("Rated RPM (rpm)", value=3000.0, key="bal_rpm")
        weight_kg = st.number_input("Rotor Weight (kg)", value=3600.0, key="bal_wt")
        radius_mm = st.number_input("Impeller Radius (mm)", value=61.0, key="bal_rad")
    with col2:
        LA = st.number_input("LA (mm)", value=900.0, key="bal_la")
        LB = st.number_input("LB (mm)", value=1500.0, key="bal_lb")
        L_total = st.number_input("L (mm)", value=2400.0, key="bal_l")
    with col3:
        grade = st.number_input("Grade (e.g., 2.5)", value=2.5, key="bal_grade")
        eper_input = st.number_input("eper (g.mm/kg) [optional, 0 to use Grade]", value=0.0, key="bal_eper")

    try:
        omega = 2.0 * math.pi * rpm / 60.0
        radius_m = mm_to_m(radius_mm)
        if eper_input > 0:
            eper = eper_input
        else:
            # If user hasn't provided eper, we won't invent formula; allow grade to be used as an index
            # Here we use grade as a proxy (user should provide proper eper from graphs in standards)
            eper = grade  # placeholder: user may input proper eper when available

        # Uper: permissible residual unbalance in g·mm (or g.mm/kg * weight_kg ???)
        # Many standards use Uper = eper * weight (where eper is mm*g/kg) -> result in g.mm
        Uper = eper * weight_kg
        UperA = Uper * (LA / L_total) if L_total != 0 else None
        st.write(f"- Angular speed ω = **{omega:.2f} rad/s**")
        st.write(f"- Using eper = **{eper:.3f} (user/grade)**")
        st.success(f"Permissible residual unbalance Uper = **{Uper:.2f} g·mm**")
        if UperA is not None:
            st.write(f"Allocated to plane A (UperA) = **{UperA:.2f} g·mm**")
    except Exception as e:
        st.error(f"Calculation error: {e}")

st.divider()

# -------------------------
# Section 7: Bearing Life Calculation (L10h)
# -------------------------
with st.expander("7. Bearing Life Calculation (L10h)", expanded=False):
    st.markdown("Compute bearing L10 life (hours):\n\n"
                "`L10h = ((C / P) ** p) * (1e6 / (60 * n))` where p = 3 for ball bearings.")
    n_rpm = st.number_input("Speed n (RPM)", value=2850.0, key="bl_n")
    Fr_value = st.number_input("Radial load Fr (N) (or enter kg and check box below)", value=215.14, key="bl_fr")
    Fa_value = st.number_input("Axial load Fa (N)", value=2.384, key="bl_fa")
    with st.expander("Input options / conversions"):
        use_kg = st.checkbox("Enter Fr & Fa as kgf instead of N (convert to N)", value=False, key="bl_use_kg")
    if use_kg:
        Fr = kg_to_newton(Fr_value)
        Fa = kg_to_newton(Fa_value)
    else:
        Fr = Fr_value
        Fa = Fa_value
    X = st.number_input("Load factor X (default 0.5)", value=0.5, key="bl_x")
    Y = st.number_input("Load factor Y (default 1)", value=1.0, key="bl_y")
    p_val = st.number_input("Exponent p (3 for ball bearings)", value=3.0, key="bl_p")

    # Catalog C values (user may change)
    st.markdown("**Catalog Dynamic Load Ratings (C)** (edit if needed)")
    c_6306 = st.number_input("C for 6306 (N)", value=29606.0, key="bl_c6306")
    c_3306 = st.number_input("C for 3306 (N)", value=41506.0, key="bl_c3306")

    try:
        P_eq = X * Fr + Y * Fa
        if P_eq <= 0:
            st.error("Equivalent load P must be > 0")
        else:
            L10_6306_h = ((c_6306 / P_eq) ** p_val) * (1e6 / (60.0 * n_rpm))
            L10_3306_h = ((c_3306 / P_eq) ** p_val) * (1e6 / (60.0 * n_rpm))
            st.write(f"- Equivalent dynamic load P = **{P_eq:.2f} N**")
            st.write("**Estimated life (hours):**")
            st.success(f"• 6306: **{L10_6306_h:,.0f} hours**")
            st.success(f"• 3306: **{L10_3306_h:,.0f} hours**")
    except Exception as e:
        st.error(f"Calculation error: {e}")

st.divider()

# -------------------------
# Section 8: Impeller Selection Tool (Simpler UI)
# -------------------------
with st.expander("8. Impeller Selection Tool", expanded=False):
    st.markdown("Assist selecting impeller diameter/trim to meet a required duty using simple scaling/affinity.")
    req_flow = st.number_input("Required Flow (LPM)", value=500.0, key="is_req_flow")
    req_head = st.number_input("Required Head (m)", value=40.0, key="is_req_head")
    base_dia = st.number_input("Reference Impeller Dia (mm)", value=200.0, key="is_base_dia")
    base_flow = st.number_input("Reference Flow at base dia (LPM)", value=600.0, key="is_base_flow")
    base_head = st.number_input("Reference Head at base dia (m)", value=45.0, key="is_base_head")

    try:
        # Estimate D_needed by scaling Q and H roughly together:
        # If Q ∝ D and H ∝ D^2, we can solve for D from both and take mean
        D_from_Q = base_dia * (req_flow / base_flow)
        D_from_H = base_dia * math.sqrt(req_head / base_head) if base_head > 0 else D_from_Q
        D_est = (D_from_Q + D_from_H) / 2.0
        st.success(f"Estimated Impeller Dia to meet duty ≈ **{D_est:.1f} mm**")
        st.info(f"(From Q => {D_from_Q:.1f} mm, from H => {D_from_H:.1f} mm)")
    except Exception as e:
        st.error(f"Calculation error: {e}")

st.divider()

# -------------------------
# Section 9: Inductance Imbalance Check
# -------------------------
with st.expander("9. Inductance Imbalance Check", expanded=False):
    st.markdown("Enter three phase inductance (mH) values to compute average, max deviation and percent unbalance.")
    col1, col2, col3 = st.columns(3)
    with col1:
        uv = st.number_input("U-V (mH)", value=107.13, key="ind_uv")
    with col2:
        vw = st.number_input("V-W (mH)", value=124.65, key="ind_vw")
    with col3:
        wu = st.number_input("W-U (mH)", value=111.41, key="ind_wu")

    try:
        arr = np.array([uv, vw, wu], dtype=float)
        avg = arr.mean()
        max_dev = np.max(np.abs(arr - avg))
        percent = (max_dev / avg) * 100.0 if avg != 0 else 0.0
        st.write(f"- Average inductance = **{avg:.2f} mH**")
        st.write(f"- Max deviation = **{max_dev:.2f} mH**")
        st.success(f"- Percent unbalance = **{percent:.2f} %**")
    except Exception as e:
        st.error(f"Calculation error: {e}")

st.divider()

# -------------------------
# Section 10: Motor Load Test
# -------------------------
with st.expander("10. Motor Load Test Analysis", expanded=False):
    st.markdown("Enter or paste load test table rows. Torque (Kg-m) will be converted to N·m; output power computed from torque and speed.")
    st.info("Provide data rowwise. You can edit the example and then press Analyze.")
    example = pd.DataFrame({
        "Load %": [25, 50, 75, 100, 125],
        "Torque (Kg-m)": [0.052, 0.104, 0.156, 0.208, 0.260],
        "Pull (Kg)": [0.8, 1.6, 2.4, 3.2, 4.0],
        "Voltage (V)": [415, 415, 415, 415, 415],
        "Current (A)": [0.8, 0.9, 1.2, 1.4, 1.7],
        "Total I/P (kW)": [0.340, 0.501, 0.707, 0.815, 1.124],
        "Speed (RPM)": [2948, 2890, 2802, 2780, 2634],
        "O/P (kW)": [0.16, 0.31, 0.46, 0.60, 0.71],
        "PF": [0.59, 0.77, 0.82, 0.81, 0.92],
        "Freq (Hz)": [50, 50, 50, 50, 50]
    })
    st.write("You can paste a CSV below or edit the example table.")
    edited = st.data_editor(example, num_rows="dynamic", key="motor_table_editor")
    if st.button("Analyze Motor Load Test"):
        try:
            df = edited.copy()
            # convert torque Kg-m to N·m
            df["Torque_Nm"] = df["Torque (Kg-m)"] * 9.80665
            # angular speed rad/s
            df["omega_rad_s"] = 2.0 * math.pi * df["Speed (RPM)"] / 60.0
            # Output power from torque: P = T * omega (W) -> kW
            df["Output_kW_calc"] = (df["Torque_Nm"] * df["omega_rad_s"]) / 1000.0
            # Efficiency = output / input * 100 (use Total I/P kW)
            df["Efficiency_calc_%"] = df["Output_kW_calc"] / df["Total I/P (kW)"] * 100.0
            # Slip calculation requires synchronous speed -> assume 2-pole motor unless user provides
            pole = st.number_input("Pole pairs (enter 2 for 2-pole motor)", min_value=1, value=2, key="motor_pole")
            sync_rpm = 60.0 * df["Freq (Hz)"] / pole
            df["Slip_%"] = (sync_rpm - df["Speed (RPM)"]) / sync_rpm * 100.0

            st.dataframe(df[[
                "Load %", "Torque (Kg-m)", "Torque_Nm", "Speed (RPM)", "Output_kW_calc",
                "Total I/P (kW)", "Efficiency_calc_%", "Slip_%"
            ]], use_container_width=True)

            # Plot efficiency vs load
            fig = px.line(df, x="Load %", y="Efficiency_calc_%", title="Efficiency vs Load (%)", markers=True)
            st.plotly_chart(fig, use_container_width=True)
            st.success("Analysis complete. Review table and chart above.")
        except Exception as e:
            st.error(f"Analysis error: {e}")

st.divider()

st.caption(f"Engineering Calculations Workbench — generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
