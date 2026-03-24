import streamlit as st
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.styles import inject_css, page_header, section_title

st.set_page_config(page_title="Patient Input | DiaForecaster AI",
                   page_icon="📋", layout="wide")
inject_css()
page_header("📋", "Patient Health Input",
            "Enter your health details — all fields are used to compute your future diabetes risk")

st.markdown("---")

# ── Two-column layout ──
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("""
    <div class="dia-card">
        <div style="font-weight:700; font-size:1rem; color:#0f172a; margin-bottom:1rem;">
            👤 Personal & Lifestyle
        </div>""", unsafe_allow_html=True)

    age    = st.number_input("Age (years)", min_value=18, max_value=100, value=45, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    smoking_history = st.selectbox("Smoking History",
        ["never", "former", "current", "not current", "No Info", "ever"])

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("")

    st.markdown("""
    <div class="dia-card">
        <div style="font-weight:700; font-size:1rem; color:#0f172a; margin-bottom:1rem;">
            🏥 Medical History
        </div>""", unsafe_allow_html=True)

    hypertension  = st.checkbox("Diagnosed with Hypertension",  value=False)
    heart_disease = st.checkbox("Diagnosed with Heart Disease",  value=False)

    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="dia-card">
        <div style="font-weight:700; font-size:1rem; color:#0f172a; margin-bottom:1rem;">
            🩺 Clinical Biomarkers
        </div>""", unsafe_allow_html=True)

    bmi = st.number_input(
        "BMI (kg/m²)", min_value=10.0, max_value=60.0, value=27.5, step=0.1,
        help="Normal: 18.5–24.9 | Overweight: 25–29.9 | Obese: ≥30")
    hba1c = st.number_input(
        "HbA1c Level (%)", min_value=3.0, max_value=15.0, value=5.5, step=0.1,
        help="Normal: <5.7% | Pre-diabetic: 5.7–6.4% | Diabetic: ≥6.5%")
    glucose = st.number_input(
        "Blood Glucose Level (mg/dL)", min_value=50, max_value=400, value=120, step=1,
        help="Normal: <100 | Pre-diabetic: 100–125 | Diabetic: ≥126")

    st.markdown("</div>", unsafe_allow_html=True)

    # Reference panel
    st.markdown("""
    <div class="dia-card" style="background:#f8fafc;">
        <div style="font-weight:700; font-size:0.88rem; color:#475569; margin-bottom:0.75rem;">
            📖 Clinical Reference Ranges
        </div>
        <table style="font-size:0.8rem; color:#64748b; width:100%; border-collapse:collapse;">
            <tr><td style="padding:3px 0;"><b>HbA1c</b></td>
                <td>Normal &lt;5.7% · Pre 5.7–6.4% · Diabetic ≥6.5%</td></tr>
            <tr><td style="padding:3px 0;"><b>Glucose</b></td>
                <td>Normal &lt;100 · Pre 100–125 · Diabetic ≥126 mg/dL</td></tr>
            <tr><td style="padding:3px 0;"><b>BMI</b></td>
                <td>Normal 18.5–24.9 · Overweight 25–29.9 · Obese ≥30</td></tr>
            <tr><td style="padding:3px 0;"><b>Age Risk</b></td>
                <td>Increases significantly after 45 years</td></tr>
        </table>
    </div>""", unsafe_allow_html=True)

# ── Summary expander ──
st.markdown("---")
with st.expander("📝 Review Input Before Predicting"):
    s1, s2, s3 = st.columns(3)
    with s1:
        st.write(f"**Age:** {age}")
        st.write(f"**Gender:** {gender}")
        st.write(f"**Smoking:** {smoking_history}")
    with s2:
        st.write(f"**BMI:** {bmi}")
        st.write(f"**HbA1c:** {hba1c}%")
        st.write(f"**Glucose:** {glucose} mg/dL")
    with s3:
        st.write(f"**Hypertension:** {'Yes ⚠️' if hypertension else 'No ✅'}")
        st.write(f"**Heart Disease:** {'Yes ⚠️' if heart_disease else 'No ✅'}")

# ── Predict button ──
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    if st.button("🔍  Predict My Diabetes Risk", type="primary", use_container_width=True):
        st.session_state['patient_data'] = {
            'age': age, 'gender': gender, 'bmi': bmi,
            'hypertension': hypertension, 'heart_disease': heart_disease,
            'smoking_history': smoking_history,
            'HbA1c_level': hba1c, 'blood_glucose_level': glucose,
        }
        st.switch_page("pages/2_Risk_Results.py")
