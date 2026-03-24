import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.styles       import inject_css, page_header, section_title, RISK_COLORS, RISK_ICONS, RISK_LIGHT
from utils.gauge        import draw_gauge
from risk.stratification import (preprocess_patient, predict_risk,
                                  whatif_predict, get_whatif_insights)

st.set_page_config(page_title="What-If Simulator | DiaForecaster AI",
                   page_icon="🔄", layout="wide")
inject_css()
page_header("🔄", "What-If Risk Simulator",
            "Adjust your health values and see how your risk changes in real time")

if 'patient_data' not in st.session_state:
    st.warning("⚠️ Please complete **Patient Input** first.")
    if st.button("Go to Patient Input"):
        st.switch_page("pages/1_Patient_Input.py")
    st.stop()

patient = st.session_state['patient_data']
orig_scaled, _, _ = preprocess_patient(patient)
orig_prob, orig_level = predict_risk(orig_scaled)

st.markdown("---")

slider_col, result_col = st.columns([1.1, 1], gap="large")

with slider_col:
    section_title("🎚️ Adjust Health Values")
    st.caption("Move sliders to simulate lifestyle or treatment changes.")
    st.markdown("")

    with st.container():
        new_bmi = st.slider(
            f"BMI  (current: **{patient['bmi']:.1f}**)",
            15.0, 50.0, float(patient['bmi']), 0.5,
            help="Normal: 18.5–24.9")
        new_glucose = st.slider(
            f"Blood Glucose mg/dL  (current: **{patient['blood_glucose_level']}**)",
            70, 300, int(patient['blood_glucose_level']), 1,
            help="Normal fasting: <100 mg/dL")
        new_hba1c = st.slider(
            f"HbA1c %  (current: **{patient['HbA1c_level']}**)",
            3.5, 10.0, float(patient['HbA1c_level']), 0.1,
            help="Normal: <5.7%")
        new_age = st.slider(
            f"Age  (current: **{patient['age']}**)",
            18, 100, int(patient['age']), 1)

    st.markdown("")
    chk1, chk2 = st.columns(2)
    with chk1:
        new_hypertension  = st.checkbox("Hypertension",  value=bool(patient['hypertension']))
    with chk2:
        new_heart_disease = st.checkbox("Heart Disease",  value=bool(patient['heart_disease']))

    # Status badges
    st.markdown("---")
    def badge(val, ok, warn, label):
        if val <= ok:   return f"🟢 **{label}** — Normal"
        if val <= warn: return f"🟡 **{label}** — Borderline"
        return            f"🔴 **{label}** — High Risk"

    st.markdown(badge(new_bmi,     24.9, 29.9, f"BMI {new_bmi:.1f}"))
    st.markdown(badge(new_glucose,  99,  125,  f"Glucose {new_glucose} mg/dL"))
    st.markdown(badge(new_hba1c,   5.6,  6.4,  f"HbA1c {new_hba1c}%"))

with result_col:
    mods = {
        'bmi': new_bmi, 'blood_glucose_level': new_glucose,
        'HbA1c_level': new_hba1c, 'age': new_age,
        'hypertension': new_hypertension, 'heart_disease': new_heart_disease,
    }
    new_prob, new_level = whatif_predict(patient, mods)
    delta   = new_prob - orig_prob
    insight = get_whatif_insights(orig_prob, new_prob, orig_level, new_level)

    section_title("📊 Risk Comparison")
    st.markdown("")

    # Before / After cards
    bc, ac = st.columns(2)
    with bc:
        c  = RISK_COLORS[orig_level]
        bg = RISK_LIGHT[orig_level]
        st.markdown(f"""
        <div class="dia-card" style="border-top:4px solid {c}; background:{bg};
             text-align:center; padding:1.25rem;">
            <div style="font-size:0.75rem; font-weight:700; color:#64748b;
                 letter-spacing:0.05em; text-transform:uppercase;">BEFORE</div>
            <div style="font-size:2.2rem; margin:0.3rem 0;">{RISK_ICONS[orig_level]}</div>
            <div style="font-size:1rem; font-weight:700; color:{c};">{orig_level}</div>
            <div style="font-size:1.6rem; font-weight:800; color:#0f172a;">
                {orig_prob*100:.1f}%
            </div>
        </div>""", unsafe_allow_html=True)

    with ac:
        c  = RISK_COLORS[new_level]
        bg = RISK_LIGHT[new_level]
        arr   = '▼' if delta < 0 else '▲'
        clr   = '#059669' if delta < 0 else '#dc2626'
        st.markdown(f"""
        <div class="dia-card" style="border-top:4px solid {c}; background:{bg};
             text-align:center; padding:1.25rem;">
            <div style="font-size:0.75rem; font-weight:700; color:#64748b;
                 letter-spacing:0.05em; text-transform:uppercase;">AFTER</div>
            <div style="font-size:2.2rem; margin:0.3rem 0;">{RISK_ICONS[new_level]}</div>
            <div style="font-size:1rem; font-weight:700; color:{c};">{new_level}</div>
            <div style="font-size:1.6rem; font-weight:800; color:#0f172a;">
                {new_prob*100:.1f}%
            </div>
            <div style="font-size:0.88rem; font-weight:700; color:{clr};">
                {arr} {abs(delta)*100:.1f}%
            </div>
        </div>""", unsafe_allow_html=True)

    # Insight box
    insight_bg = '#d1fae5' if delta < 0 else '#fee2e2' if delta > 0.05 else '#f1f5f9'
    insight_bdr = '#059669' if delta < 0 else '#dc2626' if delta > 0.05 else '#94a3b8'
    st.markdown(f"""
    <div style="background:{insight_bg}; border-left:4px solid {insight_bdr};
         border-radius:8px; padding:0.75rem 1rem; margin:0.75rem 0;
         font-size:0.9rem; color:#0f172a;">
        💡 {insight}
    </div>""", unsafe_allow_html=True)

    # Gauge pair
    g1c, g2c = st.columns(2)
    with g1c:
        st.caption("Before")
        fig_b = draw_gauge(orig_prob, orig_level, figsize=(3, 2.0))
        st.pyplot(fig_b, use_container_width=True)
        plt.close()
    with g2c:
        st.caption("After")
        fig_a = draw_gauge(new_prob, new_level, figsize=(3, 2.0))
        st.pyplot(fig_a, use_container_width=True)
        plt.close()

# ── Scenario reference ──
st.markdown("---")
section_title("⚡ Scenario Reference")
st.markdown("")
s1, s2, s3 = st.columns(3)
with s1:
    st.markdown("""<div class="dia-card" style="border-top:3px solid #10b981;">
    <b>🥗 Diet & Exercise</b><br>
    <span style="font-size:0.82rem;color:#64748b;">BMI −3 · Glucose −20 · HbA1c −0.3</span>
    </div>""", unsafe_allow_html=True)
with s2:
    st.markdown("""<div class="dia-card" style="border-top:3px solid #3b82f6;">
    <b>💊 Medication</b><br>
    <span style="font-size:0.82rem;color:#64748b;">Glucose −30 · HbA1c −0.6</span>
    </div>""", unsafe_allow_html=True)
with s3:
    st.markdown("""<div class="dia-card" style="border-top:3px solid #ef4444;">
    <b>📈 No Change (5 yr)</b><br>
    <span style="font-size:0.82rem;color:#64748b;">Age +5 · BMI +1 · Glucose +8</span>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
if st.button("📈 See My 5-Year Trajectory", use_container_width=True):
    st.switch_page("pages/4_5Year_Trajectory.py")
