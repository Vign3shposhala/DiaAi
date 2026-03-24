import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import joblib, sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.styles  import inject_css, page_header, risk_banner, section_title, RISK_COLORS, RISK_ICONS
from utils.gauge   import draw_gauge
from risk.stratification        import preprocess_patient, predict_risk
from explainability.shap_analysis import compute_local_importance, generate_nlp_explanation, FEATURE_LABELS

st.set_page_config(page_title="Risk Results | DiaForecaster AI",
                   page_icon="📊", layout="wide")
inject_css()
page_header("📊", "Your Diabetes Risk Results",
            "Ensemble model prediction · SHAP feature analysis · Plain-English explanation")

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')

if 'patient_data' not in st.session_state:
    st.warning("⚠️ No patient data found. Please complete **Patient Input** first.")
    if st.button("Go to Patient Input"):
        st.switch_page("pages/1_Patient_Input.py")
    st.stop()

patient = st.session_state['patient_data']

with st.spinner("🔄 Running ensemble model and computing explanations..."):
    scaled, feature_cols, raw_row = preprocess_patient(patient)
    prob, level                   = predict_risk(scaled)
    model                         = joblib.load(os.path.join(SAVE_DIR, 'ensemble_model.pkl'))
    importances, _                = compute_local_importance(model, scaled, feature_cols)
    explanation                   = generate_nlp_explanation(level, prob, importances, patient)

st.markdown("---")

# ── Risk banner ──
risk_banner(level, prob)

# ── Metric row ──
st.markdown("")
mc1, mc2, mc3, mc4 = st.columns(4)

def _delta_color(condition): return "inverse" if condition else "normal"

mc1.metric("Risk Score",    f"{prob*100:.1f}%")
mc2.metric("Risk Level",    f"{RISK_ICONS[level]} {level}")
mc3.metric("HbA1c",
           f"{patient['HbA1c_level']}%",
           delta="⚠️ High"   if patient['HbA1c_level'] >= 6.5 else "✅ Normal",
           delta_color=_delta_color(patient['HbA1c_level'] >= 6.5))
mc4.metric("Blood Glucose",
           f"{patient['blood_glucose_level']} mg/dL",
           delta="⚠️ High"   if patient['blood_glucose_level'] >= 126 else "✅ Normal",
           delta_color=_delta_color(patient['blood_glucose_level'] >= 126))

st.markdown("---")

# ── Charts: SHAP + Gauge ──
section_title("Why This Prediction?")
st.markdown("")

chart_col, gauge_col = st.columns([1.3, 1], gap="large")

with chart_col:
    feats      = [FEATURE_LABELS.get(f, f) for f in list(importances.keys())[:8]]
    vals       = list(importances.values())[:8]
    bar_colors = ['#ef4444' if v > 0 else '#10b981' for v in vals]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor('white')
    bars = ax.barh(feats[::-1], vals[::-1],
                   color=bar_colors[::-1], edgecolor='white',
                   height=0.58, zorder=2)
    ax.axvline(x=0, color='#94a3b8', linewidth=1.2, linestyle='--', zorder=1)
    ax.set_xlabel("Contribution to Risk", fontsize=10, color='#475569')
    ax.set_title("SHAP-style Feature Importance",
                 fontsize=12, fontweight='700', color='#0f172a', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e2e8f0')
    ax.spines['bottom'].set_color('#e2e8f0')
    ax.tick_params(colors='#64748b', labelsize=9)
    ax.set_facecolor('#fafafa')

    for bar, val in zip(bars[::-1], vals[::-1]):
        x_pos = val + 0.007 if val >= 0 else val - 0.007
        ha    = 'left'       if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'{val:+.3f}', va='center', ha=ha, fontsize=8.5, color='#475569')

    red_p   = mpatches.Patch(color='#ef4444', label='Increases Risk')
    green_p = mpatches.Patch(color='#10b981', label='Reduces Risk')
    ax.legend(handles=[red_p, green_p], loc='lower right',
              fontsize=9, framealpha=0.9, edgecolor='#e2e8f0')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with gauge_col:
    st.markdown("""
    <div style="font-weight:700; font-size:0.88rem; color:#64748b;
         text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.5rem;">
        Risk Gauge
    </div>""", unsafe_allow_html=True)
    gauge_fig = draw_gauge(prob, level, figsize=(5, 3.4))
    st.pyplot(gauge_fig)
    plt.close()

    # Quick health status pills
    st.markdown("")
    def pill(label, val, ok, warn):
        if val <= ok:   c, bg = '#059669', '#d1fae5'
        elif val <= warn: c, bg = '#d97706', '#fef3c7'
        else:             c, bg = '#dc2626', '#fee2e2'
        return f"<span style='background:{bg};color:{c};border-radius:999px;padding:0.2rem 0.7rem;font-size:0.78rem;font-weight:600;margin:2px;display:inline-block;'>{label}</span>"

    bmi_pill  = pill(f"BMI {patient['bmi']:.1f}",       patient['bmi'],                24.9, 29.9)
    glc_pill  = pill(f"Glucose {patient['blood_glucose_level']}",
                     patient['blood_glucose_level'], 99,   125)
    hba_pill  = pill(f"HbA1c {patient['HbA1c_level']}%", patient['HbA1c_level'],       5.6,  6.4)
    age_c, age_bg = ('#059669','#d1fae5') if patient['age']<35 else ('#d97706','#fef3c7') if patient['age']<50 else ('#dc2626','#fee2e2')
    age_pill  = f"<span style='background:{age_bg};color:{age_c};border-radius:999px;padding:0.2rem 0.7rem;font-size:0.78rem;font-weight:600;margin:2px;display:inline-block;'>Age {patient['age']}</span>"

    st.markdown(f"""
    <div style="margin-top:0.5rem;">
        <div style="font-size:0.78rem; color:#94a3b8; margin-bottom:0.4rem; font-weight:600;">
            HEALTH STATUS
        </div>
        {bmi_pill}{glc_pill}{hba_pill}{age_pill}
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── NLP Explanation ──
section_title("Plain-English Explanation")
st.markdown("")
exp_color  = RISK_COLORS[level]
exp_light  = {'Low':'#d1fae5','Medium':'#fef3c7','High':'#fee2e2','Critical':'#ede9fe'}[level]
st.markdown(f"""
<div class="dia-card" style="border-left:5px solid {exp_color}; background:{exp_light};">
    {explanation.replace(chr(10),'<br>').replace('**','<b>').replace('<b>','<b>').replace('</b>','</b>')}
</div>""", unsafe_allow_html=True)

# ── Navigation buttons ──
st.markdown("---")
n1, n2 = st.columns(2)
with n1:
    if st.button("🔄 Try What-If Simulator", use_container_width=True):
        st.switch_page("pages/3_WhatIf_Simulator.py")
with n2:
    if st.button("📈 View 5-Year Trajectory", use_container_width=True):
        st.switch_page("pages/4_5Year_Trajectory.py")
