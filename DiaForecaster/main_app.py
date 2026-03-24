import streamlit as st
import joblib, os, sys

sys.path.append(os.path.dirname(__file__))
from utils.styles import inject_css, page_header, section_title, RISK_COLORS

st.set_page_config(
    page_title="DiaForecaster AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
inject_css()

# ── Header ──
page_header("🏥", "DiaForecaster AI",
            "Explainable AI System for Future Diabetes Risk Prediction")

# ── Live metrics ──
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
try:
    m    = joblib.load(os.path.join(SAVE_DIR, 'metrics.pkl'))
    acc  = f"{m['accuracy']*100:.2f}%"
    auc  = f"{m['auc_roc']:.4f}"
    rec  = f"{m['recall']*100:.2f}%"
    f1   = f"{m['f1']*100:.2f}%"
    spec = f"{m['specificity']*100:.2f}%"
    thr  = joblib.load(os.path.join(SAVE_DIR, 'optimal_threshold.pkl'))
    thr_str = f"{thr:.3f}"
except Exception:
    acc = auc = rec = f1 = spec = thr_str = "N/A"

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("✅ Accuracy",    acc)
c2.metric("📈 AUC-ROC",     auc)
c3.metric("🔴 Recall",      rec,  help="#1 medical metric — never miss a patient")
c4.metric("⚖️ F1 Score",    f1)
c5.metric("🔵 Specificity", spec)

st.markdown("---")

# ── Feature cards ──
section_title("What DiaForecaster AI Does")
st.markdown("<br>", unsafe_allow_html=True)

f1c, f2c, f3c, f4c = st.columns(4)
cards = [
    ("🤖", "Ensemble Model",
     "Random Forest · Gradient Boosting · Logistic Regression<br>Class-weighted for maximum Recall."),
    ("🔍", "Explainable AI",
     "Perturbation-based SHAP analysis shows which health factors drive each individual prediction."),
    ("🔄", "What-If Simulator",
     "Adjust BMI, glucose, HbA1c with sliders and see your risk update in real time."),
    ("📈", "5-Year Trajectory",
     "Year-by-year risk forecast with no-change, diet & exercise, and medication scenarios."),
]
for col, (icon, title, desc) in zip([f1c, f2c, f3c, f4c], cards):
    with col:
        st.markdown(f"""
        <div class="dia-card" style="height:160px;">
            <div style="font-size:1.8rem; margin-bottom:0.5rem;">{icon}</div>
            <div style="font-weight:700; color:#0f172a; margin-bottom:0.4rem; font-size:0.97rem;">
                {title}
            </div>
            <div style="font-size:0.84rem; color:#64748b; line-height:1.5;">
                {desc}
            </div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Navigation ──
section_title("Get Started")
st.markdown("")
n1, n2, n3 = st.columns([1,1,2])
with n1:
    if st.button("📋 Start Risk Assessment", use_container_width=True):
        st.switch_page("pages/1_Patient_Input.py")
with n2:
    if st.button("📉 Model Performance",     use_container_width=True):
        st.switch_page("pages/5_Model_Performance.py")

# ── Pages guide ──
st.markdown("---")
section_title("Navigation Guide")
st.markdown("")
g1,g2,g3,g4,g5 = st.columns(5)
guide = [
    ("1️⃣","Patient Input","Enter health details"),
    ("2️⃣","Risk Results","Risk level + SHAP chart"),
    ("3️⃣","What-If","Simulate changes"),
    ("4️⃣","Trajectory","5-year forecast"),
    ("5️⃣","Performance","Metrics & ROC"),
]
for col,(num,ttl,desc) in zip([g1,g2,g3,g4,g5], guide):
    with col:
        st.markdown(f"""
        <div class="dia-card" style="text-align:center; padding:1rem;">
            <div style="font-size:1.4rem;">{num}</div>
            <div style="font-weight:700; color:#0f172a; font-size:0.88rem;">{ttl}</div>
            <div style="font-size:0.78rem; color:#94a3b8;">{desc}</div>
        </div>""", unsafe_allow_html=True)

# ── Disclaimer ──
st.markdown("---")
st.markdown("""
<div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
     padding:1rem 1.25rem; font-size:0.82rem; color:#64748b;">
⚠️ <b>Medical Disclaimer:</b> DiaForecaster AI is a research and educational tool built for a final year
academic project. It is <b>not</b> a substitute for professional medical advice, diagnosis, or treatment.
The 5-Year Trajectory uses statistical assumptions and is a simulation, not a clinical guarantee.
Always consult a qualified healthcare provider for medical decisions.
</div>""", unsafe_allow_html=True)
