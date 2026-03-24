import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.styles       import inject_css, page_header, section_title, RISK_COLORS, RISK_ICONS, RISK_LIGHT
from risk.stratification import calculate_trajectory, trajectory_with_intervention

st.set_page_config(page_title="5-Year Trajectory | DiaForecaster AI",
                   page_icon="📈", layout="wide")
inject_css()
page_header("📈", "5-Year Diabetes Risk Trajectory",
            "Statistical simulation of risk progression — with and without lifestyle interventions")

if 'patient_data' not in st.session_state:
    st.warning("⚠️ Please complete **Patient Input** first.")
    if st.button("Go to Patient Input"):
        st.switch_page("pages/1_Patient_Input.py")
    st.stop()

patient = st.session_state['patient_data']

with st.spinner("📊 Computing 5-year projections..."):
    base_traj = calculate_trajectory(patient, years=5)
    diet_traj = trajectory_with_intervention(patient, 'diet_exercise', years=5)
    med_traj  = trajectory_with_intervention(patient, 'medication',    years=5)

st.markdown("---")

# ── Main chart ──
section_title("📉 Risk Progression — Three Scenarios")
st.markdown("")

yr_labels  = [t['year']        for t in base_traj]
base_probs = [t['probability'] for t in base_traj]
diet_probs = [t['probability'] for t in diet_traj]
med_probs  = [t['probability'] for t in med_traj]
x_vals     = list(range(6))

fig, ax = plt.subplots(figsize=(12, 5.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('#fafafa')

# Risk zone fills
zone_fills = [
    (0,  25, '#10b981', 'LOW',      0.07),
    (25, 50, '#f59e0b', 'MEDIUM',   0.07),
    (50, 75, '#ef4444', 'HIGH',     0.07),
    (75, 100,'#7c3aed', 'CRITICAL', 0.07),
]
for y0, y1, c, lbl, alpha in zone_fills:
    ax.fill_between([-0.4, 5.4], y0, y1, alpha=alpha, color=c, zorder=1)
    ax.text(-0.45, (y0+y1)/2, lbl, color=c, fontsize=8, fontweight='700',
            ha='right', va='center')

# Lines with gradient markers
line_styles = [
    (base_probs, '#1e40af', 'o-',  3.0, 9, 'No Change (current trend)'),
    (diet_probs, '#10b981', 's--', 2.5, 8, 'With Diet & Exercise'),
    (med_probs,  '#3b82f6', '^--', 2.5, 8, 'With Medication'),
]
for probs, color, style, lw, ms, label in line_styles:
    ax.plot(x_vals, probs, style, color=color, linewidth=lw,
            markersize=ms, label=label, zorder=4,
            markerfacecolor='white', markeredgewidth=2.5)

# Annotate base line values
for i, (y, t) in enumerate(zip(base_probs, base_traj)):
    c = RISK_COLORS[t['risk_level']]
    ax.annotate(f"{y:.0f}%",
                xy=(i, y), xytext=(0, 14), textcoords='offset points',
                ha='center', fontsize=8.5, fontweight='700', color=c,
                arrowprops=dict(arrowstyle='-', color='#cbd5e1', lw=0.8))

# Threshold dashes
for y, c in [(25,'#f59e0b'),(50,'#ef4444'),(75,'#7c3aed')]:
    ax.axhline(y=y, color=c, linestyle=':', linewidth=1.2, alpha=0.5, zorder=2)

ax.set_xticks(x_vals)
ax.set_xticklabels(yr_labels, fontsize=11, fontweight='500', color='#475569')
ax.set_xlim(-0.4, 5.4)
ax.set_ylim(0, 112)
ax.set_xlabel("Time Point", fontsize=11, color='#475569')
ax.set_ylabel("Diabetes Risk Probability (%)", fontsize=11, color='#475569')
ax.set_title("Projected Diabetes Risk Over 5 Years",
             fontsize=14, fontweight='700', color='#0f172a', pad=14)
ax.legend(loc='upper left', fontsize=10, framealpha=0.95,
          edgecolor='#e2e8f0', fancybox=True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#e2e8f0')
ax.spines['bottom'].set_color('#e2e8f0')
ax.tick_params(colors='#64748b')
ax.grid(axis='y', alpha=0.4, linestyle='--', color='#e2e8f0')
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")

# ── Year-by-year tables ──
section_title("📋 Year-by-Year Breakdown")
st.markdown("")
t_col1, t_col2 = st.columns(2)

with t_col1:
    st.markdown("**📌 No Change Scenario**")
    for t in base_traj:
        c = RISK_COLORS[t['risk_level']]
        bg = RISK_LIGHT[t['risk_level']]
        st.markdown(f"""
        <div style='border-left:4px solid {c}; background:{bg};
             padding:0.55rem 1rem; margin:0.25rem 0; border-radius:6px;
             display:flex; justify-content:space-between; align-items:center;'>
            <span style="font-weight:700; color:#0f172a;">{t['year']}</span>
            <span style="color:{c}; font-weight:700;">{RISK_ICONS[t['risk_level']]} {t['risk_level']}</span>
            <span style="font-weight:800; color:#0f172a;">{t['probability']}%</span>
            <span style="font-size:0.78rem; color:#94a3b8;">
                BMI {t['bmi']} · Glu {int(t['glucose'])} · HbA1c {t['hba1c']}%
            </span>
        </div>""", unsafe_allow_html=True)

with t_col2:
    st.markdown("**🥗 Diet & Exercise Scenario**")
    for base_t, d in zip(base_traj, diet_traj):
        c    = RISK_COLORS[d['risk_level']]
        bg   = RISK_LIGHT[d['risk_level']]
        saved = base_t['probability'] - d['probability']
        saved_str = (f'<span style="color:#059669; font-weight:700;">▼ {saved:.1f}% saved</span>'
                     if saved > 0 else '')
        st.markdown(f"""
        <div style='border-left:4px solid {c}; background:{bg};
             padding:0.55rem 1rem; margin:0.25rem 0; border-radius:6px;
             display:flex; justify-content:space-between; align-items:center;'>
            <span style="font-weight:700; color:#0f172a;">{d['year']}</span>
            <span style="color:{c}; font-weight:700;">{RISK_ICONS[d['risk_level']]} {d['risk_level']}</span>
            <span style="font-weight:800; color:#0f172a;">{d['probability']}%</span>
            {saved_str}
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Milestones ──
section_title("⚠️ Key Risk Milestones")
st.markdown("")
milestones = []
for i in range(1, len(base_traj)):
    pl, cl = base_traj[i-1]['risk_level'], base_traj[i]['risk_level']
    if cl != pl:
        milestones.append((base_traj[i]['year'], pl, cl, base_traj[i]['probability']))

if milestones:
    for yr, fl, tl, p in milestones:
        if tl in ('High','Critical'):
            st.error(f"🚨 **{yr}:** Risk escalates {fl} → {tl} ({p}%) — Medical consultation strongly advised")
        else:
            st.warning(f"⚠️ **{yr}:** Risk level changes {fl} → {tl} ({p}%)")
else:
    cl = base_traj[0]['risk_level']
    if cl in ('Low','Medium'):
        st.success(f"✅ Great news! Risk remains at **{cl}** throughout the 5-year period.")
    else:
        st.error(f"🚨 Risk remains at **{cl}** — immediate medical action is recommended.")

st.markdown("---")

# ── Summary metrics ──
section_title("💡 Summary")
st.markdown("")
now_t, yr5_t, diet5 = base_traj[0], base_traj[-1], diet_traj[-1]
reduction = now_t['probability'] - diet5['probability']

sm1, sm2, sm3 = st.columns(3)
sm1.metric("Risk Now",                      f"{now_t['probability']}%")
sm2.metric("Risk Yr 5 — No Change",         f"{yr5_t['probability']}%",
           delta=f"+{yr5_t['probability']-now_t['probability']:.1f}%",
           delta_color="inverse")
sm3.metric("Risk Yr 5 — Diet & Exercise",   f"{diet5['probability']}%",
           delta=f"{'−' if reduction>0 else '+'}{abs(reduction):.1f}% vs now",
           delta_color="inverse" if reduction>0 else "normal")

st.info("📌 These projections assume average annual trends based on population statistics. "
        "Actual outcomes depend on lifestyle choices, medical treatment, and regular monitoring.")

st.markdown("---")
if st.button("📉 View Full Model Performance", use_container_width=True):
    st.switch_page("pages/5_Model_Performance.py")
