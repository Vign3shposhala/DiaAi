"""
Shared CSS theme and UI helpers for DiaForecaster AI.
Medical-grade clinical dashboard aesthetic — clean, trustworthy, precise.
"""

RISK_COLORS  = {'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#ef4444', 'Critical': '#7c3aed'}
RISK_LIGHT   = {'Low': '#d1fae5', 'Medium': '#fef3c7', 'High': '#fee2e2', 'Critical': '#ede9fe'}
RISK_ICONS   = {'Low': '🟢',      'Medium': '🟡',      'High': '🔴',      'Critical': '🚨'}
RISK_BORDER  = {'Low': '#059669', 'Medium': '#d97706', 'High': '#dc2626', 'Critical': '#6d28d9'}

BRAND_NAVY   = '#0f172a'
BRAND_BLUE   = '#1e40af'
BRAND_ACCENT = '#3b82f6'

GLOBAL_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Root overrides ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}
code, .stCode {
    font-family: 'DM Mono', monospace !important;
}

/* ── App background ── */
.stApp {
    background: #f8fafc;
}
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebarNav"] a {
    border-radius: 8px !important;
    padding: 0.5rem 0.75rem !important;
    margin: 2px 0 !important;
    transition: background 0.2s;
}
[data-testid="stSidebarNav"] a:hover {
    background: #1e293b !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.25rem !important;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s;
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.10);
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1e40af, #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.01em;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(30,64,175,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(30,64,175,0.35) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #f1f5f9;
    padding: 4px;
    border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 500 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: white !important;
    border-radius: 10px !important;
    border: 1px solid #e2e8f0 !important;
    font-weight: 600 !important;
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 1.5rem 0;
}

/* ── Selectbox / Number Input ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    border-radius: 8px !important;
    border-color: #cbd5e1 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Checkbox ── */
.stCheckbox label {
    font-weight: 500 !important;
}

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] {
    padding-top: 0.5rem;
}

/* ── Info / Warning / Error / Success ── */
.stInfo, .stWarning, .stError, .stSuccess {
    border-radius: 10px !important;
    border-left-width: 4px !important;
}

/* ── Progress bars ── */
.stProgress > div > div {
    border-radius: 999px !important;
}

/* ── Custom cards ── */
.dia-card {
    background: white;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}
.dia-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #1e40af 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.dia-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.dia-header::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 30%;
    width: 260px; height: 260px;
    background: rgba(59,130,246,0.12);
    border-radius: 50%;
}
.risk-banner {
    padding: 1.5rem 2rem;
    border-radius: 14px;
    border-left-width: 6px;
    border-left-style: solid;
    margin: 1rem 0;
}
.feature-pill {
    display: inline-block;
    background: #eff6ff;
    color: #1e40af;
    border-radius: 999px;
    padding: 0.25rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.2rem;
    border: 1px solid #bfdbfe;
}
</style>
"""

def inject_css():
    """Call this at the top of every page."""
    import streamlit as st
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

def page_header(icon: str, title: str, subtitle: str = ""):
    """Render a consistent branded page header."""
    import streamlit as st
    sub_html = f"<p style='margin:0.4rem 0 0; opacity:0.80; font-size:0.97rem;'>{subtitle}</p>" if subtitle else ""
    st.markdown(f"""
    <div class="dia-header">
        <h1 style='margin:0; font-size:2rem; font-weight:700; letter-spacing:-0.02em;'>
            {icon} {title}
        </h1>
        {sub_html}
    </div>""", unsafe_allow_html=True)

def risk_banner(level: str, prob: float):
    """Render a coloured risk-level banner."""
    import streamlit as st
    color  = RISK_COLORS[level]
    light  = RISK_LIGHT[level]
    border = RISK_BORDER[level]
    icon   = RISK_ICONS[level]
    st.markdown(f"""
    <div class="risk-banner" style="background:{light}; border-color:{border};">
        <div style='display:flex; align-items:center; gap:1rem; flex-wrap:wrap;'>
            <span style='font-size:2.5rem;'>{icon}</span>
            <div>
                <div style='font-size:1.6rem; font-weight:700; color:{color};'>
                    {level.upper()} RISK
                </div>
                <div style='font-size:1rem; color:#475569; margin-top:2px;'>
                    Probability: <b style='color:{color};'>{prob*100:.1f}%</b>
                    &nbsp;·&nbsp; Based on ensemble model analysis
                </div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

def section_title(text: str):
    import streamlit as st
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:0.5rem; margin:1.2rem 0 0.6rem;'>
        <div style='width:4px; height:1.3rem; background:#3b82f6; border-radius:2px;'></div>
        <span style='font-size:1.05rem; font-weight:700; color:#0f172a;'>{text}</span>
    </div>""", unsafe_allow_html=True)
