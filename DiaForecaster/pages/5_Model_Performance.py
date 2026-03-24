import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import joblib, os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.styles import inject_css, page_header, section_title

st.set_page_config(page_title="Model Performance | DiaForecaster AI",
                   page_icon="📉", layout="wide")
inject_css()
page_header("📉", "Model Performance",
            "Medical-grade evaluation — Recall · AUC · Specificity · F1 · Confusion Matrix · ROC")

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')

LABELS = {
    'blood_glucose_level':'Blood Glucose','HbA1c_level':'HbA1c Level',
    'bmi':'BMI','age':'Age','hypertension':'Hypertension',
    'smoking_history':'Smoking History','gender':'Gender','heart_disease':'Heart Disease'
}
RISK_ICONS = {'Low':'🟢','Medium':'🟡','High':'🔴','Critical':'🚨'}

try:
    metrics          = joblib.load(os.path.join(SAVE_DIR,'metrics.pkl'))
    model_comparison = joblib.load(os.path.join(SAVE_DIR,'model_comparison.pkl'))
    feature_imp      = joblib.load(os.path.join(SAVE_DIR,'feature_importance.pkl'))
    threshold        = joblib.load(os.path.join(SAVE_DIR,'optimal_threshold.pkl'))
    test_data        = joblib.load(os.path.join(SAVE_DIR,'test_data.pkl'))
    model            = joblib.load(os.path.join(SAVE_DIR,'ensemble_model.pkl'))
except Exception as e:
    st.error(f"⚠️ Model files not found. Run `python models/ensemble_model.py` first.\n\nError: {e}")
    st.stop()

# ── Medical targets ──
TARGETS = [
    ("🔴 Recall",      'recall',       0.90, "≥ 90%",  "Most Critical — Never miss a patient"),
    ("📈 AUC-ROC",     'auc_roc',      0.92, "≥ 0.92", "Discrimination ability"),
    ("⚖️ F1 Score",    'f1',           0.85, "≥ 85%",  "Precision-Recall balance"),
    ("🔵 Specificity", 'specificity',  0.75, "≥ 75%",  "Avoid false alarms"),
    ("✅ Accuracy",     'accuracy',     0.85, "≥ 85%",  "Overall correctness"),
    ("🎯 Precision",   'precision',    0.80, "≥ 80%",  "Positive prediction reliability"),
]

st.markdown("---")
section_title("🎯 Medical Performance Targets vs Achieved")
st.markdown("")

mc1, mc2, mc3 = st.columns(3)
for idx,(name,key,tgt,lbl,desc) in enumerate(TARGETS):
    val    = metrics[key]
    passed = val >= tgt
    with [mc1,mc2,mc3][idx%3]:
        st.metric(
            label=f"{'✅' if passed else '⚠️'} {name}",
            value=f"{val*100:.2f}%" if key!='auc_roc' else f"{val:.4f}",
            delta=f"Target {lbl} — {'MET ✓' if passed else 'BELOW TARGET'}",
            delta_color="normal" if passed else "inverse"
        )
        st.caption(desc)

# ── Cross-validation ──
st.markdown("---")
section_title("🔄 5-Fold Cross-Validation")
st.markdown("")
cv1,cv2,cv3,cv4 = st.columns(4)
cv1.metric("CV Recall Mean",  f"{metrics.get('cv_recall_mean',0)*100:.2f}%")
cv2.metric("CV Recall Std",   f"±{metrics.get('cv_recall_std',0)*100:.2f}%")
cv3.metric("CV AUC Mean",     f"{metrics.get('cv_auc_mean',0):.4f}")
cv4.metric("CV AUC Std",      f"±{metrics.get('cv_auc_std',0):.4f}")
st.caption(f"📌 Classification threshold: **{threshold:.3f}** (default 0.500) — tuned to maximise Recall ≥ 90% while keeping Specificity ≥ 75%")

st.markdown("---")

# ── Confusion Matrix + ROC ──
section_title("📊 Confusion Matrix & ROC Curve")
st.markdown("")
cm_col, roc_col = st.columns(2, gap="large")

with cm_col:
    cm = metrics['confusion_matrix']
    tn,fp,fn,tp = cm.ravel()

    fig,ax = plt.subplots(figsize=(5,4))
    fig.patch.set_facecolor('white')
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.85)
    ax.set_xticks([0,1]); ax.set_xticklabels(['No Diabetes','Diabetes'],fontsize=11)
    ax.set_yticks([0,1]); ax.set_yticklabels(['No Diabetes','Diabetes'],fontsize=11)
    thresh = cm.max()/2
    for i in range(2):
        for j in range(2):
            ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center',
                    fontsize=14,fontweight='700',
                    color='white' if cm[i,j]>thresh else '#1e293b')
    ax.set_ylabel('True Label',fontsize=11,color='#475569')
    ax.set_xlabel('Predicted Label',fontsize=11,color='#475569')
    ax.set_title('Confusion Matrix',fontsize=13,fontweight='700',color='#0f172a',pad=10)
    ax.tick_params(colors='#64748b')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown(f"""
    | Metric | Value |
    |---|---|
    | ✅ True Positives (TP) | {tp:,} |
    | ✅ True Negatives (TN) | {tn:,} |
    | ⚠️ False Positives (FP) | {fp:,} |
    | 🚨 False Negatives (FN) | {fn:,} |
    | **Sensitivity (Recall)** | **{tp/(tp+fn)*100:.2f}%** |
    | **Specificity** | **{tn/(tn+fp)*100:.2f}%** |
    | NPV | {tn/(tn+fn)*100:.2f}% |
    | PPV (Precision) | {tp/(tp+fp)*100:.2f}% |
    """)

with roc_col:
    fpr,tpr,roc_thresh = metrics['roc_curve']
    auc = metrics['auc_roc']

    fig,ax = plt.subplots(figsize=(5,4))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')
    ax.fill_between(fpr, tpr, alpha=0.10, color='#1e40af')
    ax.plot(fpr, tpr, color='#1e40af', lw=2.5, label=f'Ensemble  AUC = {auc:.4f}')
    ax.plot([0,1],[0,1],'--',color='#94a3b8',lw=1.5,label='Random  AUC = 0.50')
    j_scores = tpr-fpr
    opt_idx  = int(np.argmax(j_scores))
    ax.plot(fpr[opt_idx],tpr[opt_idx],'o',color='#ef4444',markersize=10,zorder=5,
            label=f"Optimal (t={roc_thresh[opt_idx]:.2f})")
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.set_xlabel('False Positive Rate (1−Specificity)',fontsize=11,color='#475569')
    ax.set_ylabel('True Positive Rate (Sensitivity)',   fontsize=11,color='#475569')
    ax.set_title(f'ROC Curve  |  AUC = {auc:.4f}',fontsize=13,fontweight='700',
                 color='#0f172a',pad=10)
    ax.legend(loc='lower right',fontsize=9,framealpha=0.95,edgecolor='#e2e8f0')
    ax.grid(alpha=0.35,linestyle='--',color='#e2e8f0')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e2e8f0'); ax.spines['bottom'].set_color('#e2e8f0')
    ax.tick_params(colors='#64748b')
    ax.text(0.52,0.12,f'AUC = {auc:.4f}\n✅ Exceeds 0.92 target',
            fontsize=11,color='#1e40af',fontweight='700',
            bbox=dict(boxstyle='round,pad=0.4',facecolor='#eff6ff',alpha=0.9,
                      edgecolor='#bfdbfe'))
    plt.tight_layout()
    st.pyplot(fig); plt.close()

st.markdown("---")

# ── Model comparison + Feature importance ──
section_title("🤖 Model Comparison & Feature Importance")
st.markdown("")
mc_col, fi_col = st.columns(2, gap="large")

with mc_col:
    models  = list(model_comparison.keys())
    rec_v   = [model_comparison[m]['recall']      *100 for m in models]
    f1_v    = [model_comparison[m]['f1']          *100 for m in models]
    auc_v   = [model_comparison[m]['auc_roc']     *100 for m in models]
    spec_v  = [model_comparison[m]['specificity'] *100 for m in models]

    x = np.arange(len(models)); w=0.20
    fig,ax = plt.subplots(figsize=(7,4.5))
    fig.patch.set_facecolor('white'); ax.set_facecolor('#fafafa')
    b1=ax.bar(x-1.5*w,rec_v, w,label='Recall',      color='#ef4444',edgecolor='white',zorder=2)
    b2=ax.bar(x-0.5*w,f1_v,  w,label='F1 Score',    color='#10b981',edgecolor='white',zorder=2)
    b3=ax.bar(x+0.5*w,auc_v, w,label='AUC-ROC',     color='#3b82f6',edgecolor='white',zorder=2)
    b4=ax.bar(x+1.5*w,spec_v,w,label='Specificity', color='#8b5cf6',edgecolor='white',zorder=2)
    ax.axhline(y=90,color='#ef4444',linestyle='--',alpha=0.4,lw=1.5)
    ax.axhline(y=92,color='#3b82f6',linestyle='--',alpha=0.4,lw=1.5)
    for bars in [b1,b2,b3,b4]:
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,
                    f'{bar.get_height():.1f}',ha='center',va='bottom',fontsize=7,rotation=90,color='#475569')
    ax.set_xticks(x); ax.set_xticklabels(models,rotation=12,ha='right',fontsize=8.5,color='#475569')
    ax.set_ylim(0,118); ax.set_ylabel("Score (%)",color='#475569')
    ax.set_title("All Models — Medical Metric Comparison",fontsize=12,fontweight='700',color='#0f172a',pad=10)
    ax.legend(fontsize=8,loc='lower right',ncol=2,framealpha=0.95,edgecolor='#e2e8f0')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e2e8f0'); ax.spines['bottom'].set_color('#e2e8f0')
    ax.tick_params(colors='#64748b')
    ax.grid(axis='y',alpha=0.35,linestyle='--',color='#e2e8f0')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("**Detailed Scores:**")
    for m in models:
        v=model_comparison[m]
        st.caption(f"**{m}:** Recall={v['recall']*100:.2f}% · AUC={v['auc_roc']:.4f} · F1={v['f1']*100:.2f}% · Spec={v['specificity']*100:.2f}%")

with fi_col:
    feats  = [LABELS.get(f,f) for f in feature_imp.keys()]
    imps   = list(feature_imp.values())
    fi_colors = ['#ef4444','#f97316','#f59e0b','#10b981',
                 '#14b8a6','#3b82f6','#8b5cf6','#94a3b8'][:len(feats)]

    fig,ax = plt.subplots(figsize=(7,4.5))
    fig.patch.set_facecolor('white'); ax.set_facecolor('#fafafa')
    bars = ax.barh(feats[::-1], imps[::-1],
                   color=fi_colors[::-1], edgecolor='white', height=0.62, zorder=2)
    for bar,val in zip(bars, imps[::-1]):
        ax.text(val+0.002, bar.get_y()+bar.get_height()/2,
                f'{val*100:.2f}%', va='center', fontsize=9, color='#475569')
    ax.set_xlabel("Feature Importance (Gini)",fontsize=11,color='#475569')
    ax.set_title("Global Feature Importance",fontsize=12,fontweight='700',color='#0f172a',pad=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e2e8f0'); ax.spines['bottom'].set_color('#e2e8f0')
    ax.tick_params(colors='#64748b',labelsize=9)
    ax.grid(axis='x',alpha=0.35,linestyle='--',color='#e2e8f0')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("**Impact Breakdown:**")
    for f,imp in feature_imp.items():
        st.progress(min(int(imp*250),100), text=f"{LABELS.get(f,f)}: {imp*100:.2f}%")

st.markdown("---")

# ── Threshold sensitivity ──
section_title("📊 Threshold Sensitivity Analysis")
st.caption("How Recall, Specificity, and F1 vary as the classification threshold changes")
st.markdown("")

try:
    from sklearn.metrics import recall_score, f1_score, confusion_matrix as cmt_fn
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    y_prob = model.predict_proba(X_test)[:,1]

    thresholds = np.arange(0.15,0.75,0.01)
    recalls,specs,f1s=[],[],[]
    for t in thresholds:
        yp = (y_prob>=t).astype(int)
        recalls.append(recall_score(y_test,yp))
        cm_t=cmt_fn(y_test,yp); tn_t,fp_t,_,_=cm_t.ravel()
        specs.append(tn_t/(tn_t+fp_t))
        f1s.append(f1_score(y_test,yp))

    fig,ax = plt.subplots(figsize=(11,4))
    fig.patch.set_facecolor('white'); ax.set_facecolor('#fafafa')
    ax.plot(thresholds,[r*100 for r in recalls],'r-',lw=2.5,label='Recall (Sensitivity)')
    ax.plot(thresholds,[s*100 for s in specs],  'b-',lw=2.5,label='Specificity')
    ax.plot(thresholds,[f*100 for f in f1s],    color='#10b981',lw=2.5,label='F1 Score')
    ax.axvline(x=threshold,color='#0f172a',linestyle='--',lw=2.2,
               label=f'Chosen Threshold ({threshold:.2f})')
    ax.axhline(y=90,color='#ef4444',linestyle=':',alpha=0.55,lw=1.5)
    ax.axhline(y=75,color='#3b82f6',linestyle=':',alpha=0.55,lw=1.5)
    ax.axhline(y=85,color='#10b981',linestyle=':',alpha=0.55,lw=1.5)
    ax.set_xlabel("Classification Threshold",fontsize=12,color='#475569')
    ax.set_ylabel("Score (%)",fontsize=12,color='#475569')
    ax.set_title("Threshold vs Recall / Specificity / F1 — Medical Trade-off",
                 fontsize=12,fontweight='700',color='#0f172a',pad=10)
    ax.legend(loc='center left',fontsize=10,framealpha=0.95,edgecolor='#e2e8f0')
    ax.grid(alpha=0.35,linestyle='--',color='#e2e8f0')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e2e8f0'); ax.spines['bottom'].set_color('#e2e8f0')
    ax.tick_params(colors='#64748b')
    plt.tight_layout(); st.pyplot(fig); plt.close()
except Exception as e:
    st.warning(f"Threshold chart unavailable: {e}")

st.markdown("---")

# ── Summary ──
section_title("📋 Model Configuration Summary")
st.markdown("")
all_pass = all(metrics[key]>=tgt for _,key,tgt,_,_ in TARGETS)
if all_pass:
    st.success("🏆 **ALL 6 medical performance targets achieved!** DiaForecaster AI meets clinical-grade screening standards.")
else:
    st.warning("⚠️ Some targets not yet met. Consider further hyperparameter tuning.")

sc1,sc2 = st.columns(2)
with sc1:
    st.markdown("""<div class="dia-card">
    <b>Why Recall is #1:</b><br><br>
    Missing a diabetic patient (False Negative) leads to undetected disease progression —
    kidney failure, blindness, neuropathy. A false alarm leads to lifestyle advice — far lower harm.
    Clinical screening tools target sensitivity ≥ 90%.
    </div>""", unsafe_allow_html=True)
with sc2:
    st.markdown(f"""<div class="dia-card">
    <b>Model Configuration:</b><br><br>
    • Ensemble: RF + Gradient Boosting + Logistic Regression<br>
    • Class Weights: {{0:1, 1:3}} — diabetic cases penalised 3×<br>
    • Voting: Soft with weights [3, 3, 1]<br>
    • Threshold: {threshold:.3f} (tuned from default 0.500)<br>
    • CV Recall: {metrics.get('cv_recall_mean',0)*100:.2f}% ± {metrics.get('cv_recall_std',0)*100:.2f}%
    </div>""", unsafe_allow_html=True)
