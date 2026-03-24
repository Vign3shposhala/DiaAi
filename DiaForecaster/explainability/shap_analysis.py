import numpy as np
import pandas as pd
import joblib
import os

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')

FEATURE_LABELS = {
    'blood_glucose_level': 'Blood Glucose Level',
    'HbA1c_level':         'HbA1c Level',
    'bmi':                 'BMI',
    'age':                 'Age',
    'hypertension':        'Hypertension',
    'smoking_history':     'Smoking History',
    'gender':              'Gender',
    'heart_disease':       'Heart Disease'
}

def compute_local_importance(model, patient_input, feature_cols):
    """
    FIX: Accepts both DataFrame and numpy array.
    Converts to numpy internally to avoid DataFrame [0,i] indexing bug.
    Perturbation-based local feature importance (SHAP approximation).
    """
    # ── Always work with numpy array internally ──
    if isinstance(patient_input, pd.DataFrame):
        patient_array = patient_input.values.copy()
    else:
        patient_array = np.array(patient_input).copy()

    if patient_array.ndim == 1:
        patient_array = patient_array.reshape(1, -1)

    # Wrap in DataFrame for sklearn (avoids feature name warnings)
    def _predict(arr):
        return model.predict_proba(pd.DataFrame(arr, columns=feature_cols))[0][1]

    base_prob = _predict(patient_array)
    importances = {}

    for i, feat in enumerate(feature_cols):
        # Perturb up
        arr_up = patient_array.copy()
        arr_up[0, i] += 1.5
        prob_up = _predict(arr_up)

        # Perturb down
        arr_dn = patient_array.copy()
        arr_dn[0, i] -= 1.5
        prob_dn = _predict(arr_dn)

        sensitivity = abs(prob_up - base_prob) + abs(prob_dn - base_prob)
        direction   = 1 if prob_up > base_prob else -1
        importances[feat] = direction * sensitivity / 2

    # Normalise
    total = sum(abs(v) for v in importances.values())
    if total > 0:
        importances = {k: v / total for k, v in importances.items()}

    # Sort by absolute contribution
    importances = dict(sorted(importances.items(),
                               key=lambda x: abs(x[1]), reverse=True))
    return importances, base_prob


def generate_nlp_explanation(risk_level, prob, importances, original_data):
    """Convert SHAP-style importances to plain English medical explanation."""

    top_features = list(importances.items())[:3]
    factor_descriptions = []

    for feat, importance in top_features:
        val = original_data.get(feat, 'N/A')

        if feat == 'blood_glucose_level':
            if val > 140:
                factor_descriptions.append(f"elevated blood glucose ({val} mg/dL)")
            elif val > 100:
                factor_descriptions.append(f"borderline blood glucose ({val} mg/dL)")
            else:
                factor_descriptions.append(f"normal blood glucose ({val} mg/dL)")

        elif feat == 'HbA1c_level':
            if val >= 6.5:
                factor_descriptions.append(f"high HbA1c level ({val}%) — a key diabetes marker")
            elif val >= 5.7:
                factor_descriptions.append(f"pre-diabetic HbA1c range ({val}%)")
            else:
                factor_descriptions.append(f"normal HbA1c ({val}%)")

        elif feat == 'bmi':
            if val >= 30:
                factor_descriptions.append(f"obese BMI ({val:.1f}) increasing metabolic risk")
            elif val >= 25:
                factor_descriptions.append(f"overweight BMI ({val:.1f})")
            else:
                factor_descriptions.append(f"healthy BMI ({val:.1f})")

        elif feat == 'age':
            if val >= 60:
                factor_descriptions.append(f"advanced age ({val} years) — a non-modifiable risk factor")
            elif val >= 45:
                factor_descriptions.append(f"middle age ({val} years) with increasing risk")
            else:
                factor_descriptions.append(f"younger age ({val} years)")

        elif feat == 'hypertension':
            factor_descriptions.append(
                "presence of hypertension (comorbidity)" if val == 1
                else "no hypertension (protective factor)"
            )

        elif feat == 'heart_disease':
            factor_descriptions.append(
                "existing heart disease (significant comorbidity)" if val == 1
                else "no heart disease"
            )

        elif feat == 'smoking_history':
            smoking_labels = {
                0: 'never smoked', 1: 'no info', 2: 'current smoker',
                3: 'former smoker', 4: 'ever smoked', 5: 'not currently smoking'
            }
            factor_descriptions.append(smoking_labels.get(int(val), 'smoking history unknown'))

        elif feat == 'gender':
            gender_labels = {0: 'male', 1: 'female', 2: 'other'}
            factor_descriptions.append(f"gender: {gender_labels.get(int(val), 'unknown')}")

    factors_text = ", ".join(factor_descriptions) if factor_descriptions else "multiple health indicators"

    risk_messages = {
        "Low": (
            "Your current health profile suggests a low probability of developing diabetes.",
            "Continue your healthy habits. Schedule a routine check-up annually."
        ),
        "Medium": (
            "Your health profile indicates moderate risk. Early intervention can significantly reduce future risk.",
            "Consider lifestyle improvements: increase physical activity, reduce sugar intake, and schedule a medical check-up within 6 months."
        ),
        "High": (
            "Your health profile reveals several risk factors that require attention to prevent diabetes onset.",
            "It is strongly recommended to consult a physician soon. Dietary changes, regular exercise, and blood glucose monitoring are advised."
        ),
        "Critical": (
            "Your profile indicates critical risk. Immediate medical intervention can help prevent or delay diabetes onset.",
            "Please seek medical consultation immediately. Comprehensive metabolic testing, dietary overhaul, and medical management are urgently recommended."
        ),
    }

    urgency, action = risk_messages.get(risk_level, risk_messages["High"])

    return (
        f"**Risk Assessment: {risk_level} ({prob*100:.1f}% probability)**\n\n"
        f"{urgency}\n\n"
        f"**Primary contributing factors:** {factors_text}.\n\n"
        f"**Recommendation:** {action}"
    )


if __name__ == '__main__':
    print("Explainability module ready.")
