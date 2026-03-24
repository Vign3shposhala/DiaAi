import numpy as np
import pandas as pd
import joblib
import os

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')

# ── Module-level model cache (FIX: load once, not on every call) ──
_MODEL_CACHE  = {}

def _get_model():
    if 'ensemble' not in _MODEL_CACHE:
        _MODEL_CACHE['ensemble'] = joblib.load(os.path.join(SAVE_DIR, 'ensemble_model.pkl'))
    return _MODEL_CACHE['ensemble']

def _get_scaler():
    if 'scaler' not in _MODEL_CACHE:
        _MODEL_CACHE['scaler'] = joblib.load(os.path.join(SAVE_DIR, 'scaler.pkl'))
    return _MODEL_CACHE['scaler']

def _get_feature_cols():
    if 'feature_cols' not in _MODEL_CACHE:
        _MODEL_CACHE['feature_cols'] = joblib.load(os.path.join(SAVE_DIR, 'feature_cols.pkl'))
    return _MODEL_CACHE['feature_cols']

# ── Risk level mapping ──
RISK_THRESHOLDS = {
    'Low':      (0.00, 0.25),
    'Medium':   (0.25, 0.50),
    'High':     (0.50, 0.75),
    'Critical': (0.75, 1.01),
}

RISK_COLORS = {
    'Low':      '#2ecc71',
    'Medium':   '#f39c12',
    'High':     '#e74c3c',
    'Critical': '#8e44ad',
}

RISK_ICONS = {
    'Low':      '🟢',
    'Medium':   '🟡',
    'High':     '🔴',
    'Critical': '🚨',
}

def assign_risk_level(probability):
    for level, (low, high) in RISK_THRESHOLDS.items():
        if low <= probability < high:
            return level
    return 'Critical'

# ── Encoding maps ──
GENDER_MAP  = {'Male': 0, 'Female': 1, 'Other': 2}
SMOKING_MAP = {
    'never': 0, 'No Info': 1, 'current': 2,
    'former': 3, 'ever': 4, 'not current': 5
}

def preprocess_patient(patient_dict):
    """
    Convert raw patient dict → scaled DataFrame ready for model.
    Returns (scaled_df, feature_cols, raw_encoded_row).
    """
    scaler       = _get_scaler()
    feature_cols = _get_feature_cols()

    row = {
        'age':                 float(patient_dict['age']),
        'gender':              float(GENDER_MAP.get(patient_dict['gender'], 0)),
        'bmi':                 float(patient_dict['bmi']),
        'hypertension':        float(int(patient_dict['hypertension'])),
        'heart_disease':       float(int(patient_dict['heart_disease'])),
        'smoking_history':     float(SMOKING_MAP.get(patient_dict['smoking_history'], 0)),
        'HbA1c_level':         float(patient_dict['HbA1c_level']),
        'blood_glucose_level': float(patient_dict['blood_glucose_level']),
    }

    df         = pd.DataFrame([row])[feature_cols]
    scaled_arr = scaler.transform(df)
    scaled_df  = pd.DataFrame(scaled_arr, columns=feature_cols)
    return scaled_df, feature_cols, row


def predict_risk(scaled_df):
    """
    Accepts a scaled DataFrame (or numpy array).
    Returns (probability, risk_level).
    """
    model = _get_model()
    if isinstance(scaled_df, pd.DataFrame):
        prob = model.predict_proba(scaled_df)[0][1]
    else:
        prob = model.predict_proba(
            pd.DataFrame(scaled_df, columns=_get_feature_cols())
        )[0][1]
    return float(prob), assign_risk_level(float(prob))


# ── What-If Simulator ──
def whatif_predict(original_patient_dict, modifications):
    """Re-predict after applying modifications to patient dict."""
    modified = original_patient_dict.copy()
    modified.update(modifications)
    scaled, _, _ = preprocess_patient(modified)
    return predict_risk(scaled)


def get_whatif_insights(original_prob, new_prob, original_level, new_level):
    delta = new_prob - original_prob
    if abs(delta) < 0.01:
        return "These changes have minimal impact on your risk level."
    elif delta < -0.10:
        return f"Excellent! These lifestyle changes would significantly reduce your risk from {original_level} → {new_level}."
    elif delta < 0:
        return "Good progress. These changes would moderately reduce your risk."
    elif delta > 0.10:
        return f"Warning: These changes would significantly increase your risk from {original_level} → {new_level}."
    else:
        return "These changes would slightly increase your risk."


# ── 5-Year Trajectory ──
ANNUAL_TRENDS = {
    'age':                 1.0,
    'bmi':                 0.15,
    'blood_glucose_level': 1.5,
    'HbA1c_level':         0.05,
}

def calculate_trajectory(patient_dict, years=5):
    """Simulate risk year-by-year with no lifestyle change."""
    trajectory = []
    current    = patient_dict.copy()

    for year in range(years + 1):
        scaled, _, _ = preprocess_patient(current)
        prob, level  = predict_risk(scaled)

        trajectory.append({
            'year':       'Now' if year == 0 else f'Year {year}',
            'year_num':   year,
            'probability': round(prob * 100, 1),
            'risk_level': level,
            'bmi':        round(float(current['bmi']), 1),
            'glucose':    round(float(current['blood_glucose_level']), 0),
            'hba1c':      round(float(current['HbA1c_level']), 1),
            'age':        int(current['age']),
        })

        if year < years:
            for feat, delta in ANNUAL_TRENDS.items():
                if feat in current:
                    current[feat] = float(current[feat]) + delta

    return trajectory


def trajectory_with_intervention(patient_dict, intervention, years=5):
    """Simulate trajectory after a lifestyle or medication intervention."""
    current = patient_dict.copy()

    if intervention == 'diet_exercise':
        current['bmi']                 = max(18.5, float(current['bmi']) - 2.0)
        current['blood_glucose_level'] = max(80,   float(current['blood_glucose_level']) - 15)
        current['HbA1c_level']         = max(4.0,  float(current['HbA1c_level']) - 0.3)
        annual_bmi_chg     = -0.10
        annual_glucose_chg = -0.50
        annual_hba1c_chg   =  0.02
    elif intervention == 'medication':
        current['blood_glucose_level'] = max(80,  float(current['blood_glucose_level']) - 25)
        current['HbA1c_level']         = max(4.0, float(current['HbA1c_level']) - 0.5)
        annual_bmi_chg     =  0.10
        annual_glucose_chg =  0.50
        annual_hba1c_chg   =  0.02
    else:
        annual_bmi_chg     =  0.15
        annual_glucose_chg =  1.50
        annual_hba1c_chg   =  0.05

    trajectory = []
    for year in range(years + 1):
        scaled, _, _ = preprocess_patient(current)
        prob, level  = predict_risk(scaled)
        trajectory.append({
            'year':        'Now' if year == 0 else f'Year {year}',
            'year_num':    year,
            'probability': round(prob * 100, 1),
            'risk_level':  level,
        })
        if year < years:
            current['age']                 = float(current['age']) + 1
            current['bmi']                 = max(18.5, float(current['bmi']) + annual_bmi_chg)
            current['blood_glucose_level'] = max(80,   float(current['blood_glucose_level']) + annual_glucose_chg)
            current['HbA1c_level']         = max(4.0,  float(current['HbA1c_level']) + annual_hba1c_chg)

    return trajectory


if __name__ == '__main__':
    print("Risk stratification module ready.")
