import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import joblib
import os

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')

GENDER_MAP  = {'Male': 0, 'Female': 1, 'Other': 2}
SMOKING_MAP = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5}

def load_and_clean(filepath):
    """Load CSV and encode categorical columns."""
    df = pd.read_csv(filepath)
    df['gender']          = df['gender'].map(GENDER_MAP).fillna(2)
    df['smoking_history'] = df['smoking_history'].map(SMOKING_MAP).fillna(1)
    df.dropna(inplace=True)
    return df

def balance_classes(X, y):
    """Manual oversample minority class (no imblearn required)."""
    df_combined          = pd.DataFrame(X).copy()
    df_combined['target'] = y.values if hasattr(y, 'values') else y

    majority = df_combined[df_combined['target'] == 0]
    minority = df_combined[df_combined['target'] == 1]

    if len(majority) >= len(minority):
        minority_up = resample(minority, replace=True,
                               n_samples=len(majority), random_state=42)
        balanced = pd.concat([majority, minority_up])
    else:
        majority_up = resample(majority, replace=True,
                               n_samples=len(minority), random_state=42)
        balanced = pd.concat([minority, majority_up])

    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced.drop('target', axis=1), balanced['target']

def preprocess(filepath):
    """Full pipeline: clean → scale → balance → split."""
    df = load_and_clean(filepath)

    feature_cols = ['age', 'gender', 'bmi', 'hypertension',
                    'heart_disease', 'smoking_history',
                    'HbA1c_level', 'blood_glucose_level']

    X = df[feature_cols]
    y = df['diabetes']

    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    X_bal, y_bal = balance_classes(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    joblib.dump(scaler,       os.path.join(SAVE_DIR, 'scaler.pkl'))
    joblib.dump(feature_cols, os.path.join(SAVE_DIR, 'feature_cols.pkl'))

    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    print(f"Class balance — Train: {y_train.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test, scaler, feature_cols

if __name__ == '__main__':
    import sys
    _default = os.path.join(os.path.dirname(__file__), '..', 'data', 'diabetes_prediction_dataset.csv')
    fp = sys.argv[1] if len(sys.argv) > 1 else _default
    preprocess(fp)
    print("Preprocessing complete.")
