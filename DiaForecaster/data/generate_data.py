import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 100000

age = np.random.randint(18, 80, n)
gender = np.random.choice(['Male', 'Female', 'Other'], n, p=[0.49, 0.49, 0.02])
bmi = np.clip(np.random.normal(27.5, 6.5, n), 10, 60)
hypertension = np.random.binomial(1, np.clip(0.05 + 0.006*(age-18)/62, 0, 0.5), n)
heart_disease = np.random.binomial(1, np.clip(0.03 + 0.004*(age-18)/62, 0, 0.4), n)
smoking_history = np.random.choice(
    ['never','former','current','not current','No Info','ever'], n,
    p=[0.35, 0.20, 0.15, 0.10, 0.10, 0.10]
)

hba1c_base = 4.5 + (bmi-18)*0.07 + (age-18)*0.025 + hypertension*0.4 + heart_disease*0.3
hba1c = np.clip(hba1c_base + np.random.normal(0, 0.5, n), 3.5, 9.5)

glucose_base = 80 + (bmi-18)*1.5 + (age-18)*0.6 + hypertension*12 + hba1c*10
blood_glucose = np.clip(glucose_base + np.random.normal(0, 15, n), 80, 300).astype(int)

risk_score = (
    (hba1c >= 6.5).astype(float)*4.0 + (hba1c >= 5.7).astype(float)*1.5 +
    (blood_glucose >= 126).astype(float)*3.5 + (blood_glucose >= 100).astype(float)*1.0 +
    (bmi >= 30).astype(float)*2.0 + (bmi >= 25).astype(float)*0.8 +
    (age >= 50).astype(float)*1.5 + (age >= 35).astype(float)*0.6 +
    hypertension*1.2 + heart_disease*0.8
)
prob = 1 / (1 + np.exp(-(risk_score - 9.0)))
diabetes = np.random.binomial(1, prob, n)

df = pd.DataFrame({
    'age': age, 'gender': gender, 'bmi': bmi.round(2),
    'hypertension': hypertension, 'heart_disease': heart_disease,
    'smoking_history': smoking_history, 'HbA1c_level': hba1c.round(1),
    'blood_glucose_level': blood_glucose, 'diabetes': diabetes
})
OUT_PATH = os.path.join(os.path.dirname(__file__), 'diabetes_prediction_dataset.csv')
df.to_csv(OUT_PATH, index=False)
print(f"Dataset: {df.shape}, Positive rate: {df['diabetes'].mean():.2%}")
print(df['diabetes'].value_counts())
