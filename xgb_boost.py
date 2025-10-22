# -------------------------------
# 1. Import Libraries
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE

# -------------------------------
# 2. Load Data
# -------------------------------
data = pd.read_csv("C:\Users\Harsha S\OneDrive\Desktop\diabetes readmission\diabetic_data.xls"")

# -------------------------------
# 3. Cleaning
# -------------------------------
data = data.drop(['weight', 'payer_code', 'medical_specialty'], axis=1)
data = data.replace("?", np.nan)
data = data.drop(['encounter_id', 'patient_nbr'], axis=1)

# -------------------------------
# 4. Target Variable
# -------------------------------
data['readmitted'] = data['readmitted'].replace({'>30': 0, 'NO': 0, '<30': 1})

# -------------------------------
# 5. Encode Categorical Variables
# -------------------------------
categorical_cols = data.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col].astype(str))

# -------------------------------
# 6. Features & Labels
# -------------------------------
X = data.drop('readmitted', axis=1)
y = data['readmitted']

# -------------------------------
# 7. Handle Class Imbalance with SMOTE
# -------------------------------
sm = SMOTE(sampling_strategy='minority', random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# -------------------------------
# 8. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# -------------------------------
# 9. Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 10. Train XGBoost
# -------------------------------
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)

# Save model & scaler
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

# -------------------------------
# 11. Evaluation
# -------------------------------
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"✅ Precision: {precision_score(y_test, y_pred):.4f}")
print(f"✅ Recall: {recall_score(y_test, y_pred):.4f}")
print(f"✅ F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"✅ ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# -------------------------------
# 12. SHAP Explainability
# -------------------------------
print("\n🔎 Running SHAP explainability with XGBoost...")

# Convert test set back to DataFrame for SHAP
X_test_df = pd.DataFrame(X_test, columns=X.columns)

# Take a small sample for faster SHAP (adjust as needed)
X_sample = X_test_df.sample(200, random_state=42)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_sample)

# --- SHAP Summary Plot ---
shap.summary_plot(shap_values, X_sample, show=False)
plt.title("SHAP Summary Plot - XGBoost")
plt.savefig("shap_summary_xgb.png", bbox_inches="tight")
plt.close()

# --- SHAP Waterfall Plot for first prediction ---
shap.plots.waterfall(shap.Explanation(values=shap_values[0], 
                                      base_values=explainer.expected_value, 
                                      data=X_sample.iloc[0,:]), show=False)
plt.savefig("shap_waterfall_xgb.png", bbox_inches="tight")
plt.close()

print("✅ SHAP plots saved: shap_summary_xgb.png & shap_waterfall_xgb.png")
