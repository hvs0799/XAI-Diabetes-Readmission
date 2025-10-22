import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# -------------------------------
# Load model, scaler, and features
# -------------------------------
model = joblib.load("xgb_model.pkl")        # your trained XGBoost model
scaler = joblib.load("scaler.pkl")          # StandardScaler
feature_names = joblib.load("feature_names.pkl")  # list of feature names

st.set_page_config(page_title="Hospital Readmission Prediction", layout="wide")

st.title("🏥 Diabetes Readmission Prediction Dashboard (XGBoost)")
st.write("Enter patient details below to predict **hospital readmission** within 30 days.")

# -------------------------------
# Hardcoded categorical options (from UCI dataset)
# -------------------------------
categorical_options = {
    "race": ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"],
    "gender": ["Male", "Female", "Unknown/Invalid"],
    "age": ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)",
            "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
    "max_glu_serum": ["None", "Norm", ">200", ">300"],
    "A1Cresult": ["None", "Norm", ">7", ">8"],
    "change": ["Ch", "No"],
    "diabetesMed": ["Yes", "No"],

    # Medication-related categorical columns
    "metformin": ["No", "Steady", "Up", "Down"],
    "repaglinide": ["No", "Steady", "Up", "Down"],
    "nateglinide": ["No", "Steady", "Up", "Down"],
    "chlorpropamide": ["No", "Steady", "Up", "Down"],
    "glimepiride": ["No", "Steady", "Up", "Down"],
    "acetohexamide": ["No", "Steady", "Up", "Down"],
    "glipizide": ["No", "Steady", "Up", "Down"],
    "glyburide": ["No", "Steady", "Up", "Down"],
    "tolbutamide": ["No", "Steady", "Up", "Down"],
    "pioglitazone": ["No", "Steady", "Up", "Down"],
    "rosiglitazone": ["No", "Steady", "Up", "Down"],
    "acarbose": ["No", "Steady", "Up", "Down"],
    "miglitol": ["No", "Steady", "Up", "Down"],
    "troglitazone": ["No", "Steady", "Up", "Down"],
    "tolazamide": ["No", "Steady", "Up", "Down"],
    "examide": ["No", "Steady", "Up", "Down"],
    "citoglipton": ["No", "Steady", "Up", "Down"],
    "insulin": ["No", "Steady", "Up", "Down"],
    "glyburide-metformin": ["No", "Steady", "Up", "Down"],
    "glipizide-metformin": ["No", "Steady", "Up", "Down"],
    "glimepiride-pioglitazone": ["No", "Steady", "Up", "Down"],
    "metformin-rosiglitazone": ["No", "Steady", "Up", "Down"],
    "metformin-pioglitazone": ["No", "Steady", "Up", "Down"],
}

# -------------------------------
# Collect inputs from user
# -------------------------------
st.sidebar.header("🔹 Enter Patient Details")

inputs = {}
for col in feature_names:
    if col in categorical_options:
        inputs[col] = st.sidebar.selectbox(col, categorical_options[col])
    else:
        inputs[col] = st.sidebar.number_input(col, value=0.0)

# -------------------------------
# Preprocess Input
# -------------------------------
# Convert to DataFrame
input_df = pd.DataFrame([inputs])

# Encode categorical variables
for col, options in categorical_options.items():
    if col in input_df:
        input_df[col] = input_df[col].apply(lambda x: options.index(x))

# Reorder columns to match training
input_df = input_df[feature_names]

# Scale numeric features
input_scaled = scaler.transform(input_df)

# -------------------------------
# Prediction
# -------------------------------
pred_prob = model.predict_proba(input_scaled)[0][1]
prediction = model.predict(input_scaled)[0]

st.subheader("Prediction Result")
if prediction == 1:
    st.error(f"⚠️ Patient likely to be readmitted (probability: {pred_prob:.2f})")
else:
    st.success(f"✅ Patient unlikely to be readmitted (probability: {pred_prob:.2f})")

# -------------------------------
# SHAP Explainability
# -------------------------------
st.subheader("🔎 SHAP Explainability")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_scaled)

# Waterfall Plot
fig, ax = plt.subplots(figsize=(5, 3))
shap.waterfall_plot = shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value, shap_values[0], input_df.iloc[0], feature_names=feature_names, show=False
)
plt.title("SHAP Waterfall Plot for Patient Prediction")
st.pyplot(fig)

# Bar Plot - top feature contributions
st.subheader("Top Feature Contributions")
shap_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": shap_values[0]
}).sort_values(by="Importance", key=abs, ascending=False)

st.write(shap_importance.head(10))
