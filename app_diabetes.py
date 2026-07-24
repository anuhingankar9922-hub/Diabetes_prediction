import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

st.title("🩺 Diabetes Prediction System")
st.write("Enter the patient's details below to predict whether they have diabetes.")

pregnancies = st.number_input("Pregnancies", min_value=0, value=1)

glucose = st.number_input("Glucose", min_value=0.0, value=120.0)

blood_pressure = st.number_input("Blood Pressure", min_value=0.0, value=70.0)

skin_thickness = st.number_input("Skin Thickness", min_value=0.0, value=20.0)

insulin = st.number_input("Insulin", min_value=0.0, value=80.0)

bmi = st.number_input("BMI", min_value=0.0, value=25.0)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    value=0.5,
    format="%.3f"
)

age = st.number_input("Age", min_value=1, value=30)

if st.button("Predict"):

    input_data = pd.DataFrame([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]], columns=columns)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("The person is likely to have Diabetes.")
    else:
        st.success("The person is NOT likely to have Diabetes.")

    st.write(f"Probability of Diabetes: **{probability[0][1]*100:.2f}%**")