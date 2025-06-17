import streamlit as st
import numpy as np
import pickle  
from sklearn.preprocessing import StandardScaler

# Load trained model
model = pickle.load(open('model/diabetes_model.pkl', 'rb'))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

st.title('Diabetes Prediction App')

# User input
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=23.0)
age = st.number_input("Age", min_value=1, max_value=100, value=30)
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)

# Predict
if st.button("Predict"):
    input_data = np.array([[glucose, bmi, age, pregnancies, dpf]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Prediction: Diabetic")
    else:
        st.success("Prediction: Not Diabetic")
