import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('titanic_model.pkl')

import os
st.write("Current working directory:", os.getcwd())


st.title("Titanic Survival Prediction App")
st.write("This app predicts whether a Titanic passenger would survive based on their details.")

# User inputs
pclass = st.selectbox("Pclass (1 = 1st, 2 = 2nd, 3 = 3rd):", [1, 2, 3])
sex = st.selectbox("Sex (0 = Female, 1 = Male):", [0, 1])
age = st.slider("Age:", 0, 100, 25)
sibsp = st.number_input("Siblings/Spouses Aboard:", min_value=0, max_value=10)
parch = st.number_input("Parents/Children Aboard:", min_value=0, max_value=10)
fare = st.number_input("Fare:", min_value=0.0, value=50.0)
embarked = st.selectbox("Embarked (0 = C, 1 = Q, 2 = S):", [0, 1, 2])

# Prepare input data for prediction
input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("This passenger would have survived! 🎉")
    else:
        st.error("Unfortunately, this passenger would not have survived.")
