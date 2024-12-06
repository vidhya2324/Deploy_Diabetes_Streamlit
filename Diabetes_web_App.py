
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 18:38:21 2024

@author: Vidhya
"""

import numpy as np
import pickle
import streamlit as st

# Loading the saved model
try:
    loaded_model = pickle.load(open('C:/Users/dell/Desktop/Deploy_diabetes/trained_model.sav', 'rb'))
except FileNotFoundError:
    st.error("Model file not found. Please check the file path.")

# Function for diabetes prediction
def diabetes_prediction(input_data):
    # Convert input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Prediction
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

# Main function
def main():
    # Title
    st.title('Diabetes Prediction Web App')

    # Input from the user
    gender = st.selectbox('Gender', ['0: Female', '1: Male'])  # Assuming binary values
    age = st.number_input('Enter your age', min_value=0, max_value=120, step=1)
    hypertension = st.selectbox('Do you have hypertension?', ['0: No', '1: Yes'])
    heart_disease = st.selectbox('Do you have heart disease?', ['0: No', '1: Yes'])
    smoking_history = st.selectbox('Smoking history', ['0: Never', '1: Former', '2: Current'])
    bmi = st.number_input('Enter your BMI', min_value=0.0, max_value=100.0, step=0.1)
    HbA1c_level = st.number_input('Enter your HbA1c level', min_value=0.0, max_value=20.0, step=0.1)
    blood_glucose_level = st.number_input('Enter your blood glucose level', min_value=0, max_value=500, step=1)

    # Code for Prediction
    diagnosis = ''

    # Create a button for Prediction
    if st.button('Diabetes Test Result'):
        try:
            diagnosis = diabetes_prediction([gender.split(':')[0], age, hypertension.split(':')[0], 
                                             heart_disease.split(':')[0], smoking_history.split(':')[0], 
                                             bmi, HbA1c_level, blood_glucose_level])
            st.success(diagnosis)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Run the app
if __name__ == '__main__':
    main()

















