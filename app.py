import streamlit as st
import pandas as pd
import pickle

# Load the trained model
# Ensure 'best_linear_regression_model.pkl' is in the same directory as this app.py or provide the full path
try:
    with open('best_linear_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'best_linear_regression_model.pkl' not found. Please ensure it's in the correct directory.")
    st.stop()

# Set page config
st.set_page_config(page_title="Salary Prediction App", layout="centered")

st.title("💼 Salary Prediction App")
st.write("Enter the years of experience to get a salary prediction using our trained Linear Regression model.")

# Input widget for Years of Experience
years_experience = st.slider(
    "Years of Experience",
    min_value=0.0,
    max_value=15.0,
    value=5.0,
    step=0.1,
    format="%.1f"
)

# Make prediction when button is clicked or slider changes
if st.button("Predict Salary") or years_experience is not None:
    # Create a DataFrame for prediction
    input_data = pd.DataFrame({'YearsExperience': [years_experience]})

    # Make prediction
    predicted_salary = model.predict(input_data)[0]

    st.success(f"Predicted Salary: ${predicted_salary:,.2f}")

st.markdown("---")
st.info("This app uses a Linear Regression model trained on a sample salary dataset.")
