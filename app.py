import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("🎯 Customer Churn Prediction App")

# Load model and preprocessor
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

try:
    model, preprocessor = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Input fields (use your original ones)
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Partner', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['Yes', 'No'])
    tenure = st.number_input('Tenure (months)', min_value=0, max_value=100, value=12)
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])

with col2:
    phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.selectbox('Payment Method', [
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    monthly_charges = st.number_input('Monthly Charges ($)', min_value=0.0, value=50.0)
    total_charges = st.number_input('Total Charges ($)', min_value=0.0, value=600.0)

# Prediction
if st.button('🔍 Predict Churn Risk'):
    try:
        # Create input DataFrame matching training format
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [senior_citizen],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })

        # Apply same preprocessing as training
        input_processed = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_processed)[0]
        probability = model.predict_proba(input_processed)[0]

        # Display results
        st.subheader("Prediction Results:")
        if prediction == 1:
            st.error(f"⚠️ HIGH RISK: Customer is likely to churn")
            st.write(f"Churn Probability: {probability[1]:.2%}")
            st.write(f"Stay Probability: {probability[0]:.2%}")
        else:
            st.success(f"✅ LOW RISK: Customer is likely to stay")
            st.write(f"Stay Probability: {probability[0]:.2%}")
            st.write(f"Churn Probability: {probability[1]:.2%}")

    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
