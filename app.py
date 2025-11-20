import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("🎯 Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn based on their information")

# Load model and feature columns
@st.cache_resource
def load_model_and_features():
    try:
        model = joblib.load('model.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        return model, feature_columns
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model, feature_columns = load_model_and_features()
st.success(f"✅ Model loaded successfully! Expecting {len(feature_columns)} features")

# Create input fields for ALL original features
st.header("Enter Customer Information")

# Group inputs logically
col1, col2 = st.columns(2)

with col1:
    # Demographics
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Partner', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['Yes', 'No'])
    
    # Account Info
    tenure = st.number_input('Tenure (months)', min_value=0, max_value=100, value=12)
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.selectbox('Payment Method', [
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ])

with col2:
    # Services
    phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
    multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
    device_protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
    streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
    
    # Charges
    monthly_charges = st.number_input('Monthly Charges ($)', min_value=0.0, value=50.0)
    total_charges = st.number_input('Total Charges ($)', min_value=0.0, value=600.0)

# Prediction button
if st.button('🔍 Predict Churn Risk'):
    try:
        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })
        
        # Apply the same preprocessing that was used during training
        # This is a simplified example - adjust based on your actual preprocessing
        
        # One-hot encode categorical variables (must match training)
        categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                              'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                              'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                              'PaperlessBilling', 'PaymentMethod']
        
        # Create one-hot encoded features
        encoded_data = pd.get_dummies(input_data, columns=categorical_columns)
        
        # Ensure all expected columns exist (fill missing with 0)
        for col in feature_columns:
            if col not in encoded_data.columns:
                encoded_data[col] = 0
        
        # Select only the columns in the same order as training
        encoded_data = encoded_data[feature_columns]
        
        # Verify the shape
        st.write(f"Input shape: {encoded_data.shape}")
        
        # Make prediction
        prediction = model.predict(encoded_data)[0]
        probability = model.predict_proba(encoded_data)[0]
        
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
        st.write("Please check your input values and try again.")

# Add information about the model
st.sidebar.header("About This App")
st.sidebar.info("This app predicts customer churn using machine learning.")
st.sidebar.write(f"Model expects {len(feature_columns)} features after preprocessing.")
