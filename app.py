import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page config
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Title
st.title("🎯 Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn based on their information")

# Load your trained model (make sure you have saved it as 'model.pkl')
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load the model
try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except:
    st.error("❌ Model not found! Please make sure 'model.pkl' exists in the same folder")
    st.stop()

# Create input fields for customer features
st.header("Enter Customer Information")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input('Tenure (months)', min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input('Monthly Charges ($)', min_value=0.0, value=50.0)
    total_charges = st.number_input('Total Charges ($)', min_value=0.0, value=600.0)

with col2:
    contract_type = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])

# Additional features
st.subheader("Additional Information")
col3, col4 = st.columns(2)

with col3:
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Has Partner', ['Yes', 'No'])
    dependents = st.selectbox('Has Dependents', ['Yes', 'No'])

with col4:
    phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
    multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
    online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])

# Encode categorical variables (adjust based on your model's encoding)
def encode_features():
    # Contract type encoding
    contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    contract_encoded = contract_mapping[contract_type]
    
    # Internet service encoding
    internet_mapping = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
    internet_encoded = internet_mapping[internet_service]
    
    # Other binary features (simplified - adjust based on your model)
    features = [
        tenure,
        monthly_charges,
        total_charges,
        contract_encoded,
        internet_encoded,
        1 if senior_citizen == 'Yes' else 0,
        1 if gender == 'Male' else 0,
        1 if partner == 'Yes' else 0,
        1 if dependents == 'Yes' else 0,
        1 if phone_service == 'Yes' else 0,
        # Add more features as needed based on your model
    ]
    
    return np.array(features).reshape(1, -1)

# Prediction button
if st.button('🔍 Predict Churn Risk'):
    try:
        # Prepare input data
        input_data = encode_features()
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
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

# Add some information about the model
st.sidebar.header("About This App")
st.sidebar.info("This app predicts customer churn using machine learning.")
st.sidebar.write("The model was trained on telecom customer data.")