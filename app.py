import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title='Customer Churn Predicition',
    layout='centered',
    page_icon='icon/churn-rate.png'
)

model_path = Path('models/best_model.pkl')

@st.cache_resource
def load_model():
    return joblib.load(model_path)


try:
    model_artifact = load_model()
except FileNotFoundError:
    st.error('Model file not found')
    st.info('Please make sure "Best Model.pkl" exisit inside the models folder')
    st.stop()
    
model = model_artifact['model']
threshold = model_artifact['threshold']
feature_columns = model_artifact['feature_columns']

st.title("Customer Churn Predicition")
st.write(
    "Enter the customer's information below to prediction"
    "For the estimate Customer will churn or not"
)

st.divider()

with st.form('Customer Churn Predicition'):
    st.subheader("Customer information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        Gender = st.selectbox(
            'Sex',
            ['Male', "Female"]
        )
        SeniorCitizen = st.number_input(
            'SeniorCitizen',
            min_value=0,
            max_value=1,
            value=0,
            step=1
        )
        
        Partner = st.selectbox(
            'Partner',
            ['Yes', 'No']
        )
        
        Dependents = st.selectbox(
            'Dependents',
            ['Yes', 'No']
        )
        
        tenure = st.number_input(
            'Tenure',
            min_value=0,
            max_value=100,
            value=50,
            step=1
        )
        
    with col2:
        PhoneService = st.selectbox(
            'PhoneService',
            ['Yes', 'No']
        )
        
        MultipleLines = st.selectbox(
            'MultipleLines',
            ['Yes', 'No', 'No phone service']
        )
        
        InternetService = st.selectbox(
            'InternetService',
            ['Fiber optic','DSL','No']
        )
        
        OnlineSecurity = st.selectbox(
            'OnlineSecurity',
            ['Yes', 'No','No internet service']
        )
        
        OnlineBackup=st.selectbox(
            'OnlineBackup',
            ['Yes', 'No', 'No internet service']
        )
        
    with col3:
        DeviceProtection = st.selectbox(
            'DeviceProtection',
            ['Yes', 'No', 'No internet service']
        )
        
        TechSupport = st.selectbox(
            'TechSupport',
            ['Yes', 'No', 'No internet service']
        )
        
        StreamingTV = st.selectbox(
            'StreamingTV',
            ['Yes', 'No', 'No internet service']
        )
        
        StreamingMovies = st.selectbox(
            'StreamingMovies',
            ['Yes', 'No', 'No internet service']
        )
        
        Contract = st.selectbox(
            'Contract',
            ['Month-to-month', 'Two year', 'One year']
        )
    
    with col4:
        PaperlessBilling = st.selectbox(
            'PaperlessBilling',
            ['Yes', 'No']
        )
        
        PaymentMethod = st.selectbox(
            'PaymentMethod',
            ['Electronic check', 'Mailed check','Bank transfer (automatic)', 'Credit card (automatic)']
        )
        
        MonthlyCharges = st.number_input(
            'MonthlyCharges',
            min_value=0,
            max_value=150,
            value=20,
            step=1
        )
        
        TotalCharges = st.number_input(
            'TotalCharges',
            min_value=0,
            max_value=9999,
            value=100,
            step=1
        )
    predict_button = st.form_submit_button(
        "Pedict the Customer will churn?"
    )
    
if predict_button:
    
    input_data = pd.DataFrame({
        'gender' : [Gender],
        'SeniorCitizen' : [SeniorCitizen],
        'Partner' : [Partner],
        "Dependents" : [Dependents],
        'tenure' : [tenure],
        'PhoneService' : [PhoneService],
        'MultipleLines' : [MultipleLines],
        'InternetService' : [InternetService],
        'OnlineSecurity' : [OnlineSecurity],
        'OnlineBackup'  : [OnlineBackup],
        'DeviceProtection' : [DeviceProtection],
        'TechSupport' : [TechSupport],
        'StreamingTV' : [StreamingTV],
        'StreamingMovies' : [StreamingMovies],
        'Contract' : [Contract],
        'PaperlessBilling' : [PaperlessBilling],
        'PaymentMethod' : [PaymentMethod],
        'MonthlyCharges' : [MonthlyCharges],
        'TotalCharges' : [TotalCharges]
    })
    
    input_data = input_data[feature_columns] #for traning order
    
    proba = model.predict_proba(input_data)[0][1]
    prediction = int(proba >= threshold)
    
    st.divider()
    st.subheader('Prediction Result')
    
    if prediction == 1:
        st.error("Yes, this customer is likely to churn.")
    else:
        st.success("No, this customer is not likely to churn.")
        
    with st.expander('input data'):
        st.dataframe(
            input_data,
            use_container_width=True
        )