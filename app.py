import pandas as pd
import streamlit as st
import joblib

# Load model
model = joblib.load('credit_model.pkl')

st.title('Credit Risk Analysis')
st.write('Enter applicant details to predict credit risk')

# ---------------------------
# INPUTS
# ---------------------------
age = st.number_input('Age', min_value=18, max_value=75, value=30)
sex = st.selectbox('Sex', ['male','female'])
job = st.number_input('Job (0–3)', min_value=0, max_value=3, value=1)
housing = st.selectbox('Housing', ['own','rent','free'])
saving_accounts = st.selectbox('Saving Accounts', ['little','moderate','rich','quite rich','unknown'])
checking_account = st.selectbox('Checking Account', ['little','moderate','rich','unknown'])
credit_amount = st.number_input('Credit Amount', min_value=100, value=1000)
duration = st.number_input('Duration (months)', min_value=4, max_value=72, value=12)

# ---------------------------
# CREATE INPUT DATAFRAME
# ---------------------------
input_df = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'job': [job],
    'housing': [housing],
    'saving accounts': [saving_accounts],
    'checking account': [checking_account],
    'credit amount': [credit_amount],
    'duration': [duration],
})

# ---------------------------
# APPLY SAME ENCODING
# ---------------------------
input_encoded = pd.get_dummies(input_df)

# Load training columns (IMPORTANT)
model_columns = joblib.load('model_columns.pkl')

# Align columns
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

# ---------------------------
# PREDICTION
# ---------------------------
if st.button('Predict Risk'):
    prediction = model.predict(input_encoded)[0]

    if prediction == 1:
        st.success('Predicted: Good Credit Risk ✅')
    else:
        st.error('Predicted: Bad Credit Risk ❌')