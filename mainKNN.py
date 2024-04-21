import streamlit as st
import numpy as np
from PIL import Image
import pickle

load_model = pickle.load(open('model.pkl', 'rb'))
st.title('10 Year CHD Prediction')

img = Image.open('chd.jpg')
img = img.resize((700, 418))
st.image(img, use_column_width=False)

st.sidebar.title('Predict')
st.sidebar.write('Enter the patient details')

male = st.sidebar.selectbox('Male', [0, 1])
age = st.sidebar.slider('Age', 20, 100, 40)
education = st.sidebar.selectbox('Education', [1, 2, 3, 4])
currentSmoker = st.sidebar.selectbox('Current Smoker', [0, 1])
cigsPerDay = st.sidebar.slider('Cigarettes per Day', 0, 70, 10)
BPMeds = st.sidebar.selectbox('BP Medication', [0, 1])
prevalentStroke = st.sidebar.selectbox('Prevalent Stroke', [0, 1])
prevalentHyp = st.sidebar.selectbox('Prevalent Hypertension', [0, 1])
diabetes = st.sidebar.selectbox('Diabetes', [0, 1])
totChol = st.sidebar.slider('Total Cholesterol', 100, 600, 200)
sysBP = st.sidebar.slider('Systolic Blood Pressure', 80, 250, 120)
diaBP = st.sidebar.slider('Diastolic Blood Pressure', 40, 150, 80)
BMI = st.sidebar.slider('BMI', 15.0, 60.0, 25.0)
heartRate = st.sidebar.slider('Heart Rate', 40, 150, 70)
glucose = st.sidebar.slider('Glucose', 40, 400, 100)

input_data = [[male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp,
               diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]]


st.subheader('Label Class')
CHDtest = np.array(['Tidak','Ya'])
st.write(CHDtest)

prediksi = load_model.predict(input_data)
prediksi_proba = load_model.predict_proba(input_data)

st.write('probabilitas:',prediksi_proba)
st.write('Prediction:', CHDtest[prediksi])