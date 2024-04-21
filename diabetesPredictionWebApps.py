import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from sklearn.naive_bayes import GaussianNB

st.write("""
# Diabetes Prediction
A Comprehensive Dataset for Predicting Diabetes.\n
The Diabetes prediction is a collection of medical data from patients, along with their diabetes status (positive or negative). The data includes features such as age, gender, body mass index (BMI), hypertension, heart disease, smoking history, HbA1c level, and blood glucose level. This dataset can be used to build machine learning models to predict diabetes in patients based on their medical history and demographic information. This can be useful for healthcare professionals in identifying patients who may be at risk of developing diabetes and in developing personalized treatment plans.\n
Data files [Diabetes prediction dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
BY [MOHAMMED MUSTAFA](https://www.kaggle.com/iammustafatz)
""")

img = Image.open('dataset-cover.png')
img = img.resize((700, 418))
st.image(img, use_column_width=False)

st.sidebar.header('Input Parameters')

# Upload File CSV untuk parameter inputan
upload_file = st.sidebar.file_uploader('Upload your CSV file', type=['csv'])
if upload_file is not None:
    inputan = pd.read_csv(upload_file)
else:
    def input_user():
        gender = st.sidebar.selectbox('Gender', ('Male', 'Female', 'Other'))
        age = st.sidebar.slider('Age', 0.08, 80.0, 40.0)
        hypertension = st.sidebar.slider('Hypertension', 0, 1, 0)
        heart_disease = st.sidebar.slider('Heart Disease', 0, 1, 0)
        smoking_history = st.sidebar.selectbox('Smoking History', ('No Info', 'never', 'former', 'current', 'not current'))
        bmi = st.sidebar.slider('BMI', 10.0, 95.7, 40.0)
        HbA1c_level = st.sidebar.slider('HbA1c', 3.5, 9.0, 5.0)
        blood_glucose_level = st.sidebar.slider('Blood Glucose Level', 80, 300, 150)
        data = {
            'gender' : gender,
            'age' : age,
            'hypertension' : hypertension,
            'heart_disease' : heart_disease,
            'smoking_history' : smoking_history,
            'bmi' : bmi,
            'HbA1c_level' : HbA1c_level,
            'blood_glucose_level' : blood_glucose_level
        }
        fitur = pd.DataFrame(data, index=[0])
        return fitur
    inputan = input_user()

# menggabungkan inputan dan dataset diabetes prediction
diabetesPrediction_raw = pd.read_csv('diabetes_prediction_dataset.csv')
diabetesPredictions = diabetesPrediction_raw.drop(columns=['diabetes'])
df = pd.concat([inputan, diabetesPredictions], axis=0)

# encode untuk fitur ordinal
encode = ['gender', 'smoking_history']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1] # ambil baris pertama (input data user)

# menampilkan parameter hasil inputan
st.subheader('Input Parameters')

if upload_file is not None:
    st.write("""
    True = 1\n
    False = 0
    """)
    st.write(df)
else:
    st.write('Waiting for the csv file to upload. currently using the input sample')
    st.write("""
    True = 1\n
    False = 0
    """)
    st.write(df)

# load model NBC
load_model = pickle.load(open('modelNBC_diabetes_prediction.pkl', 'rb'))

# terapkan NBC
prediksi = load_model.predict(df)
prediksi_proba = load_model.predict_proba(df)

st.subheader('Label Class Description')
st.write("""
Positive = 1\n
Negative = 0
""")
status_diabetes = np.array([0, 1])
st.write(status_diabetes)

st.subheader('Prediction Results (Diabetes Prediction)')
st.write("""
Positive = 1\n
Negative = 0
""")
st.write(status_diabetes[prediksi])

st.subheader('The Probability of the Predicted Outcome (Diabetes Prediction)')
st.write("""
Positive = 1\n
Negative = 0
""")
st.write(prediksi_proba)