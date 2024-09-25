import streamlit as st
import pandas as pd 
import pickle
import numpy as np

# Load the scaler, label encoder, model, and class names
scaler = pickle.load(open("scaler.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))
class_names = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Unknown',
               'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
               'Banker', 'Writer', 'Accountant', 'Designer',
               'Construction Engineer', 'Game Developer', 'Stock Investor',
               'Real Estate Developer']

def Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                    weekly_self_study_hours, math_score, history_score, physics_score,
                    chemistry_score, biology_score, english_score, geography_score,
                    total_score, average_score):
    
    # Encode categorical variables
    gender_encoded = 1 if gender.lower() == 'female' else 0
    part_time_job_encoded = 1 if part_time_job else 0
    extracurricular_activities_encoded = 1 if extracurricular_activities else 0
    
    # Create feature array
    feature_array = np.array([[gender_encoded, part_time_job_encoded, absence_days, extracurricular_activities_encoded,
                               weekly_self_study_hours, math_score, history_score, physics_score,
                               chemistry_score, biology_score, english_score, geography_score, total_score, average_score]])
    
    # Scale features
    scaled_features = scaler.transform(feature_array)
    
    # Predict using the model
    probabilities = model.predict_proba(scaled_features)
    
    # Get top five predicted classes along with their probabilities
    top_classes_idx = np.argsort(-probabilities[0])[:5]
    top_classes_names_probs = [(class_names[idx], probabilities[0][idx]) for idx in top_classes_idx]
    
    return top_classes_names_probs

# Streamlit app
st.title('Career Recommendations')   
final_reccomendation = Recommendations(
    gender = st.selectbox('Gender', ['Male', 'Female']),
    part_time_job = st.checkbox('Part-time Job'),
    extracurricular_activities = st.checkbox('Extracurricular Activities'),
    absence_days = st.number_input('Absence Days', min_value=0, max_value=365, value=0),
    weekly_self_study_hours = st.number_input('Weekly Self Study Hours', min_value=0, max_value=168, value=0),
    math_score = st.number_input('Math Score', min_value=0, max_value=100, value=0),
    history_score = st.number_input('History Score', min_value=0, max_value=100, value=0),
    physics_score = st.number_input('Physics Score', min_value=0, max_value=100, value=0),
    chemistry_score = st.number_input('Chemistry Score', min_value=0, max_value=100, value=0),
    biology_score = st.number_input('Biology Score', min_value=0, max_value=100, value=0),
    english_score = st.number_input('English Score', min_value=0, max_value=100, value=0),
    geography_score = st.number_input('Geography Score', min_value=0, max_value=100, value=0),
    total_score = st.number_input('Total Score', min_value=0, max_value=1500, value=0),
    average_score = st.number_input('Average Score', min_value=0.0, max_value=100.0, value=0.0)
)

if st.button('Get Recommendations'):
    st.write("Top recommended careers with probabilities:")
    for class_name, probability in final_reccomendation:
        st.write(f"{class_name}: {probability:.2f}")
