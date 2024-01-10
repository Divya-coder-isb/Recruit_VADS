#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import pickle
import os

# Function to load model and vectorizer
def load_model_and_vectorizer(model_filename, vectorizer_filename):
    try:
        loaded_model = pickle.load(open(model_filename, 'rb'))
        loaded_vectorizer = pickle.load(open(vectorizer_filename, 'rb'))
        return loaded_model, loaded_vectorizer
    except FileNotFoundError:
        st.error(f"Error loading model or vectorizer: File not found. Model: {model_filename}, Vectorizer: {vectorizer_filename}")
        return None, None

# Specify the correct paths for the model and vectorizer
model_filename = 'D:/1 ISB/Term 2/FP/FP project/Recruit_VADS_model.pkl'
vectorizer_filename = 'D:/1 ISB/Term 2/FP/FP project/Tfidf_Vectorizer.pkl'

# Load the model and vectorizer
loaded_model, loaded_vectorizer = load_model_and_vectorizer(model_filename, vectorizer_filename)

# Continue with the rest of the code if the model and vectorizer are successfully loaded
if loaded_model is not None and loaded_vectorizer is not None:
    # Load resume data for displaying Candidate Name and Email ID
    resume_data_path = 'D:/1 ISB/Term 2/FP/FP project/Modifiedresumedata_data.csv'
    resume_data = pd.read_csv(resume_data_path)

    # Streamlit UI
    st.title('Recruit VADS - Candidate Relevancy Predictor')

    # Input fields
    job_title = st.text_input('Job Title:')
    skills = st.text_area('Skills:')
    experience = st.text_input('Experience:')
    certification = st.text_input('Certification:')

    # Apply button
    if st.button('Apply'):
        # Get relevancy score using the model
        input_features = [job_title, skills, certification, experience]
        input_vector = loaded_vectorizer.transform(input_features).toarray()
        similarity = loaded_model.dot(input_vector.T)

        # Sort candidates by descending order of similarity
        sorted_indices = similarity.argsort(axis=0)[::-1]
        sorted_similarity = similarity[sorted_indices]

        # Format output as a dataframe
        output_df = pd.DataFrame()
        output_df['Candidate Name'] = resume_data['Candidate Name'][sorted_indices].squeeze()
        output_df['Email ID'] = resume_data['Email ID'][sorted_indices].squeeze()
        output_df['Relevancy Score'] = (sorted_similarity * 100).round(2).squeeze()
        output_df['Relevancy Score'] = output_df['Relevancy Score'].astype(str) + '%'

        # Display the results
        st.table(output_df[['Candidate Name', 'Email ID', 'Relevancy Score']])


# In[2]:


import os

model_filename = 'D:/1 ISB/Term 2/FP/FP project/Recruit_VADS_model.pkl'
vectorizer_filename = 'D:/1 ISB/Term 2/FP/FP project/Tfidf_Vectorizer.pkl'

# Check if the model file exists
if os.path.exists(model_filename):
    print(f"Model file exists at: {model_filename}")
else:
    print(f"Model file does not exist at: {model_filename}")

# Check if the vectorizer file exists
if os.path.exists(vectorizer_filename):
    print(f"Vectorizer file exists at: {vectorizer_filename}")
else:
    print(f"Vectorizer file does not exist at: {vectorizer_filename}")


# In[ ]:




