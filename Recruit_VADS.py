#!/usr/bin/env python
# coding: utf-8

# In[8]:


import streamlit as st
import pandas as pd
import pickle
import os

# Function to load model and vectorizer with error handling
def load_model_and_vectorizer(model_filename, vectorizer_filename):
    # Get the current working directory
    current_working_directory = os.getcwd()
    
    # Combine paths with the current working directory
    model_path = os.path.join(current_working_directory, model_filename)
    vectorizer_path = os.path.join(current_working_directory, vectorizer_filename)

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        try:
            loaded_model = pickle.load(open(model_path, 'rb'))
            loaded_vectorizer = pickle.load(open(vectorizer_path, 'rb'))
            return loaded_model, loaded_vectorizer
        except Exception as e:
            st.error(f"Error loading model or vectorizer: {e}")
            return None, None
    else:
        st.error(f"Model or vectorizer file not found. Model: {model_path}, Vectorizer: {vectorizer_path}")
        return None, None

# Load the model and vectorizer
model_filename = 'Recruit_VADS_model.pkl'
vectorizer_filename = 'Tfidf_Vectorizer.pkl'

loaded_model, loaded_vectorizer = load_model_and_vectorizer(model_filename, vectorizer_filename)

# Continue with the rest of the code if the model and vectorizer are successfully loaded
if loaded_model is not None and loaded_vectorizer is not None:
    # Load resume data for displaying Candidate Name and Email ID
    resume_data_path = r"D:\1 ISB\Term 2\FP\FP project\Modifiedresumedata_data.csv"
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

