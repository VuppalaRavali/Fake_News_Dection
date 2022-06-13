import streamlit as st
import pandas as pd
import joblib
import sklearn


text_input = st.text_input("Enter the News","")

text_input=text_input.lower()
text_area_input = st.text_area("Result")

if st.button("Submit"):
    
    # Unpickle classifier
    clf = joblib.load("finalized_model.pkl")
    
    # input to lower
    X = [[text_input]]   
    # st.text(X)
    # Get prediction
    prediction = clf.predict(X)
    
    # Output prediction
    st.text(f"This news is a {prediction}")
