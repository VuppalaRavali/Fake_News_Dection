import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn


text_input = st.text_input("Enter the News","")

text_input=text_input.lower()

if st.button("Submit"):
    
    # Unpickle classifier
    clf = joblib.load("finalized_model.pkl")
    
    # input to lower
    # X = np.array(text_input).reshape(-1,1)
    # st.text(X)
    # Get prediction
    prediction = clf.predict(np.array(text_input).reshape(1,))
    
    # Output prediction
    st.text(f"This news is a {prediction}")
