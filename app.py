import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Configure the Streamlit app
st.set_page_config(
    page_title="Rock vs Mine Prediction",
    page_icon="⛏️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Set the CSS styles for the app
st.markdown(
    """
    <style>
    body {
        background-color: #f0f0f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Rock vs Mine Prediction")

# Load the saved model
loaded_model = pickle.load(open('model.pkl', 'rb'))

# Define the input fields for the sample data
input_data = st.text_area("Enter comma-separated features:", "0.0412,0.1135,0.0518,0.0232,0.0646,0.1124,0.1787,0.2407,0.2682,0.2058,0.1546,0.2671,0.3141,0.2904,0.3531,0.5079,0.4639,0.1859,0.4474,0.4079,0.5400,0.4786,0.4332,0.6113,0.5091,0.4606,0.7243,0.8987,0.8826,0.9201,0.8005,0.6033,0.2120,0.2866,0.4033,0.2803,0.3087,0.3550,0.2545,0.1432,0.5869,0.6431,0.5826,0.4286,0.4894,0.5777,0.4315,0.2640,0.1794,0.0772,0.0798,0.0376,0.0143,0.0272,0.0127,0.0166,0.0095,0.0225,0.0098,0.0085")
input_data = input_data.split(',')
input_data = [float(x.strip()) for x in input_data]

# Define a function to make predictions
def predict_rock_or_mine(input_data):
    # Change the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the numpy array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    return prediction[0]

# Call the predict_rock_or_mine function when the user clicks the "Predict" button
if st.button("Predict"):
    prediction = predict_rock_or_mine(input_data)
    st.write("### Prediction Result")
    if prediction == "R":
        st.write("The object is Rock!")
    else:
        st.write("The object is Mine!")

# Display a sample prediction if the user clicks the "Use Sample Data" button
if st.button("Use Sample Data"):
    sample_data = [0.0412, 0.1135, 0.0518, 0.0232, 0.0646, 0.1124, 0.1787, 0.2407, 0.2682, 0.2058, 0.1546, 0.2671, 0.3141, 0.2904, 0.3531, 0.5079, 0.4639, 0.1859, 0.4474, 0.4079, 0.5400, 0.4786, 0.4332, 0.6113, 0.5091, 0.4606, 0.7243, 0.8987, 0.8826, 0.9201, 0.8005, 0.6033, 0.2120, 0.2866, 0.4033, 0.2803, 0.3087, 0.3550, 0.2545, 0.1432, 0.5869, 0.6431, 0.5826, 0.4286, 0.4894, 0.5777, 0.4315, 0.2640, 0.1794, 0.0772, 0.0798, 0.0376, 0.0143, 0.0272, 0.0127, 0.0166, 0.0095, 0.0225, 0.0098, 0.0085]
    prediction = predict_rock_or_mine(sample_data)
    st.write("### Prediction Result")
    if prediction == "R":
        st.write("The object is Rock!")
    else:
        st.write("The object is Mine!")
