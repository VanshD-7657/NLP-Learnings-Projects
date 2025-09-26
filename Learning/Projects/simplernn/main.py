import numpy as np 
import pandas as pd
import tensorflow as tf
import streamlit as st  
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model



# Load the dataset
word_index = imdb.get_word_index()
reverse_word_index = {value:key for (key,value) in word_index.items()}

# Loading the pre-trained model
model = load_model(r'C:\Users\AMAN\Downloads\DATA Scientist\GitHub\NLP Learning\Learning\Projects\simplernn\simplernn_model.h5')

# Function to decode Reviews
def decode_review(encoded_review):
  return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_input(text):
  words = text.lower().split()
  encoded_review = [word_index.get(word,2) + 3 for word in words]
  padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
  return padded_review



# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write("Enter a movie review to predict its sentiment (Positive/Negative)")

# User input
user_input = st.text_area("Movie Review", "Type your review here...")

if st.button('Predict Sentiment'):
  preprocess_input = preprocess_input(user_input)
 
   # Make prediction
  prediction = model.predict(preprocess_input)
  sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

  # Display the result

  st.write(f'Sentiment: {sentiment}')
  st.write(f'Prediction Score: {prediction[0][0]:.4f}')

else:
  st.write("Please enter a review and click 'Predict Sentiment' ")