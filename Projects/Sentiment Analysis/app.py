import streamlit as st
import pandas as pd
import numpy as np  

st.write("Enter the review below to know the sentiment")
review = st.text_area("Enter the review")
if st.button("Predict"):
    import pickle
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    # Load the dataset
    data = pd.read_csv(r"C:\Users\AMAN\Downloads\DATA Scientist\GitHub\NLP Learning\Projects\Sentiment Analysis\amazon_alexa.tsv",sep='\t')

    # Preprocess the data
    X = data['verified_reviews']
    y = data['feedback']

    # Vectorize the text data
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    # Train the model
    model = LogisticRegression()
    model.fit(X_vectorized, y)

    # Save the model and vectorizer
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('CountVectorizer.pkl', 'wb') as vec_file:
        pickle.dump(vectorizer, vec_file)

    # Load the model and vectorizer
    with open('model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    with open('CountVectorizer.pkl', 'rb') as vec_file:
        loaded_vectorizer = pickle.load(vec_file)

    # Transform the input review
    review_vectorized = loaded_vectorizer.transform([review])

    # Predict the sentiment
    prediction = loaded_model.predict(review_vectorized)

    if prediction[0] == 1:
        st.write("The sentiment of the review is Positive")
    else:
        st.write("The sentiment of the review is Negative")