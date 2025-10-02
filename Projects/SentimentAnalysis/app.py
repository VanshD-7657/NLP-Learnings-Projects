import streamlit as st
import pickle

st.title("Sentiment Analysis App 🎯")
st.write("Enter a review below to analyze its sentiment.")

# Load model and vectorizer (already trained & saved earlier)
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("CountVectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Input from user
review = st.text_area("Enter your review here...")

if st.button("Predict"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review before predicting.")
    else:
        # Transform input text
        review_vectorized = vectorizer.transform([review])

        # Prediction
        prediction = model.predict(review_vectorized)[0]

        if prediction == 1:
            st.success("✅ The sentiment of the review is **Positive**")
        else:
            st.error("❌ The sentiment of the review is **Negative**")
