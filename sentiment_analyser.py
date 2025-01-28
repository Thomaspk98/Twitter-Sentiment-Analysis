import streamlit as st
import joblib
import re

# Load the model and TF-IDF vectorizer
def load_model_and_vectorizer():
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

# Clean the input text using regex
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|\#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

# Predict sentiment
def predict_sentiment(model, vectorizer, tweet):
    cleaned_tweet = clean_text(tweet)
    tweet_tfidf = vectorizer.transform([cleaned_tweet])
    prediction = model.predict(tweet_tfidf)[0]
    return prediction

# Main function for the Streamlit app
def main():
    model, vectorizer = load_model_and_vectorizer()

    st.title("Twitter Sentiment Analysis")
    st.write("This app predicts the sentiment of a tweet as **Negative** or **Positive**.")

    tweet = st.text_area("Enter a tweet:", "")

    # Predict sentiment when the user clicks the button
    if st.button("Predict Sentiment"):
        if tweet.strip() == "":
            st.warning("Please enter a tweet.")
        else:
            sentiment =  predict_sentiment(model, vectorizer, tweet)
            output = sentiment.capitalize()
            if sentiment == 'negative':
                st.error(f"Predicted Sentiment: **{output}** 😠")
            else:
                st.success(f"Predicted Sentiment: **{output}** 😊")

# Run the app
if __name__ == "__main__":
    main()