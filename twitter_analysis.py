import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Step 1: Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path, encoding='latin-1', header=None)
    data.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    return data

# Step 2: Preprocess the text data using regex
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|\#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

def preprocess_data(data):
    data = data.dropna(subset=['text', 'target'])
    data['cleaned_text'] = data['text'].apply(clean_text)
    data['target'] = data['target'].map({0: 'negative', 2: 'neutral', 4: 'positive'})
    return data

# Step 3: Split the dataset into training and testing data
def split_data(data):
    x = data['cleaned_text']
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
    return X_train, X_test, y_train, y_test

# Step 4: Feature extraction using TF-IDF vectorizer
def extract_features(X_train, X_test):
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    return X_train_tfidf, X_test_tfidf, tfidf

# Step 5: Train the model
def train_model(X_train_tfidf, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    return model

# Step 6: Evaluate the model
def evaluate_model(model, X_test_tfidf, y_test):
    y_pred = model.predict(X_test_tfidf)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Step 7: Save the model
def save_model(model, file_path):
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

def save_vectorizer(vectorizer, file_path):
    joblib.dump(vectorizer, file_path)
    print(f"TF-IDF Vectorizer saved to {file_path}")


# Step 8: Main function
def main():
    file_path = 'training.1600000.processed.noemoticon.csv'
    data = load_data(file_path)
    data = preprocess_data(data)

    X_train, X_test, y_train, y_test = split_data(data)
    X_train_tfidf, X_test_tfidf, tfidf = extract_features(X_train, X_test)

    model = train_model(X_train_tfidf, y_train)
    evaluate_model(model, X_test_tfidf, y_test)

    save_model(model, 'sentiment_model.pkl')
    save_vectorizer(tfidf, 'tfidf_vectorizer.pkl')


# Run the project
if __name__ == "__main__":
    main()