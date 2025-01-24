import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.sentiment import vader
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

class CoffeeSentimentAnalyzer:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.sia = vader.SentimentIntensityAnalyzer()
        self.vectorizer = TfidfVectorizer()
        self.model = RandomForestClassifier()
        
    def preprocess_text(self, text):
        # Tokenization
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and stem
        tokens = [self.stemmer.stem(token) for token in tokens 
                 if token.isalnum() and token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def extract_features(self, text):
        # Basic features
        features = {}
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        
        # VADER sentiment scores
        sentiment_scores = self.sia.polarity_scores(text)
        features.update(sentiment_scores)
        
        return features
    
    def prepare_data(self, df):
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Extract features
        features_df = pd.DataFrame([self.extract_features(text) 
                                  for text in df['text']])
        
        # TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(df['processed_text'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), 
                               columns=self.vectorizer.get_feature_names_out())
        
        # Combine all features
        X = pd.concat([features_df, tfidf_df], axis=1)
        
        # Create labels (assuming ratings are 0-100)
        df['sentiment'] = pd.cut(pd.to_numeric(df['rating'].str.replace('[^\d.]', ''), 
                                             errors='coerce'),
                               bins=[0, 60, 80, 100],
                               labels=['negative', 'neutral', 'positive'])
        
        return X, df['sentiment']
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def save_model(self, path='coffee_sentiment_model.pkl'):
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'model': self.model
            }, f)
    
    def load_model(self, path='coffee_sentiment_model.pkl'):
        with open(path, 'rb') as f:
            models = pickle.load(f)
            self.vectorizer = models['vectorizer']
            self.model = models['model']
    
    def predict(self, text):
        processed_text = self.preprocess_text(text)
        features = self.extract_features(text)
        features_df = pd.DataFrame([features])
        
        tfidf_matrix = self.vectorizer.transform([processed_text])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), 
                               columns=self.vectorizer.get_feature_names_out())
        
        X = pd.concat([features_df, tfidf_df], axis=1)
        return self.model.predict(X)[0]