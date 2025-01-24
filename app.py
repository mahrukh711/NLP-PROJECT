import streamlit as st
import pandas as pd
from scraper import scrape_coffee_reviews
from processor import CoffeeSentimentAnalyzer
import plotly.express as px
import plotly.graph_objects as go

# Initialize the sentiment analyzer
analyzer = CoffeeSentimentAnalyzer()

# Page config
st.set_page_config(page_title="Coffee Sentiment Analysis", layout="wide")

# Title
st.title("☕ Coffee Sentiment Analysis")

# Sidebar
st.sidebar.header("Controls")
if st.sidebar.button("Scrape New Data"):
    with st.spinner("Scraping coffee reviews..."):
        df = scrape_coffee_reviews()
        df.to_csv("coffee_reviews.csv", index=False)
        st.success("Data scraped successfully!")

# Load data
try:
    df = pd.read_csv("coffee_reviews.csv")
    st.sidebar.success(f"Loaded {len(df)} reviews")
except FileNotFoundError:
    st.sidebar.warning("No data found. Please scrape new data.")
    df = pd.DataFrame()

if not df.empty:
    # Train model if needed
    if st.sidebar.button("Train Model"):
        with st.spinner("Training model..."):
            X, y = analyzer.prepare_data(df)
            analyzer.train(X, y)
            analyzer.save_model()
            st.success("Model trained successfully!")

    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Sentiment Distribution")
        if 'sentiment' in df.columns:
            fig = px.pie(df, names='sentiment', title="Review Sentiments")
            st.plotly_chart(fig)
    
    with col2:
        st.header("Rating Distribution")
        fig = px.histogram(df, x='rating', title="Rating Distribution")
        st.plotly_chart(fig)
    
    # Review Analysis
    st.header("Review Analysis")
    text_input = st.text_area("Enter a coffee review to analyze:")
    if text_input:
        try:
            sentiment = analyzer.predict(text_input)
            st.write(f"Predicted Sentiment: **{sentiment}**")
            
            # Show feature importance
            features = analyzer.extract_features(text_input)
            st.write("### Feature Analysis")
            for feature, value in features.items():
                if feature != 'compound':
                    st.write(f"**{feature}:** {value:.3f}")
        except Exception as e:
            st.error("Please train the model first!")
    
    # Show recent reviews
    st.header("Recent Reviews")
    num_reviews = st.slider("Number of reviews to display", 1, 10, 5)
    st.dataframe(df[['title', 'rating', 'text']].head(num_reviews))