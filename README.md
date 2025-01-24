# Coffee Sentiment Analysis Project

## Overview
This project implements a sentiment analysis system for coffee reviews using Natural Language Processing (NLP) techniques. It scrapes coffee reviews from coffeereview.com, processes them using various NLP methods, and provides sentiment analysis through a Streamlit web interface.

## Features
- Web scraping of coffee reviews
- Text preprocessing (tokenization, stemming)
- Sentiment analysis using VADER and machine learning
- Interactive web interface
- Real-time review analysis
- Data visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/coffee-sentiment-analysis.git
cd coffee-sentiment-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Project Structure
```
coffee-sentiment-analysis/
├── app.py              # Streamlit web application
├── processor.py        # NLP processing and model training
├── scraper.py         # Web scraping functionality
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Usage
1. Start the application using `streamlit run app.py`
2. Click "Scrape New Data" to collect coffee reviews
3. Train the model using the "Train Model" button
4. Analyze new reviews using the text input area
5. View visualizations and statistics in the dashboard

## Technical Details

### NLP Techniques Used
- Tokenization using NLTK
- Stemming with Porter Stemmer
- Stop word removal
- TF-IDF feature extraction
- VADER sentiment analysis

### Model
- Random Forest Classifier
- Features: text length, word count, VADER scores, TF-IDF

### Performance Metrics
- Sentiment classification (Positive, Neutral, Negative)
- Feature importance analysis
- Real-time sentiment scoring

## Dependencies
- beautifulsoup4==4.12.3
- nltk==3.8.1
- numpy==1.24.3
- pandas==2.0.3
- plotly==5.18.0
- requests==2.31.0
- scikit-learn==1.3.0
- streamlit==1.31.1