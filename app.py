import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# NLTK dependencies
nltk.download('vader_lexicon')

# Load AI models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sia = SentimentIntensityAnalyzer()
sentiment_analyzer = pipeline("sentiment-analysis")

# Function to scrape news from a URL
def scrape_news(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = " ".join([p.get_text() for p in paragraphs])
        return article_text[:1024] if len(article_text) > 1024 else article_text
    except Exception as e:
        return f"Error: {e}"

# Function to summarize text
def summarize_text(text):
    if len(text) < 50:
        return "❌ Article is too short to summarize."
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Function to analyze sentiment
def analyze_sentiment(text):
    result = sentiment_analyzer(text[:512])
    label = result[0]['label']
    score = result[0]['score']
    if label == "POSITIVE":
        return f"🙂 Positive ({score:.2f})"
    elif label == "NEGATIVE":
        return f"☹️ Negative ({score:.2f})"
    else:
        return f"😐 Neutral ({score:.2f})"

# Streamlit UI
st.title("📰 AI News Summarizer with Sentiment Analysis")
st.write("Enter a news article URL to get a summary and sentiment analysis.")

news_url = st.text_input("🔗 Enter News URL:")

if st.button("Summarize"):
    st.write("Fetching news article...")
    article_text = scrape_news(news_url)

    if "Error" in article_text:
        st.error("❌ Could not fetch news. Please try another URL.")
    else:
        st.subheader("📰 Original Article (First 500 characters):")
        st.write(article_text[:500] + "...")

        st.subheader("🔍 Summary:")
        summary = summarize_text(article_text)
        st.success(summary)

        st.subheader("📊 Sentiment Analysis:")
        sentiment = analyze_sentiment(summary)
        st.info(sentiment)
