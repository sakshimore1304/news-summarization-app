#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import requests
import boto3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# AWS S3 Configuration
S3_BUCKET_NAME = "your-s3-bucket-name"

def upload_to_s3(data, filename):
    s3 = boto3.client('s3')
    s3.put_object(Bucket=S3_BUCKET_NAME, Key=filename, Body=data)
    return f"File uploaded to {S3_BUCKET_NAME}/{filename}"

# Fetch news from API
def get_news(api_key, query="technology"):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
    response = requests.get(url).json()
    articles = response.get("articles", [])
    return articles

# Summarize news
def summarize_news(articles):
    summaries = []
    for article in articles[:5]:  # Summarizing top 5 articles
        title = article["title"]
        desc = article["description"]
        sentiment = sia.polarity_scores(desc)["compound"]
        summaries.append({"Title": title, "Summary": desc, "Sentiment": sentiment})
    return summaries

# Streamlit UI
st.title("News Summarization & Sentiment Analysis App")

news_api_key = st.text_input("Enter your News API Key:", type="password")
query = st.text_input("Enter topic (e.g., AI, Finance, Sports):", "technology")

if st.button("Fetch News"):
    articles = get_news(news_api_key, query)
    summaries = summarize_news(articles)
    
    df = pd.DataFrame(summaries)
    st.dataframe(df)

    # Upload summary to S3
    s3_result = upload_to_s3(df.to_csv(index=False), "news_summary.csv")
    st.write(s3_result)

    # Sentiment Chart
    st.subheader("Sentiment Analysis")
    fig, ax = plt.subplots()
    sns.histplot(df["Sentiment"], kde=True, bins=10)
    st.pyplot(fig)

