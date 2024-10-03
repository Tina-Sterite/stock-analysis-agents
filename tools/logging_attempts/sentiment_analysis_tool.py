# sentiment_analysis_tool.py
import os
import praw
import torch
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from crewai_tools import tool
import logging

# Import the logger
from logger import logger

# Download hf model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text.
    """
    try:        
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        scores = outputs.logits.softmax(dim=1).detach().numpy()[0]
        labels = ["negative", "neutral", "positive"]
        label = labels[scores.argmax()]
        return label
    except Exception as e:
        logger.error(f"Error in analyze_sentiment: {str(e)}")
        return f"An error occurred during analyzing sentiment: {str(e)}"

def get_reddit_posts(subreddit_name, stock_symbol, limit=100, days=30):
    """
    Get posts from a specific subreddit containing the stock symbol within the last specified days.
    """
    try:
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT")
        )
        subreddit = reddit.subreddit(subreddit_name)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)
    
        posts = []
        for post in subreddit.search(stock_symbol, sort='new', time_filter='month', limit=limit):
            post_date = datetime.utcfromtimestamp(post.created_utc)
            if start_date <= post_date <= end_date:
                posts.append(post.title)
        return posts
    except Exception as e:
        logger.error(f"Error in get_reddit_posts: {str(e)}")
        return f"An error occurred during get_reddit_posts: {str(e)}"

@tool
def reddit_sentiment_analysis(stock_symbol: str, subreddits: list = ['wallstreetbets', 'stocks', 'investing'], limit: int = 100):
    """
    Perform sentiment analysis on posts from specified subreddits about a stock symbol.
    
    Args:
        stock_symbol (str): The stock symbol to search for.
        subreddits (list): List of subreddits to search in.
        limit (int): Number of posts to fetch from each subreddit.
    
    Returns:
        list: List of sentiment labels for each post.
    """
    try:
        all_sentiments = []
        sentiments_counts={'neutral': 0, 'negative': 0, 'positive': 0}
    
        for subreddit in subreddits:
            # call the API to get the posts
            posts = get_reddit_posts(subreddit, stock_symbol, limit)
            # Log token usage
            logger.info(f"Reddit API Token Used: {os.getenv('REDDIT_CLIENT_ID')[:5]}...")
            logger.info(f"Reddit Client Secret: {os.getenv('REDDIT_CLIENT_SECRET')[:5]}...")
            logger.info(f"Reddit User Agent: {os.getenv('REDDIT_USER_AGENT')[:5]}...")
            for post in posts:
                sentiment = analyze_sentiment(post)
                all_sentiments.append((sentiment))
                sentiments_counts[sentiment]+=1
               

        return sentiments_counts
    except Exception as e:
        logger.error(f"Error in reddit_sentiment_analysis: {str(e)}")
        return f"An error occurred during sentiment analysis: {str(e)}"