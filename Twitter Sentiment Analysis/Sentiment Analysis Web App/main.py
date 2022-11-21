import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
#from transformers import pipeline
from functions import *
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import operator
from textblob import TextBlob
import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import pycountry
import re
import string
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="twitter.png",
    layout="centered",
    initial_sidebar_state="auto",
)
st.title("Twitter Sentiment Analysis")

st.markdown("""
 <style>
 .big-font {
     font-size:30px !important;
 }
 </style>
 """, unsafe_allow_html=True)

#emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

keyword = st.text_input('Hashtag')
noOfTweet = st.text_input('Number Of Tweets')

#Authentication
consumer_key = 'ZMMjkS37qTi9vuXsetoKbOGfP'
consumer_secret = 'eWdB2hAqm3hdkL3FhbKALAbtAViAEGAt7PIrIDXYGTBJcn44h8'
access_token = '1588836299200512000-spyB54fqpBesHUwDHOwRQWekIc33Se'
access_token_secret = '4OrK6v0GdVmzuyOWIlCN4FHbbfgRoU3E5soH9xIXhZpmO'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

nltk.download('vader_lexicon')

if st.button("Sentiment Analysis") and (keyword or len(keyword) != 0) and (str(keyword[0]) == '#') and (int(noOfTweet) > 0):
    with st.spinner(f"In my feelings..."):

        tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang="en", wait_on_rate_limit=True).items(int(noOfTweet))
        positive = 0
        negative = 0
        neutral = 0
        polarity = 0
        tweet_list = []
        neutral_list = []
        negative_list = []
        positive_list = []

        for tweet in tweets:
            tweet_list.append(tweet.text)
            analysis = TextBlob(tweet.text)
            score = SentimentIntensityAnalyzer().polarity_scores(tweet.text)
            neg = score['neg']
            neu = score['neu']
            pos = score['pos']
            comp = score['compound']
            polarity += analysis.sentiment.polarity

            if neg > pos:
                negative_list.append(tweet.text)
                negative += 1
            elif pos > neg:
                positive_list.append(tweet.text)
                positive += 1
            elif pos == neg:
                neutral_list.append(tweet.text)
                neutral += 1

        positive = percentage(positive, noOfTweet)
        negative = percentage(negative, noOfTweet)
        neutral = percentage(neutral, noOfTweet)
        polarity = percentage(polarity, noOfTweet)
        positive = format(positive, '.1f')
        negative = format(negative, '.1f')
        neutral = format(neutral, '.1f')

        tweet_list = pd.DataFrame(tweet_list)
        neutral_list = pd.DataFrame(neutral_list)
        negative_list = pd.DataFrame(negative_list)
        positive_list = pd.DataFrame(positive_list)

        labels = ['Positive [' + str(positive) + '%]', 'Neutral [' + str(neutral) + '%]',
                  'Negative [' + str(negative) + '%]']
        sizes = [positive, neutral, negative]
        colors = ['green', 'pink', 'red']

        patches, texts = plt.pie(sizes, colors=colors, startangle=90)
        plt.style.use('default')
        plt.legend(labels)
        plt.title('Sentiment Analysis Result for keyword= ' + keyword + '')
        plt.axis('equal')
        st.pyplot(plt)

        tweet_list.drop_duplicates(inplace=True)
        df = pd.DataFrame(tweet_list)
        df["text"] = df[0]

        remove_rt = lambda x: re.sub("RT @\w+: ", " ", x)
        rt = lambda x: re.sub("(@[A-Za-z0–9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", x)
        df["text"] = df.text.map(remove_rt).map(rt)
        df["text"] = df.text.str.lower()
        df[["polarity", "subjectivity"]] = df["text"].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))

        for index, row in df["text"].iteritems():
            score = SentimentIntensityAnalyzer().polarity_scores(row)
            neg = score["neg"]
            neu = score["neu"]
            pos = score["pos"]
            comp = score["compound"]
            if neg > pos:
                df.loc[index, "sentiment"] = "negative"
            elif pos > neg:
                df.loc[index, "sentiment"] = "positive"
            else:
                df.loc[index, "sentiment"] = "neutral"
                df.loc[index, "neg"] = neg
                df.loc[index, "neu"] = neu
                df.loc[index, "pos"] = pos
                df.loc[index, "compound"] = comp

        df_negative = df[df["sentiment"] == "negative"]
        df_positive = df[df["sentiment"] == "positive"]
        df_neutral = df[df["sentiment"] == "neutral"]

        pc = count_values_in_column(df, "sentiment")

        # WordCloud For The Whole Dataset
        df['clean_text'] = df['text'].apply(lambda x: remove_punctuation(x))
        stop_words = set(STOPWORDS)
        wordcloud = WordCloud(background_color='black', stopwords=stop_words, max_words=500, max_font_size=100,
                              random_state=42, width=800, height=400)

        wordcloud.generate(str(df['clean_text']))
        plt.figure(figsize=(12.0, 8.0))
        plt.imshow(wordcloud)
        plt.title(f"Word Cloud", fontdict={'size': 40, 'color': 'black', 'verticalalignment': 'bottom'})
        plt.axis('off')
        plt.tight_layout()
        st.pyplot(plt)

        # Wordcloud for the positive dataset
        df_positive['clean_text'] = df_positive['text'].apply(lambda x: remove_punctuation(x))
        stop_words = set(STOPWORDS)
        wordcloud = WordCloud(background_color='black', stopwords=stop_words, max_words=500, max_font_size=100,
                              random_state=42, width=800, height=400)
        wordcloud.generate(str(df_positive['clean_text']))
        plt.figure(figsize=(12.0, 8.0))
        plt.imshow(wordcloud)
        plt.title(f"Word Cloud for positive tweets",
                  fontdict={'size': 40, 'color': 'black', 'verticalalignment': 'bottom'})
        plt.axis('off')
        plt.tight_layout()
        st.pyplot(plt)

        # Wordcloud for the neutral dataset
        df_neutral['clean_text'] = df_neutral['text'].apply(lambda x: remove_punctuation(x))
        stop_words = set(STOPWORDS)
        wordcloud = WordCloud(background_color='black', stopwords=stop_words, max_words=500, max_font_size=100,
                              random_state=42, width=800, height=400)
        wordcloud.generate(str(df_neutral['clean_text']))
        plt.figure(figsize=(12.0, 8.0))
        plt.imshow(wordcloud)
        plt.title(f"Word Cloud for neutral tweets",
                  fontdict={'size': 40, 'color': 'black', 'verticalalignment': 'bottom'})
        plt.axis('off')
        plt.tight_layout()
        st.pyplot(plt)

        # Wordcloud for the negative dataset
        df_negative['clean_text'] = df_negative['text'].apply(lambda x: remove_punctuation(x))
        stop_words = set(STOPWORDS)
        wordcloud = WordCloud(background_color='black', stopwords=stop_words, max_words=500, max_font_size=100,
                              random_state=42, width=800, height=400)
        wordcloud.generate(str(df_negative['clean_text']))
        plt.figure(figsize=(12.0, 8.0))
        plt.imshow(wordcloud);
        plt.title(f"Word Cloud for negative tweets",
                  fontdict={'size': 40, 'color': 'black', 'verticalalignment': 'bottom'})
        plt.axis('off')
        plt.tight_layout()
        st.pyplot(plt)






else:
    st.warning("Invalid hashtag or length")

if __name__ == '__main__':
    print("Works")

