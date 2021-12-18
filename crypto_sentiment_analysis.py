import json
import time
from tqdm import tqdm
from statistics import mean
import re

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import pandas as pd
import numpy as np

import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

import plotly.express as px

# download lexicons
nltk.download('vader_lexicon')

# setup  and initialize language detector
def get_lang_detector(nlp, name):
    return LanguageDetector()

nlp = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe('language_detector', last=True)

# read telegram message json dump
def read_chat_json(filename):
    data = {}
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data

# remove non-ascii and non-alphabet characters 
# such as smilies and punctuations
def filter_non_alphabet_characters(string):
    encoded_string = string.encode("ascii", "ignore")
    decoded_string = encoded_string.decode().strip()
    regex = re.compile('[^a-zA-Z0-9]')
    #First parameter is the replacement, second parameter is your input string
    return regex.sub(' ', decoded_string)
    

# check if SHIB or DOGE word is present in message ignoring case
def is_shiba_doge(message):
    return message['text'] \
        and (' shib ' in message['text'] 
            or ' doge ' in message['text'])

# filter non-ascii characters and convert to lowercase
def preprocess_chat_message(message):
    if isinstance(message['text'], list):
        processed_str = ''
        for obj in message['text']:
            if isinstance(obj, str):
                processed_str += obj
        message['text'] = processed_str
    message['text'] = filter_non_alphabet_characters(message['text'])
    message['text'] = message['text'].lower().strip()
    return message

# filter messages containing shiba/doge and in english language
def preprocess_chat(messages):
    processed_messages = []
    for message in tqdm(messages):
        processed_message = preprocess_chat_message(message)
        if is_shiba_doge(message) \
            and nlp(message['text'])._.language['language'] == 'en':
                processed_messages.append(processed_message)
    return processed_messages

def compute_sentiment_scores(messages, sentimentAnalyzer):
    for message in messages:
        ss = sentimentAnalyzer.polarity_scores(message['text'])
        message['sentiment_score'] = ss['compound']

def plot_sentiments(date_grouping):
    print('Generating sentiment plot')
    dates = sorted([date for date in date_grouping])
    avg_sentiments = []
    for date in dates:
        avg_sentiments.append(date_grouping[date]['sentiment_avg'])

    df = pd.DataFrame(dict(
        date = np.array(dates, dtype='datetime64'),
        sentiment = avg_sentiments
    ))

    fig = px.line(df, x='date', y='sentiment', markers=True)
    fig.show()
    #fig.write_image('sentiment_plot.png', format='png', engine='kaleido')

def plot_message_counts(date_grouping):
    print('Generating message count plot')
    dates = sorted([date for date in date_grouping])
    counts = []
    for date in dates:
        counts.append(date_grouping[date]['count'])


    df = pd.DataFrame(dict(
        date = np.array(dates, dtype='datetime64'),
        count = counts
    ))

    fig = px.line(df, x='date', y='count', markers=True)
    fig.show()
    #fig.write_image('msg_count_plot.png', format='png', engine='kaleido')


if __name__ == '__main__':
    data = read_chat_json(r'data\crypto_chat.json')
    messages = data['messages']
    print('Total number of messages read: {}'.format(len(messages)))

    print('Pre-processing messages')
    messages = preprocess_chat(messages)
    print('Total number of messages after pre-processing: {}'
           .format(len(messages)))

    sid = SentimentIntensityAnalyzer()
    compute_sentiment_scores(messages, sid)

    date_grouping = {}

    for msg in messages:
        date = msg['date'].split('T')[0]
        if date not in date_grouping:
            date_grouping[date] = {'sentiment': [], 'count': 0}

        date_grouping[date]['count'] += 1
        date_grouping[date]['sentiment'].append(msg['sentiment_score'])

    for date in date_grouping:
        date_grouping[date]['sentiment_avg'] \
            = mean(date_grouping[date]['sentiment'])

    # plot results
    plot_message_counts(date_grouping)
    plot_sentiments(date_grouping)
