import pandas as pd
import json
import re
from collections import defaultdict
from nltk.corpus import stopwords
import random 

# Function to preprocess text
def preprocess_text(tweet):
    # tweet = tweet.replace('\n', ' ')
    # Remove hashtags at the end of the tweet
    tweet_without_hashtags = re.sub(r'\s*#(\w+)\s*$', r'\1', tweet)
    # Remove remaining hashtags
    tweet_without_hashtags = re.sub(r'#\w+', '', tweet_without_hashtags)
    return tweet_without_hashtags.strip()


# Function to preprocess sentences
def preprocess_sentence(sentence):
    # Remove URLs
    sentence_without_urls = re.sub(r'http\S+|https\S+', '', sentence)
    return sentence_without_urls.strip()

# Load concept words and descriptive words from JSON file
concept_words = defaultdict(list)
with open('/Data/deeksha/concept_parser/concepts/concept_words.jsonl', 'r') as file:
    for line in file:
        concept = json.loads(line)
        for word in concept['descriptive_words']:
            concept_words[concept['concept']].append(word)

# Remove stop words from descriptive words
stop_words = set(stopwords.words('english'))
for concept, words in concept_words.items():
    concept_words[concept] = [word for word in words if word.lower() not in stop_words]

# Curate a list of all descriptive words based on frequency
all_descriptive_words = []
word_frequency = defaultdict(int)
for words in concept_words.values():
    for word in words:
        word_frequency[word] += 1

# Filter descriptive words based on frequency
for word, frequency in word_frequency.items():
    if frequency >= 100:  # Adjust the threshold as needed
        all_descriptive_words.append(word)

# Function to filter tweets based on descriptive words
def filter_tweets(tweets, descriptive_words):
    filtered_tweets = set()
    for tweet in tweets:
        # Preprocess tweet
        # print('tweets',tweet)
        if not re.findall(r'http\S+|https\S+', tweet):  # Check if the tweet contains URLs  
            preprocessed_tweet = preprocess_text(tweet)
            # print('\n',preprocessed_tweet)
            # Check if at least 3 descriptive words are present in the tweet
            found_words = [word for word in descriptive_words if word in preprocessed_tweet.split()]
            count = len(found_words)
            if count >= 4:
                filtered_tweets.add(preprocessed_tweet)
    return filtered_tweets

# Function to filter news sentences based on concept words
def filter_news_sentences(sentences, descriptive_words):
    filtered_sentences = set()
    for sentence in sentences:
        preprocessed_sentence = preprocess_sentence(str(sentence))  # Ensure the sentence is converted to string
        found_words = [word for word in descriptive_words if word in preprocessed_sentence.split()]
        count = len(found_words)
        if count >= 4:
            filtered_sentences.add(preprocessed_sentence)
    return filtered_sentences

# Read tweets from CSV file
tweets_df = pd.read_csv('/Data/deeksha/concept_parser/data/tweets/tweets.csv')
news_df = pd.read_csv('/Data/deeksha/concept_parser/data/news/sentences_all_articles.csv')

# Filter tweets based on descriptive words and remove tweets containing URLs
filtered_tweets = filter_tweets(tweets_df['Tweet'], all_descriptive_words)

filtered_tweets_list = list(filtered_tweets)
random.shuffle(filtered_tweets_list)

# Create a new DataFrame with the filtered tweets
filtered_tweets_df = pd.DataFrame({'samples': filtered_tweets_list})

# Save the filtered tweets to a new CSV file
filtered_tweets_df.to_csv('/Data/deeksha/concept_parser/data/tweets/filtered_tweets.csv', index=False)

# Read news sentences from CSV file

filtered_news_sentences = filter_news_sentences(news_df['Sentence'], all_descriptive_words)

filtered_news_list = list(filtered_news_sentences)
random.shuffle(filtered_news_list)

# Create a new DataFrame with the filtered news sentences
filtered_news_df = pd.DataFrame({'samples': filtered_news_list})

# Save the filtered news sentences to a new CSV file
filtered_news_df.to_csv('/Data/deeksha/concept_parser/data/news/filtered_sentences.csv', index=False)
