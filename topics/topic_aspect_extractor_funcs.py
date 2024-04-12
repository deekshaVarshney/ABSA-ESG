import json
import pandas as pd
from bertopic import BERTopic
from tqdm import tqdm
import torch
import csv
import os
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer


# read the manual from: https://maartengr.github.io/BERTopic/index.html#fine-tune-topic-representations

def load_sentences_from_data(data_path):
        
    # Load JSON data into a DataFrame

    data = []
    with open(data_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    df = pd.DataFrame(data)

    # Separate sentences based on their classification
    risk_sentences = df[df['label'] == 'risk']['sentence'].tolist()
    opportunity_sentences = df[df['label'] == 'opportunity']['sentence'].tolist()
    return risk_sentences, opportunity_sentences


def load_document_data(directory1, directory2):
    """Assume that we have two directories by default"""
    # Load text data into a list
    data = []
    for directory in [directory1, directory2]:
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                with open(os.path.join(directory, filename), 'r') as file:
                    data.append(file.read())
    return data

# Function to apply BERTopic and get hierarchical topics
def extract_topics(sentences_documents, remove_stopwords=False, use_KeyBERT=False, use_sentence_transformer=False, umap_params=None, hdbscan_params=None, nr_topics=None):
    
    ### initialize the model
    if remove_stopwords:
        vectorizer_model = CountVectorizer(stop_words="english")
        topic_model = BERTopic(vectorizer_model=vectorizer_model)
    elif use_KeyBERT:
        representation_model = KeyBERTInspired()
        topic_model = BERTopic(representation_model=representation_model)
    else:
        topic_model = BERTopic() # can specify the number of topics/ parameters

    if umap_params:
        # should be a dictionary in the format of 
        # umap_params = {"n_neighbors": 15, "n_components": 5, "metric": "cosine"}
        topic_model.umap_model = umap_params
    
    if hdbscan_params:
        # should be a dictionary in the format of 
        # hdbscan_params = {"min_cluster_size": 15, "min_samples": 15}
        topic_model.hdbscan_model = hdbscan_params

    if nr_topics:
        topic_model.nr_topics = nr_topics

    
    if use_sentence_transformer:
        sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = sentence_transformer_model.encode(sentences_documents, show_progress_bar=True)
        topic_model.embedding_model = sentence_transformer_model
        topics, probabilities = topic_model.fit_transform(sentences_documents, embeddings)
    else:
        topics, probabilities = topic_model.fit_transform(tqdm(sentences_documents))
        embeddings = None
    topic_info = topic_model.get_topic_info()  # This gives you the hierarchy and topic sizes
    print(topic_model)
    print(len(topic_info))
    # hierarchical topics
    hierarchical_topics = topic_model.hierarchical_topics(sentences_documents)
    
    return topic_model, topic_info, hierarchical_topics, embeddings, topics
