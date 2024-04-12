import os
import pandas as pd
from topic_aspect_extractor_funcs import *

def process_csv_files(csv_files, log_path="."):
    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")
        
        # Read the CSV file
        df = pd.read_csv(csv_file)
        samples = df['samples'].tolist()  # or df['sentence'] if applicable
        
        # Extract topics for the samples
        topic_model, topic_info, hierarchical_topics, embeddings, topics = extract_topics(samples, use_sentence_transformer=True, use_KeyBERT=True)
        
        # Create a DataFrame with sentences and their corresponding topic IDs
        sentences_with_topics_df = pd.DataFrame({
            'sentence': samples,
            'topic_id': topics
        })
        
        # Save sentences with topics to CSV
        basename1 = str(csv_file.split('.')[0]) + '_sentences_with_topics'
        sentences_with_topics_save_path = os.path.join(log_path, f'{basename1}.csv')
        sentences_with_topics_df.to_csv(sentences_with_topics_save_path, index=False)
        print(f"Sentences with topics saved to: {sentences_with_topics_save_path}")
        
        # Save topics to CSV
        basename2 = str(csv_file.split('.')[0]) + '_topics'

        topics_save_path = os.path.join(log_path, f'{basename2}.csv')
        topic_info.to_csv(topics_save_path, index=False)
        print(f"Topics saved to: {topics_save_path}")
        
        # Visualize hierarchical topics
        fig_hierarchy = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
        basename3 = str(csv_file.split('.')[0]) + '_hierarchical_topics'
        fig_hierarchy_write_path = os.path.join(log_path, f'{basename3}.html')
        fig_hierarchy.write_html(fig_hierarchy_write_path)
        print(f"Hierarchical topics visualization saved to: {fig_hierarchy_write_path}")

# List of CSV files to process
csv_files = ['/Data/deeksha/concept_parser/data/tweets/filtered_tweets.csv', '/Data/deeksha/concept_parser/data/news/filtered_sentences.csv']

# Process CSV files
process_csv_files(csv_files)
