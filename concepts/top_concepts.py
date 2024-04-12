import pandas as pd
from collections import Counter
import itertools
import ast

def extract_concept_frequencies(csv_files):
    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")
        df = pd.read_csv(csv_file)
        flat_concepts = [item_sub for sublist in df['sentence_concepts'] for subsublist in ast.literal_eval(sublist) for item in subsublist for item_sub in item]
        concept_freq = Counter(flat_concepts)
        
        # Print the top 50 concepts
        print("Top 50 concepts:")
        for concept, freq in concept_freq.most_common(50):
            print(f"{concept}: {freq}")
        
        top_concepts = []
        print("Top 50 concepts:")
        for concept, freq in concept_freq.most_common(50):
            print(f"{concept}: {freq}")
            top_concepts.append(concept)
            
        if 'tweets' in csv_file:
            aspect_terms_file = '/Data/deeksha/concept_parser/data/tweets/aspect_terms_tweets.txt'
        elif 'news' in csv_file:
            aspect_terms_file = '/Data/deeksha/concept_parser/data/news/aspect_terms_news.txt'
        else:
            aspect_terms_file = ''
        
        if aspect_terms_file:
            with open(aspect_terms_file, 'w') as file:
                file.write('\n'.join(top_concepts))
            print(f"Top 50 concepts saved to: {aspect_terms_file}")

        # Dump concept frequencies to a CSV file
        path = str(csv_file.split('.')[0]) + '_concept_frequencies'
        concept_freq_df = pd.DataFrame(concept_freq.items(), columns=['Concept', 'Frequency'])
        concept_freq_df.to_csv(f'{path}.csv', index=False)
        print(f"Concept frequencies saved to: {path}.csv")

# List of CSV files to process
csv_files = ['/Data/deeksha/concept_parser/data/tweets/filtered_tweets_with_concepts.csv', '/Data/deeksha/concept_parser/data/news/filtered_sentences_with_concepts.csv']

# Process CSV files and extract concept frequencies
extract_concept_frequencies(csv_files)
