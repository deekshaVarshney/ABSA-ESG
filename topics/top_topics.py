import pandas as pd
from collections import Counter

def extract_unique_terms_with_frequency(csv_files):
    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")
        df = pd.read_csv(csv_file)
        representation_column = df['Representation']
        flat_representation = [term for sublist in representation_column for term in eval(sublist)]
        term_freq = Counter(flat_representation)
        
        # Print the top 50 terms
        print("Top 50 terms:")
        for term, freq in term_freq.most_common(50):
            print(f"{term}: {freq}")
        
        # Extract unique terms
        unique_terms = [term for term, freq in term_freq.most_common(50)]
        
        # Determine the aspect terms file path based on the CSV file type (tweets or news)
        aspect_terms_file = '/Data/deeksha/concept_parser/data/tweets/aspect_terms_tweets.txt' if 'tweets' in csv_file else '/Data/deeksha/concept_parser/data/news/aspect_terms_news.txt'
        
        # Load existing aspect terms from the file
        existing_terms = set()
        try:
            with open(aspect_terms_file, 'r') as file:
                for line in file:
                    existing_terms.add(line.strip())
        except FileNotFoundError:
            pass
        
        # Append only the new terms to the aspect terms file
        new_terms = [term for term in unique_terms if term not in existing_terms]
        with open(aspect_terms_file, 'a') as file:
            for term in new_terms:
                file.write(f"{term}\n")     

        # Extract unique terms
        unique_terms = set(flat_representation)
        path = str(csv_file.split('.')[0]) + '_unique_terms_with_frequency'
        # Write unique terms with frequency to a text file
        with open(f'{path}.txt', 'w') as file:
            file.write("Term\tFrequency\n")
            for term, freq in term_freq.items():
                file.write(f"{term}\t{freq}\n")
        print(f"Unique terms with frequency saved to: unique_terms_with_frequency_{csv_file}.txt")

# List of CSV files to process
csv_files = ['/Data/deeksha/concept_parser/data/tweets/filtered_tweets_topics.csv', '/Data/deeksha/concept_parser/data/news/filtered_sentences_topics.csv']

# Process CSV files and extract unique terms with frequency
extract_unique_terms_with_frequency(csv_files)
