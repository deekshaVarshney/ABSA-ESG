import pandas as pd
import re

def annotate_tweets_with_aspect_terms(csv_files, aspect_terms_files):
    # Iterate through each CSV file
    for csv_file in csv_files:
        print(f"Annotating tweets in file: {csv_file}")
        
        # Load aspect terms from the file corresponding to the current CSV file
        aspect_terms_file = aspect_terms_files.get(csv_file, '')
        if not aspect_terms_file:
            print(f"No aspect terms file specified for {csv_file}. Skipping...")
            continue
        
        with open(aspect_terms_file, 'r') as file:
            aspect_terms = [line.strip() for line in file.readlines()]
        
        # Load the filtered tweets from the CSV file
        filtered_tweets_df = pd.read_csv(csv_file)
        print(filtered_tweets_df.columns)
        filtered_tweets_df = filtered_tweets_df.head(50)
        # Create empty column for aspect terms
        filtered_tweets_df['Aspect Terms'] = ""
        
        # Iterate through each tweet
        for index, tweet in filtered_tweets_df.iterrows():
            found_terms = []
            # Check if any aspect term is found in the tweet
            for term in aspect_terms:
                # Construct a regular expression pattern to match whole words
                pattern = r'\b' + re.escape(term) + r'\b'
                if re.search(pattern, tweet['samples'], flags=re.IGNORECASE):
                    found_terms.append(term)
            # Assign the found aspect terms to the respective column
            filtered_tweets_df.at[index, 'Aspect Terms'] = ', '.join(found_terms)

        
        # Save the annotated tweets to a new CSV file
        annotated_tweets_file = f"{csv_file.split('.')[0]}_aspect.csv"
        filtered_tweets_df.to_csv(annotated_tweets_file, index=False)
        print(f"Annotated tweets saved to: {annotated_tweets_file}")

# List of CSV files to process
csv_files = ['/Data/deeksha/concept_parser/data/tweets/filtered_tweets.csv', '/Data/deeksha/concept_parser/data/news/filtered_sentences.csv']

# Dictionary specifying aspect terms files for each CSV file
aspect_terms_files = {
    '/Data/deeksha/concept_parser/data/tweets/filtered_tweets.csv': '/Data/deeksha/concept_parser/data/tweets/aspect_terms_tweets.txt',
    '/Data/deeksha/concept_parser/data/news/filtered_sentences.csv': '/Data/deeksha/concept_parser/data/news/aspect_terms_news.txt'
}

# Annotate tweets with aspect terms for each CSV file
annotate_tweets_with_aspect_terms(csv_files, aspect_terms_files)
