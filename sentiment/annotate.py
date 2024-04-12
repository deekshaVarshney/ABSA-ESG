from sgnlp.models.sentic_gcn import (
    SenticGCNBertTokenizer,
    SenticGCNBertEmbeddingConfig,
    SenticGCNBertEmbeddingModel,
    SenticGCNBertModel,
    SenticGCNBertPreprocessor,
    SenticGCNBertConfig,
    SenticGCNBertPostprocessor,
)
import pandas as pd
import re

def clean_sentence(sentence):
    # Remove special characters
    cleaned_sentence = re.sub(r'[^\w\s]', '', sentence)
    # Convert to lowercase
    cleaned_sentence = cleaned_sentence.lower()
    return cleaned_sentence

def sentiment_annotation(csv_files):
    # Create tokenizer
    tokenizer = SenticGCNBertTokenizer.from_pretrained("bert-base-uncased")

    # Create embedding model
    embed_config = SenticGCNBertEmbeddingConfig.from_pretrained("bert-base-uncased")
    embed_model = SenticGCNBertEmbeddingModel.from_pretrained("bert-base-uncased", config=embed_config)

    # Create preprocessor
    preprocessor = SenticGCNBertPreprocessor(
        tokenizer=tokenizer,
        embedding_model=embed_model,
        senticnet="https://storage.googleapis.com/sgnlp-models/models/sentic_gcn/senticnet.pickle",
        device="cpu",
    )

    # Create postprocessor
    postprocessor = SenticGCNBertPostprocessor()

    # Load model
    config = SenticGCNBertConfig.from_pretrained(
        "https://storage.googleapis.com/sgnlp-models/models/sentic_gcn/senticgcn_bert/config.json"
    )

    model = SenticGCNBertModel.from_pretrained(
        "https://storage.googleapis.com/sgnlp-models/models/sentic_gcn/senticgcn_bert/pytorch_model.bin", config=config
    )

    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")
        tweets_df = pd.read_csv(csv_file)
        tweets_df = tweets_df.head(50)
        processed_inputs = []

        # Iterate through each tweet
        for index, row in tweets_df.iterrows():
            aspect_terms = row["Aspect Terms"]
            if isinstance(aspect_terms, str):
                aspects = [aspect.strip() for aspect in aspect_terms.split(",")]
            else:
                aspects = []

            sentence = clean_sentence(row["samples"])  # Clean the sentence
            input_data = {
                "aspects": aspects,
                "sentence": sentence
            }
            try:
                processed_input, processed_index = preprocessor([input_data])
                raw_output = model(processed_index)
                post_output = postprocessor(processed_inputs=processed_input, model_outputs=raw_output)
                
                # Convert indices back to words
                aspect_words = []
                for aspect_index_list in post_output[0]['aspects']:
                    aspect_words.append([post_output[0]['sentence'][index] for index in aspect_index_list])
                
                processed_inputs.append({
                    'sentence': post_output[0]['sentence'],
                    'aspects': aspect_words,
                    'labels': post_output[0]['labels']
                })
                
                print(index, post_output[0], input_data)
            except Exception as e:
                print(f"Error processing row {index} in file {csv_file}: {e}")
                processed_inputs.append([])  # Append empty list for sentiment annotations

        # Add sentiment annotations to the DataFrame
        sentiment_labels = []
        aspect_words_list = []

        # Add sentiment annotations and aspect words to the DataFrame
        for output in processed_inputs:
            if output:
                sentiment_labels.append(output["labels"])
                aspect_words_list.append(output["aspects"])
            else:
                sentiment_labels.append([])
                aspect_words_list.append([])

        tweets_df["sentiment_labels"] = sentiment_labels
        tweets_df["aspect_words"] = aspect_words_list
        # Save the annotated DataFrame back to CSV
        annotated_tweets_file = f"{csv_file.split('.')[0]}_sentiment.csv"
        tweets_df.to_csv(annotated_tweets_file, index=False)
        print(f"Annotated tweets saved to: {annotated_tweets_file}")

# List of CSV files to process
csv_files = [
    '/Data/deeksha/concept_parser/data/tweets/aspect_filtered_tweets.csv',
    '/Data/deeksha/concept_parser/data/news/aspect_filtered_sentences.csv'
]

# Perform sentiment annotation for each CSV file
sentiment_annotation(csv_files)
