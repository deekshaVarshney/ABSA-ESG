import json
import csv
import re
from datetime import datetime

# Function to convert date string to datetime object
def convert_to_datetime(date_str):
    formats = ["%a, %d %b %Y %H:%M:%S %Z", "%B %d, %Y", "%B %d, %Y", "%d %b %Y", "%A, %d %b %Y %H:%M:%S GMT"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Failed to parse date: {date_str}")

# Function to format date to a consistent format
def format_date(date_str):
    date_obj = convert_to_datetime(date_str)
    return date_obj.strftime("%Y-%m-%d %H:%M:%S")

# Function to clean and format text content
def clean_text(text, max_length=20):
    cleaned_text = text.replace("\n", " ")  # Remove newline characters
    return cleaned_text#[:max_length]  # Truncate text to max_length characters

# Function to clean and format URL
def clean_url(url):
    if not url.startswith("http"):
        return "Invalid URL"
    return url

# Function to extract sentences from text
def extract_sentences(text):
    # Define regex pattern to split text into sentences
    sentence_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"
    sentences = re.split(sentence_pattern, text)
    return sentences

# Read the JSONL file
def read_jsonl_file(file_path):
    articles = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            article = json.loads(line)
            articles.append(article)
    return articles

# Dump news articles in ascending order from latest to old to a CSV file
def dump_articles_to_csv(articles):
    sorted_articles = sorted(articles, key=lambda x: convert_to_datetime(x["date"]), reverse=True)
    with open("data/news/articles_sorted.csv", "w", newline='', encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Date", "Title", "Description", "URL", "Text"])
        writer.writeheader()
        for article in sorted_articles:
            consistent_date = format_date(article.get("date", ""))
            cleaned_text = clean_text(article.get("text", ""))
            cleaned_url = clean_url(article.get("url", ""))
            writer.writerow({"Date": consistent_date,
                             "Title": article.get("title", "").strip(),
                             "Description": article.get("description", "").strip(),
                             "URL": cleaned_url,
                             "Text": cleaned_text})

# Dump sentences from news articles to a CSV file
def dump_sentences_to_csv(articles):
    with open("data/news/sentences_all_articles.csv", "w", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Article Title", "Sentence"])
        for article in articles:
            title = article.get("title", "").strip()
            # print(title)
            text = clean_text(article.get("text", ""))
            sentences = extract_sentences(text)
            for sentence in sentences:
                # print(sentence)
                writer.writerow([title, sentence.strip()])

# File path to the JSONL file
file_path = "data/news/news.jsonl"

# Read the JSONL file
articles = read_jsonl_file(file_path)

# Dump news articles to a CSV file in ascending order from latest to old
dump_articles_to_csv(articles)

# Dump sentences from news articles to a CSV file
dump_sentences_to_csv(articles)
