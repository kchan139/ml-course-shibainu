import os
import re
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Any, Union

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


#========== DATA UTILS ==========#

def preprocess_text(text: Any) -> List[str]:
    """
    Preprocess text by converting to lowercase, removing punctuation,
    tokenizing, and removing stop words.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        List of processed word tokens
    """
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return tokens


def get_frequent_words(text_series: Union[pd.Series, List[str]], top_n: int = 30) -> List[Tuple[str, int]]:
    """
    Extract frequent words from a series of text.
    
    Args:
        text_series: Series of text items to analyze
        top_n: Number of top frequent words to return
        
    Returns:
        List of tuples containing (word, frequency) sorted by frequency
    """
    all_words = []
    for text in text_series:
        all_words.extend(preprocess_text(text))
    return Counter(all_words).most_common(top_n)


def generate_wordcloud(text_series: Union[pd.Series, List[str]], title: str) -> None:
    """
    Generate and display a word cloud from a series of text.
    
    Args:
        text_series: Series of text items to visualize
        title: Title for the word cloud plot
    """
    all_text = ' '.join([' '.join(preprocess_text(text)) for text in text_series])
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def get_ngrams(text_series: Union[pd.Series, List[str]], n: int = 2, top_n: int = 30) -> List[Tuple[Tuple[str, ...], int]]:
    """
    Extract n-grams from a series of text.
    
    Args:
        text_series: Series of text items to analyze
        n: Size of n-grams (2 for bigrams, 3 for trigrams, etc.)
        top_n: Number of top frequent n-grams to return
        
    Returns:
        List of tuples containing (n-gram, frequency) sorted by frequency
    """
    all_ngrams = []
    for text in text_series:
        tokens = preprocess_text(text)
        text_ngrams = list(ngrams(tokens, n))
        all_ngrams.extend(text_ngrams)
    return Counter(all_ngrams).most_common(top_n)


def contains_html_tags(text: Any) -> bool:
    """
    Check if text contains HTML tags.
    
    Args:
        text: Text to check for HTML tags
        
    Returns:
        Boolean indicating if HTML tags were found
    """
    html_pattern = re.compile(r'<.*?>')
    return bool(html_pattern.search(str(text)))


def clean_html(text: Any) -> str:
    """
    Clean HTML tags from text using BeautifulSoup and regex.
    
    Args:
        text: Text containing HTML to clean
        
    Returns:
        Cleaned text with HTML tags removed
    """
    soup = BeautifulSoup(str(text), 'html.parser')
    clean_text = soup.get_text(separator=' ', strip=True)
    
    # Additional cleanup for any remaining tags
    clean_text = re.sub(r'<.*?>', ' ', clean_text)
    
    # Remove extra whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text


def extract_special_chars(text: Any) -> str:
    """
    Extract special characters from text.
    
    Args:
        text: Text to extract special characters from
        
    Returns:
        String containing all special characters found
    """
    special_chars = re.findall(r'[^\w\s]', str(text))
    return ''.join(special_chars)


def remove_columns(input_file="dataset/raw/Reviews.csv", 
                   output_file="dataset/processed/Reviews_removed_columns.csv"):
    """
    Remove specified columns from the dataset and save the processed dataset.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the processed CSV file
    """
    df = pd.read_csv(input_file)
    
    columns_to_remove = ['ProductId', 'UserId', 'ProfileName', 
                         'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time']
    
    df_processed = df.drop(columns=columns_to_remove, errors='ignore')
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df_processed.to_csv(output_file, index=False)
    print(f"Processed dataset saved to {output_file}")


#========== TEST UTILS ==========#

def process_html_tags(input_file, output_file):
    """
    Process HTML tags in the dataset and add a column with cleaned text.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the processed CSV file
    """
    df = pd.read_csv(input_file)
    
    df['Text_HTML_Clean'] = df['Text'].apply(clean_html)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df.to_csv(output_file, index=False)
    print(f"Dataset with HTML cleaned text saved to {output_file}")


def normalize_text_length(input_file, output_file, p95_threshold=None):
    """
    Normalize text length in dataset using truncation and add length features.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the processed CSV file
        p95_threshold: Threshold for truncation (default uses 95th percentile)
    """
    df = pd.read_csv(input_file)
    df['Text_Length'] = df['Text_HTML_Clean'].apply(len)
    
    if p95_threshold is None:
        p95_threshold = int(df['Text_Length'].quantile(0.95))
    
    df['Text_Normalized'] = df['Text_HTML_Clean'].apply(
        lambda x: x[:p95_threshold] if len(x) > p95_threshold else x
    )
    
    df['Length_Feature'] = np.log1p(df['Text_Length'])
    df = categorize_text_length_df(df)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df.to_csv(output_file, index=False)
    print(f"Dataset with normalized text lengths saved to {output_file}")


def categorize_text_length_df(df):
    """
    Add length category column to a dataframe.
    
    Args:
        df: DataFrame with Text_HTML_Clean column
        
    Returns:
        DataFrame with Length_Category column added
    """
    def get_length_category(length):
        if length <= 100:
            return 'Very Short'
        elif length <= 250:
            return 'Short'
        elif length <= 500:
            return 'Medium-Short'
        elif length <= 750:
            return 'Medium'
        elif length <= 1000:
            return 'Medium-Long'
        elif length <= 1500:
            return 'Long'
        else:
            return 'Very Long'
    
    if 'Text_Length' not in df.columns:
        df['Text_Length'] = df['Text_HTML_Clean'].apply(len)
    
    df['Length_Category'] = df['Text_Length'].apply(get_length_category)
    return df


def categorize_text_length(input_file, output_file):
    """
    Categorize text by length and save as a new dataset.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the processed CSV file
    """
    df = pd.read_csv(input_file)
    
    df = categorize_text_length_df(df)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df.to_csv(output_file, index=False)
    print(f"Dataset with text length categories saved to {output_file}")


if __name__ == "__main__":
    remove_columns()