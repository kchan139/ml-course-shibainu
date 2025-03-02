# src/data/preprocess.py

import re
import string
from pathlib import Path
import pandas as pd
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH  # Import paths from config.py

class DataPreprocessor:
    """
    This class preprocesses the 'all-data.csv' dataset by performing text cleaning using NLTK,
    encoding sentiment labels, splitting the data into training, validation, and test sets,
    and vectorizing the cleaned text using TF-IDF.
    """

    def __init__(self, file_path: str):
        """
        Initializes the DataPreprocessor with the CSV file path.
        
        Args:
            file_path: Path to the 'all-data.csv' file.
        """
        self.file_path = file_path
        self.df = pd.read_csv(file_path, encoding="UTF-8")
        # Remove extra spaces from column names for consistency
        self.df.rename(columns=lambda x: x.strip(), inplace=True)
        self.label_encoder = LabelEncoder()
        self.vectorizer = None  # Will be initialized during vectorization

        # Initialize NLTK tools
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        # Placeholders for split data
        self.X_train = None
        # self.X_val = None
        self.X_test = None
        self.y_train = None
        # self.y_val = None
        self.y_test = None

    def clean_text(self, text: str) -> str:
        """
        Cleans the input text by:
         - Converting to lowercase.
         - Removing numbers.
         - Removing punctuation.
         - Tokenizing using NLTK.
         - Removing stopwords.
         - Lemmatizing tokens.
         
        Args:
            text: Raw text from the dataset.
            
        Returns:
            A cleaned version of the text.
        """
        text = text.lower()  # Lowercase the text
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
        
        # Tokenize using NLTK's word_tokenize
        try:
            tokens = word_tokenize(text)
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('stopwords')
            tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize each token
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return " ".join(tokens)

    def clean_data(self):
        """
        Cleans the dataset by applying text cleaning to the 'News Headline' column,
        and encodes the sentiment labels. The cleaned text is stored in a new column 'clean_text',
        and the encoded sentiment in 'sentiment_encoded'.
        """
        self.df["clean_text"] = self.df["News Headline"].astype(str).apply(self.clean_text)
        self.df["sentiment_encoded"] = self.label_encoder.fit_transform(self.df["Sentiment"])

    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        """
        Splits the dataset into training, validation, and test sets.
        
        Args:
            test_size: Fraction of the data to reserve for the test set.
            val_size: Fraction of the remaining data to reserve for the validation set.
            random_state: Seed used for reproducibility.
        """
        # Ensure data is cleaned before splitting
        if "clean_text" not in self.df.columns:
            self.clean_data()

        X = self.df["clean_text"]
        y = self.df["sentiment_encoded"]

        # Split into train+val and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        return self.X_train, self.X_test, self.y_train, self.y_test


    def vectorize_text(self):
        """
        Vectorizes the cleaned text using TF-IDF.
        
        Returns:
            A tuple containing:
              - (X_train_vec, X_val_vec, X_test_vec): TF-IDF matrices for training, validation, and test sets.
              - The fitted TfidfVectorizer.
              
        Raises:
            ValueError: If the data has not been split yet.
        """
        if self.X_train is None or self.X_test is None:
            raise ValueError("Data has not been split yet. Please call split_data() first.")

        self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,3))
        X_train_vec = self.vectorizer.fit_transform(self.X_train)
        X_test_vec = self.vectorizer.transform(self.X_test)

        return (X_train_vec, X_test_vec), self.vectorizer

    def get_processed_dataframe(self) -> pd.DataFrame:
        """
        Returns the processed DataFrame containing the cleaned text and encoded sentiment.
        """
        if "clean_text" not in self.df.columns or "sentiment_encoded" not in self.df.columns:
            self.clean_data()
        return self.df

