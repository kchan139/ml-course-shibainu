# src/features/build_features.py
from scipy.sparse import issparse
import numpy as np
import os

class FeatureExtractor:
    """
    This class extracts features from the headlines.
    """
    
    def __init__(self, embedding_dim=300):
        """
        Initialize the FeatureExtractor.
        
        Args:
            embedding_dim: Dimension of word embeddings
        """
        self.embedding_dim = embedding_dim
        self.word_index = {}
        self.vocab_size = 0
        self.max_length = 0
    
    def fit_vocabulary(self, texts, max_words=10000, max_length=100):
        # Check if texts is a sparse matrix
        if issparse(texts):
            raise ValueError("texts must be a list of strings, not a sparse matrix. Please pass raw text data.")
        
        # Check if texts is a numpy array and convert to list of strings
        if isinstance(texts, (np.ndarray, np.matrix)):
            texts = [" ".join(map(str, text)) for text in texts]
        
        word_counts = {}
        for text in texts:
            for word in text.lower().split():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        self.word_index = {'<PAD>': 0, '<UNK>': 1}
        for i, (word, _) in enumerate(sorted_words[:max_words-2]):
            self.word_index[word] = i + 2
        
        self.vocab_size = min(max_words, len(word_counts) + 2)
        self.max_length = max_length
        return self.word_index, self.vocab_size
    
    def load_glove_embeddings(self, glove_path='glove.6B.300d.txt'):
        """
        Load GloVe embeddings and create an embedding matrix for the model.
        
        Args:
            glove_path: Path to the GloVe embeddings file
            
        Returns:
            embedding_matrix: Matrix of word embeddings (vocab_size x embedding_dim)
        """
        if not self.word_index:
            raise ValueError("Vocabulary not fitted. Call fit_vocabulary first.")
        
        # Initialize with zeros
        embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))
        
        # Check if the GloVe file exists
        if not os.path.exists(glove_path):
            print(f"Warning: GloVe embeddings file {glove_path} not found.")
            print("Returning zero embedding matrix. Please download GloVe embeddings.")
            return embedding_matrix
        
        # Load GloVe embeddings
        embeddings_index = {}
        with open(glove_path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        
        print(f"Loaded {len(embeddings_index)} word vectors.")
        
        # Create the embedding matrix
        for word, i in self.word_index.items():
            if i >= self.vocab_size:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        
        return embedding_matrix
    
    def texts_to_sequences(self, texts):
        """
        Convert texts to sequences of word indices.
        
        Args:
            texts: List of text samples
            
        Returns:
            sequences: List of sequences (lists of word indices)
        """
        if not self.word_index:
            raise ValueError("Vocabulary not fitted. Call fit_vocabulary first.")
        
        sequences = []
        for text in texts:
            seq = []
            for word in text.lower().split():
                seq.append(self.word_index.get(word, self.word_index['<UNK>']))
            sequences.append(seq)
        return sequences
    
    def pad_sequences(self, sequences):
        """
        Pad sequences to the same length.
        
        Args:
            sequences: List of sequences (lists of word indices)
            
        Returns:
            padded_sequences: Numpy array of padded sequences
        """
        padded_sequences = np.zeros((len(sequences), self.max_length))
        for i, seq in enumerate(sequences):
            if len(seq) > self.max_length:
                padded_sequences[i] = seq[:self.max_length]
            else:
                padded_sequences[i, :len(seq)] = seq
        return padded_sequences
