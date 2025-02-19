# src/visualization/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer

class Visualizer:
    """
    This class handles visualization of dataset characteristics and model performance.
    """
    
    def plot_class_distribution(self, data):
        """
        Visualizes the distribution of sentiment classes.
        
        Args:
            data (pd.DataFrame): DataFrame containing 'Sentiment' column
        """
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x='Sentiment', data=data)
        plt.title('Sentiment Class Distribution')
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', 
                        (p.get_x() + p.get_width()/2., p.get_height()),
                        ha='center', va='center', 
                        xytext=(0, 5), 
                        textcoords='offset points')
        plt.show()

    def generate_word_cloud(self, data, sentiment):
        """
        Generates a word cloud for a specific sentiment.
        
        Args:
            data (pd.DataFrame): DataFrame containing text data
            sentiment (str): Target sentiment to visualize
        """
        text = ' '.join(data[data['Sentiment'] == sentiment]['News Headline'].astype(str))
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=STOPWORDS,
            max_words=100
        ).generate(text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud for "{sentiment}" Sentiment', pad=20)
        plt.axis('off')
        plt.show()

    def plot_text_length_distribution(self, data):
        """
        Visualizes text length distribution by sentiment.
        
        Args:
            data (pd.DataFrame): DataFrame containing text data
        """
        data['Text Length'] = data['News Headline'].apply(lambda x: len(str(x).split()))
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Sentiment', y='Text Length', data=data)
        plt.title('Text Length Distribution by Sentiment')
        plt.ylabel('Number of Words')
        plt.xlabel('')
        plt.show()

    def plot_top_ngrams(self, data, n=2, top_k=10):
        """
        Plots top n-grams for each sentiment.
        
        Args:
            data (pd.DataFrame): DataFrame containing text data
            n (int): Number of grams (1=unigrams, 2=bigrams, etc.)
            top_k (int): Number of top n-grams to show
        """
        sentiments = data['Sentiment'].unique()
        
        for sentiment in sentiments:
            plt.figure(figsize=(12, 6))
            texts = data[data['Sentiment'] == sentiment]['News Headline'].astype(str)
            
            vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')
            matrix = vectorizer.fit_transform(texts)
            
            ngram_counts = matrix.sum(axis=0)
            ngram_names = vectorizer.get_feature_names_out()
            
            sorted_ngrams = sorted(
                zip(ngram_names, ngram_counts.tolist()[0]),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
            sns.barplot(
                x=[count for _, count in sorted_ngrams],
                y=[ngram for ngram, _ in sorted_ngrams]
            )
            plt.title(f'Top {n}-grams for "{sentiment}" Sentiment')
            plt.xlabel('Frequency')
            plt.ylabel('')
            plt.tight_layout()
            plt.show()