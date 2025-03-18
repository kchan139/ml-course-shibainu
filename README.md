# Machine Learning Course (CO3117-HCMUT)

## Project Setup Guide

### Environment Requirements
- **Python Version**: 3.9.x - 3.12.x (Recommended: 3.11)
- **Supported Operating Systems**: 
  - Windows 10/11
  - macOS (Catalina and newer)
  - Linux (Ubuntu 20.04+)

### Development Environment Setup

#### Prerequisites
- Python (3.9-3.12)

#### Setup Steps

##### Windows
1. Install Python from official website
2. Open Command Prompt
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

##### macOS/Linux
```bash
# Install Python via Homebrew (macOS) or package manager (Linux)
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Team Members and Workload Distribution

| Member              | ID      | Responsibilities                        | Assigned Model      |
| ------------------- | ------- | --------------------------------------- | ------------------- |
| Huynh Kiet Khai     | 2252338 | Model optimization                      | Decision Tree       |
| Tran Dinh Dang Khoa | 2211649 | Repository setup, project coordination  | Neural Network      |
| Ha Tuan Khang       | 2252289 | Validation strategy                     | Naive Bayes         |
| Phan Chi Vy         | 2252938 | Data preprocessing, feature engineering | Bayesian Network    |
| Nguyen Duc Tam      | 2252734 | Data Visualization                      | Hidden Markov Model |

## Introduction

This report documents Phase 1 of our machine learning project, focusing on the implementation and comparative analysis of various ML models for sentiment analysis of news headlines. Our goal is to demonstrate a practical understanding of ML models covered in Chapters 2-6 of the course curriculum. We've implemented Decision Trees, Neural Networks, Naive Bayes, and Graphical Models (Bayesian Networks and Hidden Markov Models) to classify news headlines into positive, neutral, or negative sentiment categories.

## Problem Statement & Dataset

### Problem Statement

We aim to create a sentiment analysis system capable of accurately classifying news headlines into three sentiment categories: negative, neutral, and positive. This task is challenging due to the brevity of headlines, implied sentiment, and contextual nuances.

### Dataset

Our dataset consists of news headlines labeled with sentiment scores. The preprocessing pipeline handles cleaning, tokenization, and feature extraction to prepare the data for various modeling approaches.

### Challenges

- Short text length with limited contextual information
- Implicit sentiment not directly expressed through sentiment words
- Class imbalance with neutral headlines comprising the majority
- Need for different feature representations depending on the model type

## Methodology

### Data Preprocessing

- Text cleaning (lowercase conversion, punctuation removal)
- Stopword removal and lemmatization
- Feature extraction using vectorization techniques:
    - Count-based vectorization for Decision Trees and Naive Bayes
    - Word embeddings for Neural Networks
    - Binary feature presence for Bayesian Networks
    - Discretized features for specific probabilistic models

### Model Selection

For Phase 1, we implemented:

1. Decision Trees
2. Neural Networks
3. Naive Bayes
4. Bayesian Networks
5. Hidden Markov Models

## Model Implementations

### Decision Tree Model

**Key Features:**

- Grid search for hyperparameter optimization
- Pruning through max depth and min samples leaf parameters
- Model persistence for reusability
- Handles high-dimensional text features after vectorization

### Neural Network Model

We implemented a bidirectional LSTM model specifically designed to handle class imbalance in sentiment data:

**Key Features:**

- Bidirectional LSTM architecture to capture context in both directions
- Class weights to handle imbalanced data distribution
- Custom threshold logic during prediction for better minority class detection
- Word embeddings to represent semantic relationships between words
- Regularization techniques (dropout, L2) to prevent overfitting

### Naive Bayes Model

**Key Features:**

- Optimized for sparse text feature representation using TF-IDF
- Smoothing implemented via alpha parameter tuning
- Uses weighted F1 scoring for optimization to handle class imbalance
- Maintains all components (model, vectorizer, encoder) for reproducible predictions

### Bayesian Networks

We implemented two variants of Bayesian Networks: a traditional approach with n-gram features and another using Word2Vec embeddings


**Key Features:**

- Structure learning with Hill Climbing and BIC score optimization
- Feature selection with chi-square tests for traditional approach
- Word embeddings with discretization for the Word2Vec variant
- Variable elimination for efficient inference
- Effective handling of feature dependencies

### Hidden Markov Model

**Key Features:**

- Separate models for each sentiment class
- Class-conditional sequence modeling
- Gaussian emission probabilities for continuous features
- Log-likelihood scoring for classification
- Multiple hidden states to capture text patterns

## Results and Comparative Analysis

### Performance Overview

Our models demonstrated varying performance characteristics on the test dataset:

|Model|Accuracy|
|---|---|
|Neural Network|85.37%|
|Naive Bayes|84.05%|
|Decision Tree|81.79%|
|Bayesian Network|75.14%|
|Hidden Markov Model|32.73%|

### Detailed Model Performance Metrics

#### Neural Network (RNN)

|              | precision | recall | f1-score |
| ------------ | --------- | ------ | -------- |
| Class 0      | 0.94      | 0.88   | 0.91     |
| Class 1      | 0.77      | 0.89   | 0.83     |
| Class 2      | 0.88      | 0.79   | 0.83     |
| accuracy     |           |        | 0.85     |
| macro avg    | 0.86      | 0.85   | 0.85     |
| weighted avg | 0.86      | 0.85   | 0.85     |

#### Naive Bayes

|              | precision | recall | f1-score |
| ------------ | --------- | ------ | -------- |
| Class 0      | 0.92      | 0.88   | 0.90     |
| Class 1      | 0.73      | 0.93   | 0.82     |
| Class 2      | 0.92      | 0.72   | 0.81     |
| accuracy     |           |        | 0.84     |
| macro avg    | 0.86      | 0.84   | 0.84     |
| weighted avg | 0.86      | 0.84   | 0.84     |

#### Decision Tree

|              | precision | recall | f1-score |
| ------------ | --------- | ------ | -------- |
| Class 0      | 0.94      | 0.83   | 0.88     |
| Class 1      | 0.69      | 0.94   | 0.80     |
| Class 2      | 0.90      | 0.68   | 0.78     |
| accuracy     |           |        | 0.82     |
| macro avg    | 0.85      | 0.82   | 0.82     |
| weighted avg | 0.84      | 0.82   | 0.82     |

#### Bayesian Network

|              | precision | recall | f1-score |
| ------------ | --------- | ------ | -------- |
| Class 0      | 0.95      | 0.73   | 0.83     |
| Class 1      | 0.61      | 0.90   | 0.73     |
| Class 2      | 0.83      | 0.62   | 0.71     |
| accuracy     |           |        | 0.75     |
| macro avg    | 0.80      | 0.75   | 0.75     |
| weighted avg | 0.80      | 0.75   | 0.75     |

#### Hidden Markov Model

|              | precision | recall | f1-score |
| ------------ | --------- | ------ | -------- |
| Class 0      | 0.42      | 0.02   | 0.04     |
| Class 1      | 0.00      | 0.00   | 0.00     |
| Class 2      | 0.33      | 0.97   | 0.49     |
| accuracy     |           |        | 0.33     |
| macro avg    | 0.25      | 0.33   | 0.18     |
| weighted avg | 0.25      | 0.33   | 0.18     |


### Comparative Analysis

1. **Neural Network (RNN)** achieved the highest overall accuracy at 85.37%. It shows strong balanced performance across all three classes with particularly good precision for negative (Class 0) and positive (Class 2) sentiments at 0.94 and 0.88 respectively. The model demonstrates its ability to effectively learn sequential patterns in text data.
    
2. **Naive Bayes** performed nearly as well with 84.05% accuracy. It shows excellent precision for negative and positive classes (0.92 for both) but lower precision for neutral content (0.73), suggesting it sometimes misclassifies other sentiments as neutral. The high recall for neutral content (0.93) indicates it captures most instances of this class.
    
3. **Decision Tree** achieved a solid 81.79% accuracy with similar patterns to Naive Bayes. It shows high precision for negative and positive classes (0.94 and 0.90) but struggles with neutral precision (0.69). However, it has the highest recall for neutral content (0.94), making it effective at identifying this class.
    
4. **Bayesian Network** reached 75.14% accuracy with an interesting pattern: it has the highest precision for negative sentiment (0.95) but lower recall (0.73), suggesting it's very selective about what it classifies as negative. It shows a bias toward classifying content as neutral (0.90 recall) but with lower precision (0.61).
    
5. **Hidden Markov Model** performed poorly with only 32.73% accuracy. The confusion matrix shows it classified almost everything as positive (Class 2), with 97% recall but only 33% precision for this class. It completely failed to identify neutral content (Class 1) with 0% for all metrics, suggesting fundamental issues with the model's ability to capture relevant patterns for sentiment classification.
    
6. **Class Imbalance Effects**: All models except HMM show some bias toward the neutral class (Class 1) with higher recall but lower precision, reflecting the class distribution in the training data. Neural Networks and Naive Bayes managed this challenge better than the others.
    

## Conclusion

### Phase 1 Achievements

- Successfully implemented five different model types for sentiment analysis
- Achieved strong performance with Decision Tree, Neural Networks, Naive Bayes and Bayesian Network models
- Identified strengths and weaknesses of different approaches for the sentiment classification task

### Challenges and Limitations

- Hidden Markov Model performed poorly, suggesting it may not be appropriate for this type of classification task
- Short headline text provides limited context for sentiment analysis
- Some models struggle with capturing semantic relationships in short text

## References

- Scikit-learn documentation for model implementations
- PGMPY documentation for graphical models
- TensorFlow documentation for neural network implementation
- Course lecture materials on ML algorithms