# NLP Preprocessing and Text Classification

## Problem Statement

This assignment focuses on applying Natural Language Processing (NLP) techniques to preprocess text data and build machine learning models for text classification. The goal is to implement a complete pipeline, from data cleaning to model evaluation, and compare the performance of different text vectorization methods and classification algorithms.

## Provided Solution

This notebook provides a step-by-step implementation of an NLP text classification project.

- **Data Loading and Exploration**: The Movie Reviews Dataset is loaded and inspected to understand its structure, check for missing values, and analyze the class distribution.

- **Text Preprocessing**:
    - **Tokenization and Stopword Removal**: Text is tokenized into individual words, and common English stopwords are removed using NLTK.
    - **Stemming and Lemmatization**: Both PorterStemmer (for stemming) and WordNetLemmatizer (for lemmatization) are applied to reduce words to their root forms. The results of both techniques are compared.

- **Text Vectorization**:
    - **TF-IDF**: `TfidfVectorizer` is used to convert the preprocessed text into a matrix of TF-IDF features.
    - **CountVectorizer**: `CountVectorizer` is used to create a document-term matrix based on word counts.

- **Model Building**: Two different classification models are trained on the vectorized text data:
    - **Multinomial Naive Bayes**: A probabilistic classifier suitable for text data.
    - **Logistic Regression**: A linear model for binary classification.

- **Model Evaluation**: The performance of both models is evaluated using the following metrics:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - Confusion Matrix
    - A detailed classification report is generated for each model.

- **Results**: The evaluation metrics and confusion matrices are displayed to compare the performance of the Naive Bayes and Logistic Regression classifiers. The findings demonstrate the effectiveness of these NLP techniques for sentiment classification.
