Description:
The goal of this project is to accurately classify SMS messages as spam or ham (not spam) using natural language processing (NLP) techniques. The project pipeline includes:

Data preprocessing using the NLTK library for tokenization, stopword removal, and text normalization.

Feature extraction with TF-IDF Vectorizer to convert text messages into numerical feature vectors.

Model training and evaluation using the Naive Bayes algorithm, which is particularly effective for text classification tasks.

Dataset:
The model is trained and evaluated on the publicly available UCI SMS Spam Collection Dataset, which contains thousands of labeled SMS messages.

Technologies Used:
Python

NLTK for text preprocessing

Scikit-learn for TF-IDF vectorization and Naive Bayes classification

Pandas and NumPy for data manipulation

matplotlib and seaborn for representations

Features:
Preprocessing steps such as tokenization, stopword removal, and text normalization

TF-IDF vectorization to convert text into meaningful features

Naive Bayes classifier for spam detection

Model evaluation with accuracy and precision metrics

