import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

import sklearn
print(sklearn.__version__)
nltk.download('all')
from nltk.tokenize import sent_tokenize

stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

def preprocessing(text):
    text = text.lower()

    text = re.sub('<.*?>', '', text)

    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    cleaned = []
    for i in text.split():
        if i not in stop_words:
            cleaned.append(i)
    text= ' '.join(cleaned)

    tokens = word_tokenize(text)
    stemmed = [porter.stem(token) for token in tokens]
    text= ' '.join(stemmed)
    return text


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_msg = st.text_input("Enter the Message")

if st.button("Predict"):
    # 1) preprocess
    transformed_msg = preprocessing(input_msg)
    # 2) vectorize
    vector_input = tfidf.transform([transformed_msg])
    # 3) predict
    result = model.predict(vector_input)[0]
    # 4) display
    if result == 1:
        st.error("🚫 Spam")
    else:
        st.success("✅ Not Spam")

