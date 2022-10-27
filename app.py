import pickle
import re
import string
import time

import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords
from config import config

MULTIPLE_WHITESPACE = re.compile('\s+')
email_pattern = re.compile('\S*@\S*\s?')
keys = dict.fromkeys(string.punctuation)
for k in keys:keys[k] = " "
table = str.maketrans(keys)

stop_words = stopwords.words('english')
def preprocessor(x):
    x = x.lower()
    x = email_pattern.sub('email', x)
    x = re.sub('www\S+|http://\S+|https://\S+', 'url', x)
    x = x.translate(table)
    x = MULTIPLE_WHITESPACE.sub(" ", x).strip()
    x=x.replace('none', '')
    x = " ".join([word for word in x.split() if word not in stop_words])
    return x

st.title("Initial developer Offerring")

scaler = pickle.load(open(config.scaler_path, 'rb'))
vectorizer = pickle.load(open(config.vectorizer_path, 'rb'))
model = pickle.load(open(config.model_path, 'rb'))
data = pd.read_csv(config.data_path)
text = st.text_area("Describe your job:", height=200)

if st.button('find developers'):
    text = preprocessor(text)
    vec = vectorizer.transform([text])
    scaled_vec = scaler.transform(vec.toarray())
    dists, idx = model.kneighbors(scaled_vec)
    df = data.iloc[idx[0]].reset_index(drop=True)
    st.dataframe(df)
    