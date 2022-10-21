import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import pickle
from config import config

st.title("Initial developer Offerring")

scaler = pickle.load(open(config.scaler_path, 'rb'))
vectorizer = pickle.load(open(config.vectorizer_path, 'rb'))
model = pickle.load(open(config.model_path, 'rb'))
data = pd.read_csv(config.data_path)
text = st.text_area("Describe your job:", height=200)

if st.button('find developers'):
    vec = vectorizer.transform([text])
    scaled_vec = scaler.transform(vec.toarray())
    dists, idx = model.kneighbors(scaled_vec)
    df = data.iloc[idx[0]].reset_index(drop=True)
    st.dataframe(df)
    