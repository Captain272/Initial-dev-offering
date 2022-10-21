import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords
import pickle
import time
import string
import re
import requests 
import pandas as pd
from config import config


job='front end applications build frontend application using react typescript back end applications infrastructure build sdks backend solana contracts maintain chain infrastructure code review ensure code quality software reliability automated test implementations ui ux testing processes'
scaler = pickle.load(open(config.scaler_path, 'rb'))
vectorizer = pickle.load(open(config.vectorizer_path, 'rb'))
model = pickle.load(open(config.model_path, 'rb'))
data = pd.read_csv(config.data_path)


stop_words = stopwords.words('english')
data.content = data.content.apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))
stop_words = stopwords.words('english')

MULTIPLE_WHITESPACE = re.compile('\s+')
email_pattern = re.compile('\S*@\S*\s?')
keys = dict.fromkeys(string.punctuation)
for k in keys:keys[k] = " "
table = str.maketrans(keys)

def preprocess(x):
    x = x.lower()
    x = email_pattern.sub('email', x)
    x = re.sub('www\S+|http://\S+|https://\S+', 'url', x)
    x = x.translate(table)
    x = MULTIPLE_WHITESPACE.sub(" ", x).strip()
    x=x.replace('none', '')
    x = " ".join([word for word in x.split() if word not in stop_words])
    return x

def match(job,user_data):
    print("Matching words are :")
    l=[]
    for i in job.split(" "):
        if i in user_data.split(" "):
            l.append(i)

    return(set(l))



def find_dev(text):
    text=preprocess(text)
    vec = vectorizer.transform([text])
    scaled_vec = scaler.transform(vec.toarray())
    dists, idx = model.kneighbors(scaled_vec)
    df = data.iloc[idx[0]].reset_index(drop=True)
    for i in range(len(data.iloc[idx[0]])):
        print("USER ",i)
        print(match(text,data.iloc[idx[0][i]].content))

    return pd.DataFrame(df)

print(find_dev(job))

    