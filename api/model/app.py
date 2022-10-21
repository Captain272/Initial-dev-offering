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
# from config import config


data_path = 'artifacts/cleaned_users.csv'
model_path = 'artifacts/model.pkl'
scaler_path = 'artifacts/scaler.pkl'
vectorizer_path = 'artifacts/vectorizer.pkl'

from pathlib import Path
# from unittest.case import doModuleCleanups

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

# df=pd.read_csv(f"{BASE_DIR}/nft_mod.csv")   
# df2=pd.read_csv(f"{BASE_DIR}/nftmusic_blockbeats_solana.csv")  
# with open(f"{BASE_DIR}/savemodel.pkl", "rb") as f:
#     model = pickle.load(f)


# f"{BASE_DIR}/

scaler = pickle.load(open(f"{BASE_DIR}/"+scaler_path, 'rb'))
vectorizer = pickle.load(open(f"{BASE_DIR}/"+vectorizer_path, 'rb'))
model = pickle.load(open(f"{BASE_DIR}/"+model_path, 'rb'))
data = pd.read_csv(f"{BASE_DIR}/"+data_path)


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
    l=[]
    for i in range(len(data.iloc[idx[0]])):
        print("USER ",i)
        # print(match(text,data.iloc[idx[0][i]].content))
        l.append(match(text,data.iloc[idx[0][i]].content))
    print(l)
    df['matches']=l
    return pd.DataFrame(df)


    