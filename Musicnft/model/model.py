import pickle
import re
from pathlib import Path
from unittest.case import doModuleCleanups
from scipy.spatial import distance
import pandas as pd
__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

df=pd.read_csv(f"{BASE_DIR}/nft_mod.csv")   
df2=pd.read_csv(f"{BASE_DIR}/nftmusic_blockbeats_solana.csv")  
with open(f"{BASE_DIR}/savemodel.pkl", "rb") as f:
    model = pickle.load(f)

cluster_map = pd.DataFrame()
cluster_map['data_index'] = df.index.values
cluster_map['cluster'] = model.labels_



def query(l):

    k=make_mat(df,[l])
    return k


def get_points(p):
    k=cluster_map.loc[cluster_map['cluster']==p]
    
    return k['data_index']


def make_mat(df,x):
    
    cluster=model.predict(x)

    points=get_points(cluster[0])
    
    tk=df['Track'].values
    
    mc=df['Music'].values
    
    tm=df['time'].values
    
    cp=df.iloc[points]
   
    cp=cp.fillna(df.mean())
    
    song_names=cp['name'].values
   
    p=[]
    
    k=set(df.iloc[points]['name'])

    j=[]
    for i in k:
       j.append(i[:-5]) 
    j=list(set(j))
    
    
    
    
    count=0
    for i in cp[cp.drop(columns=['name']).columns].values:
        
        p.append([distance.euclidean(x[0],i),count])
        
        count+=1
    p.sort()
    l_rec=[]
    for i in range(5):
        l_rec.append(song_names[p[i][1]])
       
    l_top=[]
    for i in j:
        ll=0
        k=0

        
        while(ll!=5):
            if i in song_names[p[k][1]]:
                
                l_top.append(song_names[p[k][1]])
                ll+=1
            k+=1
    top_rec=[]   
    alt_rec=[]
    for i in l_rec:
        dict={}
        ktop=df2.loc[df2['name']==i]
        k2=list((ktop['contract_address'].values))
        base_url="https://magiceden.io/item-details/"+k2[0]+"?name="
        a,b = i.split('#', 1)
        name=a.replace(" ","-")
        b='%23'+b
        name=name[:len(name)-1]
        name=name+b     
        

        y1=list((ktop['image_url'].values))
        dict[i]=[y1[0],base_url+name]
        top_rec.append(dict)
    for i in l_top:
        dict={}
        ktop=df2.loc[df2['name']==i]
        k3=list((ktop['contract_address'].values))
        base_url="https://magiceden.io/item-details/"+k3[0]+"?name=" 
        a,b = i.split('#', 1)
        name=a.replace(" ","-")
        b='%23'+b
        name=name[:len(name)-1]
        name=name+b     
         
        y=list((ktop['image_url'].values))
        dict[i]=[y[0],base_url+name]
        alt_rec.append(dict)    
    print(l_rec)
                
    return [ top_rec,alt_rec]  
            
           
    
