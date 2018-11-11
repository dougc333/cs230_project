import numpy as np
import pandas as pd
import os
import io
import json
import math
import pickle

from time import time

from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression

import spacy
nlp = spacy.load('en',disable=['parser', 'tagger', 'ner'])
STOPWORDS = list(spacy.lang.en.STOP_WORDS) 

from scipy.spatial.distance import cosine

#handle out of vocabulary words
#oov = np.random.rand(300)*1e-10
oov = np.zeros(300)
oov[0]=1.0

def clean_txt(txt):
    """
    Preprocess text data
    """
    txt = txt.strip()
    txt = txt.lower()
    return txt

def bow2vec(txt,nlp=nlp,fasttext=None,method='spacy',stopwords=True):
    """
    Converts txt to a list of word vectors,"bag of words" encoding,
    and then averages the vectors to produce a single vector encoding
    """
    txt = clean_txt(txt)
    
    words = nlp(txt)
    
    if stopwords:
        words = [w for w in words if not w.text in STOPWORDS]
       
    # handle case of empty string
    if len(words)==0:
        words=nlp(u"empty")
    
    if method == 'spacy':
        vecs = [word.vector if (not word.is_oov) else oov for word in words ]

    elif method=='fasttext':
        vecs= [fasttext.get_word_vector(word.text) for word in words]
  
    else:
        print("Error: unknown method!")
    
    return np.mean(vecs,0)


def cosine_similarity(v1,v2):
    return 1. - cosine(v1,v2)

def build_features(df,nlp=nlp,method='spacy',stopwords=True,add_to_df=True):
    """
    Build similarity features from a pandas dataframe of question pairs.
    """
    fasttext = None
    
    if method=='spacy':
        print("Loading spacy 'en_core_web_lg'...")
        nlp = spacy.load('en_core_web_lg',disable=['parser', 'tagger', 'ner'])
        
    elif method=='fasttext':
        import fastText as ft
        print("Loading fasttext embeddings...")        
        fasttext = ft.load_model('/data/demo_quora_data/crawl-300d-2M-subword.bin')   
         
    print("Vectorizing question1...")
    q1_vec = [bow2vec(q,nlp=nlp,fasttext=fasttext,method=method,stopwords=stopwords) 
                for q in  df['question1'].values]
    print("...Finished vectorizing question 1")

    print("Vectorizing question2...")
    q2_vec = [bow2vec(q,nlp=nlp,fasttext=fasttext,method=method,stopwords=stopwords) 
                for q in  df['question2'].values]
    print("...Finished vectorizing question2")
    
    # BoW difference vector
    bow_diff=np.array([abs(q2 - q1) for (q1,q2) in zip(q1_vec,q2_vec)])
    
    # BoW cosine feature
    print("Building BoW cosine similarity...")
    cos_sim = np.array([cosine_similarity(q1,q2) for (q1,q2) in zip(q1_vec,q2_vec)])

    # BoW distance feature
    print("Building BoW euclidean similarity...")    
    euclidean_sim = np.array([np.linalg.norm(q2 - q1) for (q1,q2) in zip(q1_vec,q2_vec)])

    # BoW sum vector
    #bow_sum=np.array([ q1 + q2 for (q1,q2) in zip(q1_vec,q2_vec)])
    
    X = np.hstack((q1_vec, 
                   q2_vec, 
                   bow_diff, 
                   cos_sim.reshape(-1,1),
                   euclidean_sim.reshape(-1,1)))

    if add_to_df:
        df['cos_sim'] = cos_sim
        df['euclidean_sim'] = euclidean_sim
        
    return X


def predict(X,model_filename):
    """ 
    Makes prediction on dataset of features using a saved model
    """
    # Load from file and generate predicts
    with open(model_filename, 'rb') as file:  
        saved_model = pickle.load(file)

    y_predict = saved_model.predict(X)
    
    return y_predict

