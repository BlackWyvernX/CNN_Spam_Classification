import re
import string
import pandas as pd
import nltk
import os
from nltk import word_tokenize
nltk.download('punkt_tab')

def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def preprocess_data():
    dir = os.getcwd()
    df = pd.read_csv(os.path.join(dir, 'spam.csv'), encoding='latin-1')
    df = df.iloc[:, :2] 
    df = df.dropna(axis=1, how='all')
    df.columns = ["label", "text"]
    df['text'] = df['text'].apply(clean_text)
    df['text'] = df['text'].apply(word_tokenize)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df
