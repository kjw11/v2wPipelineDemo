import pandas as pd
import numpy as np
import string
import pickle
import joblib
import ast
from tqdm import tqdm

from gensim.models import Word2Vec  # only for gensim 3.8.3
import gensim
import re
import os
from scipy import spatial

import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from dataloader import Dataset
nltk.download('stopwords')
nltk.download('punkt')

print("Gensim version",gensim.__version__)

data_root = './txts_samples'  # as a sample
#data_root = '/data/jwkang/091921-world-brief/txts'
model_dir = './models'

# Get the list of stop words
stop_words = stopwords.words('english')
# add new stopwords to the list
stop_words.extend(["could","though","would","also","many",'much'])


def train_country(countryname):

    filename = countryname
    print(filename)
    sentences_corpus = []

    #country_root = os.path.join(data_root, countryname)
    country_root = data_root
    #corpus_file = 'corpus_' + countryname + '.joblib'
    model_file = os.path.join(model_dir, 'model_' +countryname + '.model')

    dataloader = Dataset(country_root, filename)

    #joblib.dump(sentences_corpus,open(corpus_file,'wb'))
    model_1 = Word2Vec(dataloader, min_count=5, workers=20)
    model_1.save(model_file)


def main():

    COUNTRIES = ['HK']  # as a sample
    #COUNTRIES = ['HK', 'JM','BD', 'MY', 'TZ','LK', 'PK', 'US', 'IE', 'AU', 'GB', 'CA', 'IN', 'NZ','ZA','SG', 'PH', 'GH', 'NG', 'KE']
    countries = [each_string.lower() for each_string in COUNTRIES]

    # run for each country
    for country in countries:
        print("**Country: {}**".format(country))
        train_country(country)


if __name__ == '__main__':
    main()
