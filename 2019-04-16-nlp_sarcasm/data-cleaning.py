#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:04:46 2019

@author: hugo
"""

# ==================================================================================================
# 
# ==================================================================================================
import os
import time

os.chdir("/home/hugo/Documents/nlp-ts-processing/")

# ==================================================================================================
# Functions for loading/saving data
# ==================================================================================================
from pickle import load, dump
import pandas as pd

# On charge une partie des données
def load_data_part(path):
    return load(open(path, 'rb'))

# On sauvegarde les données traitées
def save_data_part(data, path):
    dump(data, open(path, 'wb'))

# ==================================================================================================
# Functions for cleaning data : stopwords
# ==================================================================================================
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
stopwords = set(stopwords.words('english')) 

# Suprression des stopwords d'une phrase
def clean_sentence_stopwords(sentence):
    tokens = word_tokenize(sentence)
    sentence_clean = [w for w in tokens if not w in stopwords]
    sentence_clean = ' '.join(sentence_clean)
    return sentence_clean

# Suprression des stopwords du DataFrame
def clean_data_stopwords(data):
    corpus = pd.DataFrame(data.raw)
    corpus['content'] = data.content.apply(lambda x: [clean_sentence_stopwords(e) for e in x])
    corpus['summary'] = data.summary.apply(lambda x: [clean_sentence_stopwords(e) for e in x])
    return corpus

# ==================================================================================================
# Script execution ~?' (cnn->45"/part; dailymail->45"/part); google->2"/part; reddit->15"/part)
# ==================================================================================================

print('DATA CLEANING', '*'*52, time.asctime())

datafiles = sorted(os.listdir('data-format/'))

for datafile in datafiles:
    print('Cleaning...\t', end='')
    corpus = load_data_part('data-format/'+datafile)
    corpus = clean_data_stopwords(corpus)
    save_data_part(corpus, 'data-stopwords/'+datafile)
    print('Done\t'+datafile+'\t', time.asctime())

print('END', '*'*62, time.asctime())

tab = list()
for filename in os.listdir('data-format/'):
    x = [[filename, load(open('data-format/'+filename, 'rb')).shape]]
    tab += x
    print(x)
size_not_10000 = len([(x, x[1][0]) for x in tab if x[1][0]!=10000])
