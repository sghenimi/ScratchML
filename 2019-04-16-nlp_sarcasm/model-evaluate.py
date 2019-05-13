#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 21:17:19 2019

@author: hugo
"""

# ==================================================================================================
# Working directory
# ==================================================================================================
import os
BASE_DIR = "/home/hugo/Documents/jedha/2019-04-16-nlp_sarcasm/"
os.chdir(BASE_DIR)

# ==================================================================================================
# Evaluating model
# ==================================================================================================
from pickle import load
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from data.const import N_ROWS, N_EPOCHS, BATCH_SIZE

# chargement des données nettoyées
def load_clean_data(filename):
	return load(open(filename, 'rb'))

# On transforme chaque mot en token
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# on définit une longueur de phrase maximale
def max_length(lines):
    return max(len(line.split()) for line in lines)

# on encode et on pad les sequences de mots pour qu'elles aient toutes le même format
def encode_sequences(tokenizer, length, lines):
    # On encode les séquences de mots comme des séquences de chiffre
    X = tokenizer.texts_to_sequences(lines)
    # Espaces vides -> 0 pour que toutes les phrases aient la longueur maximale
    X = pad_sequences(X, maxlen=length, padding='post')
    return X

# générer une prédiction en fonction d'une entrée
def predict(model, source):
    prediction = model.predict(source, verbose=0)
    prediction = [argmax(vector) for vector in prediction]
    return prediction

# évaluation des performances du modèle
def evaluate_model(model, sources, raw_dataset):
    actual = raw_dataset.iloc[:,-1].astype(int)
    predicted = predict(model, sources)
    rate = 1-sum(abs(actual-predicted))/len(actual)
    return rate
        
# chargement des données
data  = load_clean_data('output/data-clean-r{}-both.pkl'.format(N_ROWS))
train = load_clean_data('output/data-clean-r{}-train.pkl'.format(N_ROWS))
test  = load_clean_data('output/data-clean-r{}-test.pkl'.format(N_ROWS))

# préparation du tokenizer
tokenizer = create_tokenizer(data.iloc[:,0])
vocab_size = len(tokenizer.word_index)+1
length = max_length(data.iloc[:,0])

# préparation des données d'apprentissage et de validation
xtrain = encode_sequences(tokenizer, length, train.iloc[:,0])
xtest = encode_sequences(tokenizer, length, test.iloc[:,0])

# chargement du modèle pré entraîné
model = load_model("output/model-l{}-r{}-e{}-b{}.h5".format(3, N_ROWS, N_EPOCHS, BATCH_SIZE))

# test sur des données train et test 
print('train\t', evaluate_model(model, xtrain, train))
print('test\t', evaluate_model(model, xtest, test))
