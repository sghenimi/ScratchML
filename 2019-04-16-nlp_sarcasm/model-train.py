#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 21:12:34 2019

@author: hugo
"""
# ==================================================================================================
# Working directory
# ==================================================================================================
import os
BASE_DIR = "/home/hugo/Documents/jedha/2019-04-16-nlp_sarcasm/"
os.chdir(BASE_DIR)

# ==================================================================================================
# Loading data
# ==================================================================================================
from pickle import load
from data.const import N_ROWS

# chargement des données nettoyées
def load_clean_data(filename):
	return load(open(filename, 'rb'))

# chargement des données nettoyées
data  = load_clean_data('output/data-clean-r{}-both.pkl'.format(N_ROWS))
train = load_clean_data('output/data-clean-r{}-train.pkl'.format(N_ROWS))

# ==================================================================================================
# Building model
# ==================================================================================================
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint
from data.const import VAL_SIZE
from sklearn.model_selection import train_test_split

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

# on defnit le modèle utilisant les LSTM
def define_model(vocab_size, timesteps, n_units):
    model = Sequential()
    model.add(Embedding(
                input_dim=vocab_size, 
                output_dim=n_units, 
                input_length=timesteps,
                mask_zero=True
            ))
    model.add(LSTM(n_units))
    model.add(Dense(
                units=2,
                activation="softmax"
            ))
    return model

# préparation du tokenizer
tokenizer = create_tokenizer(data.iloc[:,0])
vocab_size = len(tokenizer.word_index)+1
length = max_length(data.iloc[:,0])

# préparation des données d'apprentissage et de validation
val_size = int(VAL_SIZE*train.shape[0])
train, val = train_test_split(train, test_size=val_size, random_state=0, 
                                 stratify=train.iloc[:,-1])
xtrain, ytrain = encode_sequences(tokenizer, length, train.iloc[:,0]), to_categorical(train.iloc[:,1])
xval, yval = encode_sequences(tokenizer, length, val.iloc[:,0]), to_categorical(val.iloc[:,1])

# définition du modèle et print(l{n_layers})
model = define_model(vocab_size, length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
n_layers = len(model.layers)
filename = "output/model-{}l.png".format(n_layers)
plot_model(model, to_file=filename, show_shapes=True)

# ==================================================================================================
# Training model
# ==================================================================================================
from data.const import N_EPOCHS, BATCH_SIZE

# Calcule du modèle (r{n_rows}-l{n_layers}-e{n_epochs}-b{batch_size})
filename = "output1/model-l{}-r{}-e{}-b{}.h5".format(n_layers, N_ROWS, N_EPOCHS, BATCH_SIZE)
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, 
                             mode='min')
model.fit(xtrain, ytrain, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_data=(xval, yval), 
          callbacks=[checkpoint], verbose=2)

