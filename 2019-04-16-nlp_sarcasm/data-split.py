#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 21:52:07 2019

@author: hugo
"""

# ==================================================================================================
# Working directory
# ==================================================================================================
import os
BASE_DIR = "/home/hugo/Documents/jedha/2019-04-16-nlp_sarcasm/"
os.chdir(BASE_DIR)

# ==================================================================================================
# Train/test split
# ==================================================================================================
from pickle import load
from pickle import dump
from data.const import N_ROWS, TEST_SIZE
from sklearn.model_selection import train_test_split

# chargement des données nettoyées
def load_clean_data(filename):
	return load(open(filename, 'rb'))

# sauvegarde d'une liste de phrase nettoyées dans un fichier
def save_clean_data(data, filename):
	dump(data, open(filename, 'wb'))
	print('Saved: %s' % filename)

# chargement et réduction des données nettoyées
n_rows = N_ROWS
data = load_clean_data("data/data-clean.pkl")
data = data[:n_rows]

# séparation en train / test
test_size = int(TEST_SIZE*data.shape[0])
train, test = train_test_split(data, test_size=test_size, random_state=0, 
                                 stratify=data.iloc[:,-1])

# sauvegarde dans deux fichiers
save_clean_data(data,  'output/data-clean-r{}-both.pkl'.format(N_ROWS))
save_clean_data(train, 'output/data-clean-r{}-train.pkl'.format(N_ROWS))
save_clean_data(test,  'output/data-clean-r{}-test.pkl'.format(N_ROWS))

