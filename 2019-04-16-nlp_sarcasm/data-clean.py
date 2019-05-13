#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:13:20 2019

@author: hugo
"""

# ==================================================================================================
# Working directory
# ==================================================================================================
import os
BASE_DIR = "/home/hugo/Documents/jedha/2019-04-16-nlp_sarcasm/"
os.chdir(BASE_DIR)

# ==================================================================================================
# Loading, cleaning and saving data
# ==================================================================================================
import pandas as pd
import re
import string
from unicodedata import normalize
from pickle import dump

# on charge les documents en mémoire
def load_doc_json(filename):
    data = pd.read_json(filename, lines=True)
    data = data.drop("article_link", axis=1)
    return data

# Nettoyer une liste de lignes
def clean(data):
    data_clean = pd.DataFrame(columns=data.columns)
    # On prépare la focntion regex pour filtrer certains caractères
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # preparer la table de traduction pour retirer la ponctuation
    table = str.maketrans('', '', string.punctuation)
    for index, line in data.iterrows():
        text = line[0]
        # on normalise les caractères en ascii
        text = normalize('NFD', text).encode('ascii', 'ignore')
        text = text.decode('UTF-8')
        # on produit des tokens de mots en utilisant les espaces comme séparateurs
        text = text.split()
        # On passe en minuscule
        text = [word.lower() for word in text]
        # on enleve la ponctuation de chaque token
        text = [word.translate(table) for word in text]
        # on supprime tous les caractères non imprimables
        text = [re_print.sub('', w) for w in text]
        # On supprime les tokens contenant des chiffres
        text = [word for word in text if word.isalpha()]
		# enfin on enregistre les token comme des chaines de caractères
        data_clean = data_clean.append({
                    data.columns[0]:' '.join(text),
                    data.columns[1]:line[1]
                }, ignore_index=True)
    return data_clean

# on sauvegarde un fichier de phrases nettoyées
def save_clean_data(data, filename):
    dump(data, open(filename, 'wb'))
    print('Saved: %s' % filename)

data = load_doc_json("data/data-raw.json")
data = clean(data)
save_clean_data(data, "data/data-clean.pkl")

