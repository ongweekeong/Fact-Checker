#!/usr/bin/env python.

"""
CS4248 ASSIGNMENT 2

TODO: Modify the variables below.  Add sufficient documentation to cross
reference your code with your writeup.

"""

# Import libraries.  Add any additional ones here.
# Generally, system libraries precede others.
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import seaborn as sns

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import gensim
from gensim.parsing.preprocessing import remove_stopwords

from tensorflow.keras import models, layers, preprocessing as kprocess
from tensorflow.keras import backend as K

nltk.download('punkt')
nltk.download('stopwords')
STOPWORDS = stopwords.words('english')
_STUDENT_NUM = 'A0168932U'

def train_model(model, X_train, y_train):
    ''' TODO: train your model based on the training data '''
    pass

def predict(model, X_test):
    ''' TODO: make predictions here '''
    return [0]*len(X_test)

def generate_result(test, y_pred, filename):
    ''' generate csv file base on the y_pred '''
    test['Verdict'] = pd.Series(y_pred)
    test.drop(columns=['Text'], inplace=True)
    test.to_csv(filename, index=False)

def preprocess(X_train):
    ''' Remove stop words, punctuations, and do normalizations '''
    X_train_clean = []
    for sentence in X_train:
        tokens = []
        tokens += word_tokenize(sentence.lower())
        words = [word for word in tokens if (word.isalpha() and not word in STOPWORDS)]
        sentence = (" ").join(words)
        X_train_clean.append(sentence)
    return pd.Series(X_train_clean)

def main():
    ''' load train, val, and test data '''
    train = pd.read_csv('train.csv')
    X_train = preprocess(train['Text'])
    y_train = train['Verdict']
    model = None # TODO: Define model

    train_model(model, X_train, y_train)
    # test your model
    y_pred = predict(model, X_train)

    # Use f1-macro as the metric
    score = f1_score(y_train, y_pred, average='macro')
    print('score on validation = {}'.format(score))

    # generate prediction on test data
    test = pd.read_csv('test.csv')
    X_test = test['Text']
    y_pred = predict(model, X_test)
    generate_result(test, y_pred, _STUDENT_NUM + ".csv")

# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()
