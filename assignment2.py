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
from sklearn.metrics import f1_score, confusion_matrix

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

# nltk.download('punkt')
# nltk.download('stopwords')
STOPWORDS = stopwords.words('english')
NUM_FEATURES = 500
_STUDENT_NUM = 'A0168932U'
vocab = {} # TODO: Settle the OOV/invalid word token.
word_counter = 1
dic_y_mapping = {}
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE

def train_model(model, X_train, y_train):
    ''' TODO: train your model based on the training data '''
    global dic_y_mapping
    sm = SMOTE(random_state=42)
    X_smote, y_smote = sm.fit_resample(X_train, y_train)
    dic_y_mapping = {n:label for n,label in 
                 enumerate(np.unique(y_smote))}
    inverse_dic = {v:k for k,v in dic_y_mapping.items()}
    y_smote = np.array([inverse_dic[y] for y in y_smote])
    # class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    ## train
    training = model.fit(x=X_smote, y=y_smote, batch_size=256, 
                     epochs=10, shuffle=True, verbose=0, 
                     validation_split=0.3)


def predict(model, X_test):
    ''' TODO: make predictions here '''
    global dic_y_mapping
    prob = model.predict(X_test)
    return [dic_y_mapping[np.argmax(pred)] for pred in 
             prob]

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

def feat_eng(X):
    unigram = []
    for sentence in X:
        lst_words = sentence.split()
        lst_grams = [" ".join(lst_words[i:i+1]) 
                    for i in range(0, len(lst_words), 1)]
        unigram.append(lst_grams)

    ## detect bigrams and trigrams
    bigrams_detector = gensim.models.phrases.Phrases(unigram, 
                    delimiter=" ".encode(), min_count=5, threshold=10)
    bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
    trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[unigram], 
                delimiter=" ".encode(), min_count=5, threshold=10)
    trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)
    return [unigram, bigrams_detector, trigrams_detector]

def main():
    ''' load train, val, and test data '''
    train = pd.read_csv('train.csv')
    corpus = preprocess(train['Text'])
    ngrams = feat_eng(corpus)
   
    # Using Skip-Gram
    word_vect = gensim.models.word2vec.Word2Vec(ngrams[0], size=NUM_FEATURES,   
            window=8, min_count=1, sg=1, iter=30)
    
        ## tokenize text
    tokenizer = kprocess.text.Tokenizer(lower=True, split=' ', 
                        oov_token="NaN", 
                        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(ngrams[0])
    vocab = tokenizer.word_index
    # print('vocab size is {}', len(vocab))
    ## create sequence
    lst_text2seq = tokenizer.texts_to_sequences(ngrams[0])
    ## padding sequence
    X_train = kprocess.sequence.pad_sequences(lst_text2seq, 
                        maxlen=80, padding="post", truncating="post")
    y_train = train['Verdict']
    
    ## start the matrix (length of vocabulary x vector size) with all 0s
    embeddings = np.zeros((len(vocab)+1, NUM_FEATURES+1))
    from sentifish import Sentiment
    for word,idx in vocab.items():
        ## update the row with vector
        try:
            embeddings[idx][0:-1] =  word_vect[word]
            embeddings[idx][-1] = Sentiment(word).analyze()
        ## if word not in model then skip and the row stays all 0s
        except:
            pass

    x_input = layers.Input(shape=(80,))
    x = layers.Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1],weights=[embeddings],input_length=80, trainable = False)(x_input)
    # x = layers.Dense(6, activation='relu')(x)
    x = layers.Flatten()(x)
    y_out = layers.Dense(3, activation='softmax')(x)
    model = models.Model(x_input, y_out)

    model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

    model.summary()

    train_model(model, X_train, y_train)
    # test your model
    y_pred = predict(model, X_train)

    # Use f1-macro as the metric
    score = f1_score(y_train, y_pred, average='macro')
    print('score on validation = {}'.format(score))
    print(confusion_matrix(y_train, y_pred))
    # generate prediction on test data
    test = pd.read_csv('test.csv')
    ngram_test = feat_eng(preprocess(test['Text']))
    ## create sequence
    text2seq_test = tokenizer.texts_to_sequences(ngram_test[0])
    ## padding sequence
    X_test = kprocess.sequence.pad_sequences(text2seq_test, 
                        maxlen=80, padding="post", truncating="post")
    y_pred = predict(model, X_test)
    generate_result(test, y_pred, _STUDENT_NUM + ".csv")

# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()
