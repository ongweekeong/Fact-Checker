#!/usr/bin/env python.

"""
CS4248 ASSIGNMENT 2

"""


import pandas as pd
import numpy as np
import re

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline

from textblob import TextBlob

''' run python -m textblob.download_corpora '''

_STUDENT_NUM = 'A0168932U'

''' Helper class to vectorize other non-text features extracted in pipeline'''
class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key."""

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

''' Helper class to vectorize other non-text features extracted in pipeline'''
class ArrayCaster(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return np.transpose(np.matrix(data))


def add_features(data):
    def get_polarity(word):
        tb = TextBlob(word)
        return tb.polarity

    def get_subjectivity(word):
        tb = TextBlob(word)
        return tb.subjectivity

    def get_num_nouns(sent):
        tb = TextBlob(sent)
        return len(tb.noun_phrases)

    def get_num_adj(sent):
        tb = TextBlob(sent)
        count = 0
        for tag in tb.tags:
            if re.search(r"\bJJ", tag[1]):
                count += 1
        return count

    def get_num_adv(sent):
        tb = TextBlob(sent)
        count = 0
        for tag in tb.tags:
            if re.search(r"\bRB", tag[1]):
                count += 1
        return count

    def get_num_vb(sent):
        tb = TextBlob(sent)
        count = 0
        for tag in tb.tags:
            if re.search(r"\bVB", tag[1]):
                count += 1
        return count

    polarity = [get_polarity(sent) for sent in data.Text]
    subjectivity = [get_subjectivity(sent) for sent in data.Text]
    num_nouns = [get_num_nouns(sent) for sent in data.Text]
    num_adj = [get_num_adj(sent) for sent in data.Text]
    num_adv = [get_num_adv(sent) for sent in data.Text]
    num_vb = [get_num_vb(sent) for sent in data.Text]
    df = data.copy()
    df["Polarity"] = polarity
    df["Subjectivity"] = subjectivity
    df["NumNouns"] = num_nouns
    df["NumAdv"] = num_adv
    df["NumAdj"] = num_adj
    df["NumVb"] = num_vb
    return df

def generate_result(test, y_pred, filename):
    ''' generate csv file base on the y_pred '''
    test['Verdict'] = pd.Series(y_pred)
    test.drop(columns=['Text'], inplace=True)
    test.to_csv(filename, index=False)

def main():
    ''' load train, val, and test data '''
    train = pd.read_csv('train.csv')


    pipeline = Pipeline([
        ("features", FeatureUnion(
            transformer_list=[
                ("polarity", Pipeline([
                    ("selector", ItemSelector(key='Polarity')),
                            ("Caster", ArrayCaster())
                ])),
                ("subjectivity", Pipeline([
                    ("selector", ItemSelector(key='Subjectivity')),
                    ("Caster", ArrayCaster())
                ])),
                ("nouns", Pipeline([
                    ("selector", ItemSelector(key='NumNouns')),
                    ("Caster", ArrayCaster())
                ])),
                ("adj", Pipeline([
                    ("selector", ItemSelector(key='NumAdj')),
                    ("Caster", ArrayCaster())
                ])),
                ("adv", Pipeline([
                    ("selector", ItemSelector(key='NumAdv')),
                    ("Caster", ArrayCaster())
                ])),
                ("vb", Pipeline([
                    ("selector", ItemSelector(key='NumVb')),
                    ("Caster", ArrayCaster())
                ])),
                ("word_features", Pipeline([
                    ("selector", ItemSelector(key='Text')),
                    ('tfidf', TfidfVectorizer(stop_words='english')),
                ])),
            ]),
        ),
        ('scale', StandardScaler(with_mean=False)),
        ("classifier", LogisticRegression(multi_class='auto', class_weight='balanced',
                                max_iter=20000))
    #     ("classifier", MLPClassifier(early_stopping=True) )
    ])

    y_train = train['Verdict']

    params = {
        'classifier__C': [0.001],
        'classifier__solver': ['liblinear'],
        'features__word_features__tfidf__ngram_range': [(1,3)],
        'features__word_features__tfidf__max_features': [2000],
        'features__word_features__tfidf__min_df': [3],
        'features__word_features__tfidf__lowercase': [False],
        'features__word_features__tfidf__sublinear_tf': [False],
    }
    gs = GridSearchCV(pipeline, params, refit=True, cv=5, scoring='f1_macro', n_jobs=6, verbose=10)
    X = add_features(train)

    gs.fit(X, y_train)
    print(gs.best_score_, gs.best_params_)

    # test your model
    y_pred = gs.predict(X)

    # Use f1-macro as the metric
    score = f1_score(y_train, y_pred, average='macro')
    print('score on validation = {}'.format(score))
    print(confusion_matrix(y_train, y_pred))

    # generate prediction on test data
    test = pd.read_csv('test.csv') 
    y_pred = gs.predict(add_features(test))
    generate_result(test, y_pred, _STUDENT_NUM + ".csv")

# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()
