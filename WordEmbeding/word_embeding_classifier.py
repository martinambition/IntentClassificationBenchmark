import pandas as pd
import csv
import itertools
# from gensim.parsing.preprocessing import preprocess_string,stem_text,remove_stopwords
from gensim.utils import tokenize
from gensim.models import KeyedVectors
from collections import OrderedDict
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import numpy as np
import os
import random

class WordEmebdingClassifier:
    def __init__(self):
        self.model = KeyedVectors.load_word2vec_format(os.path.join(os.path.abspath(''), 'glove.6B.200d.w2vformat.txt'))

    def doc2vec(self,sentences):
        sentence_vec = []
        for index, t in enumerate(sentences):
            temp = [];
            for key in tokenize(t,lowercase=True):
                if key in self.model.wv:
                    temp.append(self.model[key])
            sentence_vec.append(np.vstack(temp))
        X = np.stack([np.sum(vec, axis=0) / vec.shape[0] for vec in sentence_vec])
        return X
    def train(self,texts,labels):
        X =  self.doc2vec(texts)
        tuned_parameters = {'kernel': ['linear'], 'C': [1, 10]}
        self.clf = GridSearchCV(SVC(C=1, probability=True, class_weight="balanced"),
                           param_grid=tuned_parameters,
                           n_jobs=1,
                           cv=KFold(n_splits=5, shuffle=True),
                           scoring='accuracy')
        self.le = LabelEncoder()
        Y = self.le.fit_transform(labels)
        self.clf.fit(X,Y)

    def eval(self,texts, labels):
        X = self.doc2vec(texts)
        Y = self.le.transform(labels)
        from sklearn.metrics import accuracy_score
        return accuracy_score(self.clf.predict(X),Y)