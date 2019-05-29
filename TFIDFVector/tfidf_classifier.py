from nltk.corpus import gutenberg
from gensim.utils import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

class TFIDFClassifer:
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        # Fit the TfIdf model
        self.tfidf.fit([gutenberg.raw(file_id) for file_id in gutenberg.fileids()])
    def doc2vec(self,sentences):
        return self.tfidf.transform(sentences)
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