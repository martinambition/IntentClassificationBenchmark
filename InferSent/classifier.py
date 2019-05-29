import numpy as np
import torch
from .models import InferSent
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import xgboost as xgb
import os

class InferSentClassifier:
    def __init__(self):
        model_version = 1
        MODEL_PATH = os.path.join(os.path.dirname(__file__), "infersent%s.pkl" % model_version)
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
        self.model = InferSent(params_model)
        self.model.load_state_dict(torch.load(MODEL_PATH))
        W2V_PATH = '/Users/i303138/Documents/Learning/InferSent-master/dataset/GloVe/glove.840B.300d.txt' #if model_version == 1 else '../dataset/fastText/crawl-300d-2M.vec'
        self.model.set_w2v_path(W2V_PATH)
        self.model.build_vocab_k_words(K=100000)


    def train(self,texts,labels):
        X = self.model.encode(texts, bsize=128, tokenize=False, verbose=True)
        tuned_parameters = {'kernel': ['linear'], 'C': [1, 10]}
        self.clf = GridSearchCV(SVC(C=1, probability=True, class_weight="balanced"),
                                param_grid=tuned_parameters,
                                n_jobs=1,
                                cv=KFold(n_splits=5, shuffle=True),
                                scoring='accuracy')
        self.le = LabelEncoder()
        Y = self.le.fit_transform(labels)
        self.clf.fit(X, Y)

    def eval(self,texts, labels):
        X = self.model.encode(texts, bsize=128, tokenize=False, verbose=True)
        Y = self.le.transform(labels)
        from sklearn.metrics import accuracy_score
        return accuracy_score(self.clf.predict(X),Y)

class XGBoostClassifier(InferSentClassifier):
    def __init__(self):
        InferSentClassifier.__init__(self)

    def train(self, texts, labels):
        X = self.model.encode(texts, bsize=128, tokenize=False, verbose=True)
        xgb_model = xgb.XGBClassifier()
        self.clf = GridSearchCV(xgb_model,
                           {'max_depth': [2, 4, 6],
                            'n_estimators': [50, 100]},n_jobs=1,cv=KFold(n_splits=5, shuffle=True),verbose=1)

        self.le = LabelEncoder()
        Y = self.le.fit_transform(labels)
        self.clf.fit(X, Y)

    def eval(self, texts, labels):
        X = self.model.encode(texts, bsize=128, tokenize=False, verbose=True)
        Y = self.le.transform(labels)
        from sklearn.metrics import accuracy_score
        return accuracy_score(self.clf.predict(X),Y)

