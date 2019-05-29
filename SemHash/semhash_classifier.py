import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

class SemHashClassfier:
    def __init__(self):
        self.N = 1000  # set the desired dimensionality of HD vectors
        self.n_size = 3  # n-gram size
        self.aphabet = 'abcdefghijklmnopqrstuvwxyz#'  # fix the alphabet. Note, we assume that capital letters are not in use
        np.random.seed(1)  # for reproducibility
        self.HD_aphabet = 2 * (np.random.randn(len(self.aphabet), self.N) < 0) - 1  # generates bipolar {-1, +1}^N HD vectors; one random HD vector per symbol in the alphabet

    def find_ngrams(self,input_list, n):
        return zip(*[input_list[i:] for i in range(n)])

    def semhash_tokenizer(self,text):
        tokens = text.split(" ")
        final_tokens = []
        for unhashed_token in tokens:
            hashed_token = "#{}#".format(unhashed_token)
            final_tokens += [''.join(gram)
                             for gram in list(self.find_ngrams(list(hashed_token), 3))]
        return final_tokens

    def semhash_corpus(self,corpus):
        new_corpus = []
        for sentence in corpus:
            tokens = self.semhash_tokenizer(sentence)
            new_corpus.append(" ".join(map(str, tokens)))
        return new_corpus

    def ngram_encode(self,str_test, HD_aphabet, aphabet,
                     n_size):  # method for mapping n-gram statistics of a word to an N-dimensional HD vector
        HD_ngram = np.zeros(HD_aphabet.shape[1])  # will store n-gram statistics mapped to HD vector
        full_str = '#' + str_test + '#'  # include extra symbols to the string

        for il, l in enumerate(full_str[:-(n_size - 1)]):  # loops through all n-grams
            hdgram = HD_aphabet[aphabet.find(full_str[il]),
                     :]  # picks HD vector for the first symbol in the current n-gram
            for ng in range(1, n_size):  # loops through the rest of symbols in the current n-gram
                hdgram = hdgram * np.roll(HD_aphabet[aphabet.find(full_str[il + ng]), :],
                                          ng)  # two operations simultaneously; binding via elementvise multiplication; rotation via cyclic shift

            HD_ngram += hdgram  # increments HD vector of n-gram statistics with the HD vector for the currently observed n-gram

        HD_ngram_norm = np.sqrt(HD_aphabet.shape[1]) * (
        HD_ngram / np.linalg.norm(HD_ngram))  # normalizes HD-vector so that its norm equals sqrt(N)
        return HD_ngram_norm  # output normalized HD mapping


    def doc2vec(self,sentencs):
        hash_sentences = self.semhash_corpus(sentencs)
        vectors = []
        for i in range(len(sentencs)):
            vectors.append( self.ngram_encode(sentencs[i], self.HD_aphabet, self.aphabet,self.n_size))

        X = np.stack(vectors)
        return X;

    def train(self,texts,labels):
        X =  self.doc2vec(texts)
        #tuned_parameters = {'kernel': ['linear'], 'C': [1, 10]}

        tuned_parameters = {
            'n_estimators': [40, 200, 340],
            'max_depth': [8, 9, 10, 11, 12],
            'random_state': [0],
            # 'max_features': ['auto'],
            # 'criterion' :['gini']
        }
        self.clf = GridSearchCV(RandomForestClassifier(),
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