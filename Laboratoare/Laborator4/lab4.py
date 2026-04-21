import numpy as np
from sklearn import preprocessing, svm, metrics

train_sentences=np.load('training_sentences.npy', allow_pickle=True)
train_labels=np.load('training_labels.npy', allow_pickle=True)
test_sentences=np.load('test_sentences.npy', allow_pickle=True)
test_labels=np.load('test_labels.npy', allow_pickle=True)

def normalize_data(train_data, test_data, type=None):
    if type=='standard':
        scaler=preprocessing.StandardScaler()
    elif type in ['l1', 'l2']:
        scaler=preprocessing.Normalizer(norm=type)
    else:
        return train_data, test_data

    scaler.fit(train_data)
    return scaler.transform(train_data), scaler.transform(test_data)

class BagOfWords:
    def __init__(self):
        self.vocab={}
        self.words=[]

    def build_vocabulary(self, data):
        for doc in data:
            for word in doc:
                if word not in self.vocab:
                    self.vocab[word]=len(self.vocab)
                    self.words.append(word)
        print(len(self.vocab))

    def get_features(self, data):
        features = np.zeros((len(data), len(self.vocab)))
        for i, doc in enumerate(data):
            for word in doc:
                if word in self.vocab:
                    features[i, self.vocab[word]] += 1
        return features

bow=BagOfWords()
bow.build_vocabulary(train_sentences)

x_train=bow.get_features(train_sentences)
x_test=bow.get_features(test_sentences)

x_train_norm, x_test_norm = normalize_data(x_train, x_test, type='l2')

model=svm.SVC(C=1.0, kernel='linear')
model.fit(x_train_norm, train_labels)

predictii=model.predict(x_test_norm)
print("acuratete:", metrics.accuracy_score(test_labels, predictii))
print("f1-score:", metrics.f1_score(test_labels, predictii))

coeficienti=model.coef_[0]
indici_sortati=np.argsort(coeficienti)

cuvinte_negative = np.array(bow.words)[indici_sortati[:10]]
cuvinte_pozitive = np.array(bow.words)[indici_sortati[-10:]]

print(cuvinte_negative)
print(cuvinte_pozitive)