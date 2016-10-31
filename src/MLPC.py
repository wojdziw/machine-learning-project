from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
import numpy as np

class MLPC:
    def __init__(self, act = 'relu', a = 0.0001, hls = (100,)):
        self.classifier  = MLPClassifier(activation=act, hidden_layer_sizes=hls, alpha=a, random_state=1)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        self.scaler.fit(X)
        self.classifier.fit(self.scaler.transform(X), y)
        # self.classifier.fit(X, y)


    def predict(self, X):
        output = self.classifier.predict(self.scaler.transform(X))
        probs = self.classifier.predict_proba(self.scaler.transform(X))
        # output = self.classifier.predict(X)
        nulls = 0
        for i in range(output.shape[0]):
            predictedSth = True
            if np.sum(output[i]) == 0:
                predictedSth = False
                nulls += 1
            if output[i, 0] == 1:
                output[i] = np.zeros(output.shape[1])
                output[i, 0] = 1
            if not predictedSth:
                p = probs[i]
                maxCoef = np.max(probs[i])
                predictedNth = False
                for j in range(probs.shape[1]):
                    if probs[i, j] == maxCoef and not predictedNth:
                        if j == 0:
                            predictedNth = True
                        output[i, j] = 1
        print("MLCP nulls: " + str(nulls))
        return output
