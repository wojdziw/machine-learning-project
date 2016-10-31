from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from validation import validation
import numpy as np
from utils import mltBinaryAccuracy

class MLPC(MLPClassifier):
    def __init__(self, hidden_layer_sizes=(100, ), activation='relu', solver='adam',
                alpha=0.0001, batch_size='auto', learning_rate='constant',
                learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
                random_state=None, tol=0.0001, verbose=False, warm_start=False,
                momentum=0.9, nesterovs_momentum=True, early_stopping=False,
                validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08):
        super().__init__(hidden_layer_sizes, activation, solver, alpha,
                        batch_size, learning_rate, learning_rate_init, power_t,
                        max_iter, shuffle, random_state, tol, verbose, warm_start,
                        momentum, nesterovs_momentum, early_stopping, validation_fraction,
                        beta_1, beta_2, epsilon)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        self.scaler.fit(X)
        return super().fit(self.scaler.transform(X), y)

    def predict(self, X):
        output = super().predict(self.scaler.transform(X))
        probs = super().predict_proba(self.scaler.transform(X))
        # nulls = 0
        for i in range(output.shape[0]):
            predictedSth = True
            if np.sum(output[i]) == 0:
                predictedSth = False
                # nulls += 1
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
        # print("MLCP nulls: " + str(nulls))
        return output

    def score(self, X, y):
        return mltBinaryAccuracy(self.predict(X), y)[0]

    def predict_proba(self, X, y):
        return super().predict_proba(self.scaler.transform(X))
