from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
import numpy as np

class TwoStagePredictor:
    def __init__(self):
        self.regressor = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
        self.classifier = MLPClassifier()

    def fit(self, X, y):
        ratio = 3
        N, _ = X.shape
        X_1 = X[int(N / ratio):, :]
        X_2 = X[:int(N / ratio), :]
        y_1 = y[int(N / ratio):, :]
        y_2 = y[:int(N / ratio), :]
        self.regressor.fit(X_1, y_1)
        regresssionOutput = self.regressor.predict(X_2)
        self.classifier.fit(regresssionOutput, y_2)

    def predict(self, X):
        regresssionOutput = self.regressor.predict(X)
        output = self.classifier.predict(regresssionOutput)
        np.save("TS_out", output)
        outLabels = np.array(output)
        # Make sure that if we predict 'nothing' we don't predict sth else
        # and that if we don't predict anything, we give nothing as an ans
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
                output[i, 0] = 1
        print("TSP Nulls: " + str(nulls))
        return output
