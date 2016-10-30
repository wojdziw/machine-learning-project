from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
import numpy as np


class TwoStagePredictor:
    def __init__(self, nulls_policy='max_from_regr', regr_margin=0.2):
        self.regressor = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
        self.classifier = MLPClassifier()
        assert nulls_policy in ['return_nothing', 'max_from_regr'], "Nulls policy must be either return_nothing or max_from_regr"
        self.nulls_policy = nulls_policy
        assert regr_margin > 0 and regr_margin <= 1, "You must insure 0 < regr_margin <= 1"
        self.regr_margin = regr_margin

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
        outLabels = np.array(output)
        # Make sure that if we predict 'nothing' we don't predict sth else
        # and that if we don't predict anything, we give nothing as an ans
        nulls = 0
        nc = 0
        for i in range(output.shape[0]):
            predictedSth = True
            if np.sum(output[i]) == 0:
                predictedSth = False
                nulls += 1
            if output[i, 0] == 1:
                output[i] = np.zeros(output.shape[1])
                output[i, 0] = 1
            if not predictedSth:
                if self.nulls_policy == 'max_from_regr':
                    maxCoef = np.max(regresssionOutput[i])
                    nothingHasMax = False
                    for j, c in enumerate(regresssionOutput[i]):
                        if j == 0 and c == maxCoef:
                            output[i, j] = 1
                            nothingHasMax = True
                            nc += 1
                        if j != 0 and maxCoef - c < self.regr_margin * maxCoef and not nothingHasMax:
                            output[i, j] = 1
                else: # policy is return_nothing
                    output[i, 0] = 1

        print("TSP Nulls: " + str(nulls) + " regr_nulls: " + str(nc))
        return output
