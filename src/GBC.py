import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier

class GBC:
    def __init__(self):
        # self.classifier = MultiOutputClassifier(GradientBoostingClassifier())
        self.classifier = GradientBoostingClassifier()

    def fit(self, X, y):
        self.classifier.fit(X, y)

    def predict(self, X):
        output = self.classifier.predict(X)
        np.save("GBC_out", output)
        # Make sure that if we predict 'nothing' we don't predict sth else
        for i in range(output.shape[0]):
            if output[i, 0] == 1:
                output[i] = np.zeros(output.shape[1])
                output[i, 0] = 1
        return output
