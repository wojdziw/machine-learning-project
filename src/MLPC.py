from sklearn.neural_network import MLPClassifier
import numpy as np

class MLPC:
    def __init__(self):
        self.classifier = MLPClassifier(hidden_layer_sizes=(60, 60))

    def fit(self, X, y):
        self.classifier.fit(X, y)


    def predict(self, X):
        output = self.classifier.predict(X)
        np.save("MLPC_out", output)
        # Make sure that if we predict 'nothing' we don't predict sth else
        # If we don't predict anything, still predict 'nothing'
        # for i in range(output.shape[0]):
        #     if output[i, 0] == 1:
        #         output[i] = np.zeros(output.shape[1])
        #         output[i, 0] = 1
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
        print("MLCP nulls: " + str(nulls))
        return output
