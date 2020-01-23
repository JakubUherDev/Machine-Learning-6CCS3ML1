

# The Confusion Matrix is:
#                  |              ACTUAL
#                  |      YES              NO
#            YES   | True Positive     False Positive
# PREDICTED        |
#            NO    | False Negative    True Negative

class ConfusionMatrix( object ):

    def getMatrix(self, test_data, target_data, tree, prediction):

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for i in range(len(test_data)):

            pred = []
            if len(prediction) >= 1:
                pred = prediction
                j = i
            else:
                pred = tree.predict(test_data[i].reshape(1, -1))

            if pred[j] == 1 and target_data[i] == 1:  # True positive
                tp += 1
            if pred[j] == 1 and target_data[i] == 0:  # False positive
                fp += 1
            if pred[j] == 0 and target_data[i] == 1:  # False negative
                fn += 1
            if pred[j] == 0 and target_data[i] == 0:  # True negative
                tn += 1

        r = 2
        cMatrix = [[0 for x in range(r)] for y in range(r)]
        cMatrix[0][0] = tp
        cMatrix[0][1] = fp
        cMatrix[1][0] = fn
        cMatrix[1][1] = tn

        return (cMatrix)