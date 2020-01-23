import math
import os
import random
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

# get data
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import tree


class ConfusionMatrix( object ):

    def get_matrix(self, test_data, target_data, tree, prediction):
        # initialise counters
        tp = 0  # true positives
        tn = 0  # true negatives
        fp = 0  # false positive
        fn = 0  # false negatives

        # Now iterate for each of the values of the X_test sample data to compare the
        # prediction against the real value:

        for i in range(len(test_data)):
            pred = []
            if len(prediction) >= 1:
                pred = prediction
                j = i
            else:
                pred = tree.predict(test_data[i].reshape(1, -1))
                j = 0

            if pred[j] == 1 and target_data[i] == 1:  # True positive
                tp += 1
            if pred[j] == 1 and target_data[i] == 0:  # False positive
                fp += 1
            if pred[j] == 0 and target_data[i] == 1:  # False negative
                fn += 1
            if pred[j] == 0 and target_data[i] == 0:  # True negative
                tn += 1

        s = 2
        cMatrix = [[0 for x in range(s)] for y in range(s)]
        cMatrix[0][0] = tp
        cMatrix[0][1] = fp
        cMatrix[1][0] = fn
        cMatrix[1][1] = tn
        return (cMatrix)

    def calculatePrecision(self, c_matrix):
        # precision: tp / tp + fp
        tp = c_matrix[0][0]
        fp = c_matrix[0][1]
        dividend = tp
        divisor = tp + fp
        return dividend / divisor

    def calculateRecall(self, c_matrix):
        # recall: tp / tp + fn
        tp = c_matrix[0][0]
        fn = c_matrix[1][0]
        dividend = tp
        divisor = tp + fn
        return dividend / divisor

    def calculate_f1_score(self, c_matrix, f_score):
        precision = self.calculatePrecision(c_matrix)
        recall = self.calculateRecall(c_matrix)
        dividend = precision * recall
        divisor = precision + recall
        return f_score * (dividend / divisor)


# Create function for the statistics:
def getStats ( data, mode):
    # MEAN, STD:
    meanS = (1/len(data))*sum(data)
    stdDS = math.sqrt( (1/(len(data)-1))*sum([pow(i-meanS,2) for i in data]) )
    if mode == 1:
       statS = meanS
    elif mode == 2:
       statS = stdDS
    elif mode == 0:
       statS = [meanS, stdDS]
    return ( statS )


class crossVal(object):
    def getTrainTest(self, X, y, testRatio):
        # In theory,both X and y should have the same length
        totalRecords = len(X)
        # Estimate the records corresponding to testRatio
        totalTestRecords = math.ceil(totalRecords / (100 * testRatio))
        # Now generate an array of random numbers
        testLines = random.sample(range(0, int(totalRecords)), int(totalTestRecords))
        # Now find all those lines not in totalRecords
        allLines = range(0, totalRecords)
        trainLines = set(allLines) - set(testLines)
        # Finally, sepparate the data into Train and Test samples:
        X_train = X[list(trainLines), :]
        X_test = X[testLines, :]
        y_train = y[list(trainLines)]
        y_test = y[testLines]
        return (X_train, X_test, y_train, y_test)


bc = load_breast_cancer()

# set x to all features of the data set
X = bc.data[:,:]
y = bc.target



# hold-out cross validation
# randomly splits data into 80% train set and 20% test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


# decision tree learns how to classify data with the training data set
bc_tree = tree.DecisionTreeClassifier(criterion="entropy").fit(X_train, y_train)

# get the score of decision tree on the test data set
score = bc_tree.score(X_test, y_test)

with open("/Users/uhel/year3/ML1/labs/lab_1/results.txt", "a+") as f:
    f.write("\n3.1: Accuracy (score) on X_test and y_test data: %0.4f\n" % (score))
    scores = []
    for i in range(10):
        X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=0.1, random_state=11)
        tmp_bc_tree = tree.DecisionTreeClassifier(criterion="entropy").fit(X_train_1, y_train_1)
        scores.append(tmp_bc_tree.score(X_test_1, y_test_1))

    f.write("\n3.2: After 10 runs of cross-validation, the average score and\n     standard deviation of the trees on the test data are:\n")
    f.write("     Avg. score: %.4f\n" % (getStats(scores, 1)))
    f.write("     Std. Dev. : %.4f\n" % (getStats(scores, 2)))
    f.write("     Accuracy: %.4f (+/- %0.4f)\n" % (getStats(scores, 1), getStats(scores, 2)))



    # 5-fold cross-validation
    cv_scores = cross_val_score(bc_tree, X, y, cv=5)
    # print(cv_scores)

    # 10-fold cross-validation
    cv_scores = cross_val_score(bc_tree, X, y, cv=10)
    f.write("\n3.3: After 10-fold cross-validation, the average score and\n     standard deviation of the trees on the test data are:\n")
    f.write("     Avg. score: %.4f\n" % (getStats(cv_scores, 1)))
    f.write("     Std. Dev. : %.4f\n" % (getStats(cv_scores, 2)))
    f.write("     Accuracy: %.4f (+/- %0.4f)\n" % (getStats(cv_scores, 1), getStats(cv_scores, 2)))
    # print(cv_scores)

    crossValidation = crossVal()
    scores = []
    for i in range(10):
        cV = crossValidation.getTrainTest(X, y, 0.1)  # 10% for test data
        tmpXtrain = cV[0]
        tmpXtest = cV[1]
        tmpYtrain = cV[2]
        tmpYtest = cV[3]
        tmp_bc_tree = tree.DecisionTreeClassifier(criterion="entropy").fit(tmpXtrain, tmpYtrain)
        scores.append(tmp_bc_tree.score(tmpXtest, tmpYtest))

    f.write("\n3.4: After 10-fold cross-validation, the average score and\n     standard deviation of the trees on the test data are:\n")
    f.write("     Avg. score: %.4f\n" % (getStats(scores, 1)))
    f.write("     Std. Dev. : %.4f\n" % (getStats(scores, 2)))
    f.write("     Accuracy: %.4f (+/- %0.4f)\n" % (getStats(scores, 1), getStats(scores, 2)))

    prediction = []
    t_tree = tree.DecisionTreeClassifier(criterion="entropy").fit(X_train, y_train)
    cm = ConfusionMatrix()
    M = cm.get_matrix(X_test, y_test, t_tree, prediction)
    tp = M[0][0]
    fp = M[0][1]
    fn = M[1][0]
    tn = M[1][1]
    # Now print the confusion matrix:
    f.write("\n4.1: The confusion matrix is:\n")
    f.write("                 |      ACTUAL\n")
    f.write("                 |   YES      NO\n")
    f.write("            YES  |   %d        %d\n" % (tp, fp))
    f.write(" PREDICTED       |\n")
    f.write("            NO   |   %d        %d\n" % (fn, tn))

    precision = cm.calculatePrecision(M)
    recall = cm.calculateRecall(M)
    f1_Score = cm.calculate_f1_score(M, 2)

    f.write("\n4.2: The precision is: %.4f\n" % precision)
    f.write("\n4.3: The recall is: %.4f\n" % recall)
    f.write("\n4.4: The F1 score is: %.4f\n" % f1_Score)

    prediction = []
    precisions = []
    recalls = []
    for i in range(10):
        X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=0.1, random_state=11)
        tmp_bc_tree = tree.DecisionTreeClassifier(criterion="entropy").fit(X_train_1, y_train_1)
        cm.get_matrix(X_test_1, y_test_1, tmp_bc_tree, prediction)
        precisions.append(cm.calculatePrecision(M))
        recalls.append(cm.calculateRecall(M))

    f.write("\n4.5: After 10-fold cross-validation, the results on the test\n     data are:\n")
    f.write("\n4.5: After 10-fold cross-validation, the results on the test\n     data are:\n")
    f.write("     Precision: %.4f (+/- %.4f)\n" % (getStats(precisions, 1), getStats(precisions, 2)))
    f.write("     Recall: %.4f (+/- %.4f)\n" % (getStats(recalls, 1), getStats(recalls, 2)))

    bc_neigh = KNeighborsClassifier(n_neighbors=5)
    bc_neigh.fit(X_train, y_train)
    p_prediction = bc_neigh.predict_proba(X_test)


    tp_list = []
    fp_list = []
    precisions = []
    recalls = []
    tree_tmp = []

    # check different range of thresholds
    for j in range(500):
        threshold = 0.01 * j
        t_predictions = []

        for i in range(len(X_test)):
            if p_prediction[i, 0] >= threshold:
                t_predictions.append(0)
            else:
                t_predictions.append(1)

        c_matrix = ConfusionMatrix()
        M = c_matrix.get_matrix(X_test, y_test, tree_tmp, t_predictions)

        tp_list.append(M[0][0])
        fp_list.append(M[0][1])
        if M[0][0] != 0 or M[0][1] != 0:
            precisions.append(c_matrix.calculatePrecision(M))
        else:
            precisions.append(0)

        if M[0][0] != 0 or M[1][0] != 0:
            recalls.append(c_matrix.calculateRecall(M))
        else:
            recalls.append(0)
    avgF1_score = 2 * ((getStats(precisions, 1) * getStats(recalls, 1)) / (getStats(precisions, 1) + getStats(recalls, 1)))

    f.write("\n4.6: The averaged precision, recall and F1 score are:\n")
    f.write("     Avg. Precision: %.4f (+/- %.4f)\n" % (getStats(precisions, 1), getStats(precisions, 2)))
    f.write("     Avg. Recall: %.4f (+/- %.4f)\n" % (getStats(recalls, 1), getStats(recalls, 2)))
    f.write("     Avg. F1 score: %.4f \n" % (avgF1_score))

    # Normalise the tp and fp lists in percentual terms:
    max_tp = max(tp_list)
    max_fp = max(fp_list)
    tp_list = [(100 * (i / max_tp)) for i in tp_list]
    fp_list = [(100 * (i / max_fp)) for i in fp_list]
    # The plot:
    plt.plot(fp_list, tp_list, '-o')
    plt.xlabel('False positives rate')
    plt.ylabel('True positives rate')
    plt.axis([-10, 110, -10, 110])
    plt.show()


# -------------------------------------------
#            5. NEAREST NEIGHBOUR
# -------------------------------------------

# 5.1: Run a 10-fold cross-validation on the k-nearest neighbour classifier applied to
#      the breast cancer dataset.

    precisions = []
    recalls = []
    F1_scores = []
    tree_tmp = []
    threshold = 0.8 # static threshold - depending on this value, the results will vary.
    f.write("\n5.1: The results for the 10-fold cross validation on the\nk-nearest neighbour are:\n")
    f.write("            Precision   Recall   F1 score\n")
    for i in range(10):
        # Cross-validation data sets:
        crossValidation = crossVal()
        cV = crossValidation.getTrainTest(X, y, 0.1) # 10% for test data
        Xtrain = cV[0]
        Xtest  = cV[1]
        Ytrain = cV[2]
        Ytest  = cV[3]

        # Use train and test samples produced in 3.2.
        # Get classification:
        bcf_neigh = KNeighborsClassifier(n_neighbors=5)
        bcf_neigh.fit(Xtrain, Ytrain)
        pf_prediction = bcf_neigh.predict_proba(Xtest)

        t_prediction = []
        # New prediction list:
        for j in range(len(Xtest)):
            if pf_prediction[j,0] >= threshold: # above the threshold
               t_prediction.append(0)
            else:                              # below the threshold
               t_prediction.append(1)

        # Get the confusion matrix elements:
        cM = []
        M = []
        cM = ConfusionMatrix() # create object
        M = cM.get_matrix( Xtest, Ytest, tree_tmp, t_prediction ) # get matrix
        # For the statistics:
        precision = (M[0][0] / (M[0][0] + M[0][1]))
        recall = (M[0][0] / (M[0][0] + M[1][0]))
        F1s = 2 * ((precision * recall)/(precision + recall))

        precisions.append(precision)
        recalls.append(recall)
        F1_scores.append(F1s)

        f.write("     Run %d:   %.4f    %.4f    %.4f\n" % (i, precision, recall, F1s))

    # 5.2: Compute the average precision, recall and F1 score across the 10-fold cross-
    #      validation. That is, compute precision, recall and F1 score for each fold, and
    #      compute the average and standard deviation over the 10 values

    f.write("\n5.2: The averaged results for the 10-fold cross validation on the\nk-nearest neighbour are:\n")
    f.write("     Avg. Precision: %.4f (+/- %.4f)\n" % (getStats(precisions,1), getStats(precisions,2)))
    f.write("     Avg. Recall: %.4f (+/- %.4f)\n" % (getStats(recalls,1), getStats(recalls,2)))
    f.write("     Avg. F1 score: %.4f (+/- %.4f) \n" % (getStats(F1_scores,1), getStats(F1_scores,2)))

    f.close()

