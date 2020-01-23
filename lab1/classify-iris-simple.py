import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_iris

# plot step
# Spacing between values.
# For any output out, this is the distance between two adjacent values,
# out[i+1] - out[i].
# The default step size is 1. If step is specified as a position argument, start must also be given.
# so separation between the range if range is 0,1 and plot_step = 0.2 it will be 0, 0.2, 0.4, 0.5
plot_step = 0.02
#
# # Loads the data
iris = load_iris()
# x stores only two features out of all.
X = iris.data [:, [1, 3]] # 1 and 3 are the features we will use here.
# y stores the class labels, in here its 0, 1, 2 setosa, verisculor, virginica
y = iris.target
#
# iris_tree stpres a decision tree classifier which criterion is entropy.
# fit(): method does the learning, decision tree learns how to classify data.
iris_tree = tree.DecisionTreeClassifier(criterion="entropy").fit(X, y)

# find min and max of first feature
# find min and max of second feature
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# Return coordinate matrices from min/max of features as coordinate vectors.
# arange() method is equivilent to a range function in python with a difference
# of returning the ndArray instead of list()
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))


# ravel() method flattens the array. eg [1,2], [3,4] = [1,2,3,4]
# np_c() method slice object concatenation along the sexond axis. eg [1, 2, 3] and [4, 5, 6] = [1, 4], [2, 5], [3, 6]
# Here we use the tree
# to predict the classification of boundaries
Z = iris_tree.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

# plots the data on the axis.
plt.scatter(X[:, 0], X[:, 1], c=y.astype(np.float))

# Label axes
# x axis with sepal width
plt.xlabel( iris.feature_names[1], fontsize=10 )
# y axis with petal width
plt.ylabel( iris.feature_names[3], fontsize=10 )

# show graph
plt.show()


