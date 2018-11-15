import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""Use of KFold cross validation technique to split x& y datasets into train/test folds & 
calculate accuracy using cross_val_score for multiple algorithms to pick up the most accurate algo."""

# define column names for the columns
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

# import the dataset
irisdataset = pd.read_csv("iris.csv", names=colnames)

# create an array of dependent(y) and independent variables(x) from the dataset
x = irisdataset.iloc[:, :4].values
y = irisdataset.iloc[:, 4].values

"""# take a look at how many rows and columns are thr in ur dataset
print(irisdataset.shape)

# take a look at the first 20 rows
print(irisdataset.head(20))

# take a look at the last 20 rows
print(irisdataset.tail(30))

# take a look at the summary of each attribute in the dataset(will give u the mean, max, min, std dev, etc)
print(irisdataset.describe())

# Class distribution: take a look at how many rows belong to one class
print(irisdataset.groupby('class').size())

# plot histogram to get an idea of the data distribution
irisdataset.hist()
plt.show()
"""
# Split the dataset into test & training sets using kFold technique

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=0)
for x_train, x_test in kf.split(x):
    print("Input Training Set:", x[x_train], "Input Test Set:", x[x_test])
# If u want to check the splits created for ur dataset, use below code:
"""
for y_train, y_test in kf.split(y):
 #   print("Output Training Set:", y[y_train], "Output Test Set:", y[y_test])
"""

# Get the cross validation score for SVM model
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
svm_class = SVC()
cvscore = cross_val_score(svm_class, x, y, cv=kf)
# we pass the entire x & y datasets here along with the class instantiated for SVC.
# we are not training hte model yet,
print("Accuracy using 3Fold for SVM is:", cvscore)
print("Thus the avg accuracy of SVM model is:", cvscore.mean()*100)

# To find out corss valdn score for KNN model
from sklearn.neighbors import KNeighborsClassifier
knn_class = KNeighborsClassifier()
knn_cvscore = cross_val_score(knn_class, x, y, cv=kf)
print("Accuracy using 3Fold for KNN is:", knn_cvscore)
print("Thus the avg accuracy of KNN model is:", knn_cvscore.mean()*100)

# NOw choose a model(SVM or KNN) which has higher accuracy, here it is SVM.
# Fit the model using SVM algorithm

from sklearn.model_selection import cross_val_predict
svmmodel = SVC(kernel= 'linear', random_state=0)
svmmodel.fit(x, y)

# Predict the values using cross val prediction
# Cross_val_predict will give u prediction of all folds.
predictions = cross_val_predict(svmmodel, x, y, cv=kf)
print("Predictions using cross validation predictor for all folds are: ", predictions)
