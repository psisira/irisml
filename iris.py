import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""Prob statement: Using SVM algo, to predict the type of iris flower based on sepal-length', 'sepal-width', 'petal-length', 
and 'petal-width' attributes"""

# define column names for the columns
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

# import the dataset
irisdataset = pd.read_csv("iris.csv", names=colnames)

# create an array of dependent(y) and independent variables(x) from the dataset
x = irisdataset.iloc[:, :4].values
y = irisdataset.iloc[:, 4].values

# take a look at how many rows and columns are thr in ur dataset
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

# Split the dataset into test & training sets
#     Random_state is used to initialize random number generator. Specifiy either 0 or 1,
#     It will determine how to split the data into test and train sets.
#     It will make sure we get the same split of data everytime we run the code.
#     If nothing is specified, we will get diff splits everytime we run the code,
#     which will make it difficult to compare the results.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)

# Fit the model using SVM algorithm
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(x_train, y_train)

# Predicting the results for entire test set (x_test)
# y_pred = classifier.predict(x_test)
# print(y_pred)

"""If uwant to predict results for a given data set i.e. for specific sepal-length', 'sepal-width', 
'petal-length' and 'petal-width' attributes """
new_test_values = [[5.9, 3, 5.1, 1.8]]
z_pred = classifier.predict(new_test_values)
print("Type of iris flower for given petal and sepal lengths(new_test_values) is :", z_pred)