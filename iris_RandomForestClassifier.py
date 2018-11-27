import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""Prob statement: Using RandomForestClassifier algo, to predict the type of iris flower based on sepal-length', 'sepal-width', 'petal-length', 
and 'petal-width' attributes"""

# define column names for the columns
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

# import the dataset into a dataframe
irisdataset = pd.read_csv("iris.csv", names=colnames)

# create an array of dependent(y) and independent variables(x) from the dataset
x = irisdataset.iloc[:, :4]  # creates a dataframe
y = irisdataset.iloc[:, 4].values  # creates an array

# Split the dataset into test & training sets

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)

# Get col names from x_train dataframe using x_train.columns
x_train_cols = x_train.columns

#Get indices from x_train
ind = x_train.index
print("x indices are:", ind)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
x_train = sc_x.fit_transform(x_train) # Here we lose the col headers & x_train becomes an array
x_test = sc_x.transform(x_test)

# convert x_train to a df - index will be the indices we created above(ind)
# and column headers will be the col headers we extracted above (x_train_cols)
# We are converting x_train into a df so that it can be used for feature_importances_ later in the code.

x_train_df = pd.DataFrame(x_train, index=ind, columns=x_train_cols)

# Fit the RandomForestClassifier model to ur training dataset
from sklearn.ensemble import RandomForestClassifier
forestclassifier = RandomForestClassifier(n_estimators=100, random_state=0)
x = forestclassifier.fit(x_train, y_train)

# Predicting the results for entire test set (x_test)
y_pred = forestclassifier.predict(x_test)
print("Predicted types of flowers are:", y_pred)

# Feature importance to identify which variables have more effect on ur model
#feature_importances = pd.DataFrame(forestclassifier.feature_importances_, index=ind, columns=['importance']).sort_values('importance', ascending=False)
#print("Features/variables based on importance are:", feature_importances)
# list(zip) with feature_importances_ will help in displaying the features with
#  importance score along with the colmn headers of the features.
# Using only say imp = forestclassifier.feature_importances_ will display the features but without the col headers.
imp = list(zip(x_train_df, forestclassifier.feature_importances_))
print("Features importance is:", imp)

# CODE TO EXTRACT TREES FROM RANDOM FOREST IS BELOW (NOT WORKING)
"""
# Pull out one tree from the forest or get all the trees of the forest??
estimators = forestclassifier.estimators_[5]
print("Trees of the forest are:", estimators)

# Create feature_list for use later in export graphviz, this will just have the names of the columns
# To get col names from ndarray, use dytpe.names
# features_list = x.dtype.names
# To get col names from dataframe, use below.
# features_list = list(irisdataset.iloc[:, :4])


# Export the trees as an image in a dot file
from sklearn.tree import export_graphviz
import pydot
import pydotplus
import graphviz
from IPython.display import Image
dot_data = export_graphviz(estimators, out_file='iristree.dot', feature_names=features_list, class_names=True, rounded=True)

# Use dot file to create a graph
#pydot.graph_from_dot_file loads the mentioned dot file to get a pydot.Dot class instance
(graph, ) = pydot.graph_from_dot_file('iristree.dot')

# Then Write graph to a png file
graph.write_png('iristree.png')
"""
"""
#Another way to get graph:
#Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)
#Show graph
Image(graph.create_png())

#Another way to get graph
graph = graphviz.Source(dot_data)
print("tree is", graph)
"""

"""
# To predict type of flower for given sepal/petal dimensions.

# new_test_values = [[5.9, 3, 5.1, 1.8]]
# z_pred = classifier.predict(new_test_values)
# print("Type of iris flower for given petal and sepal lengths(new_test_values) is :", z_pred)
"""
"""
# Make confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate accuracy score of the model
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print("Accuracy Score using RandomForestClassifier is:", ac)

"""


