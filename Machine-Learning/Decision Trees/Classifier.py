import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

Coffee = pd.read_csv('coffee.csv')

# Filter out the coffee that is missing scores
Coffee = Coffee[Coffee['total_cup_points']!=0]
Coffee.shape

# Summarize the categorical features
Coffee.describe(include=['O'])
X = Coffee[['aroma', 'flavor', 'aftertaste', 'acidity', 'body', 'balance', 'uniformity', 'clean_cup', 'cupper_points', 'sweetness']]
y = Coffee['species']

# Create training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Initialize the tree and fit the tree
DTC = DecisionTreeClassifier()
#DTC = DecisionTreeClassifier(max_depth=5)
#DTC = DecisionTreeClassifier(max_samples_split=4)
#DTC = DecisionTreeClassifier(min_samples_leaf=3)
#DTC = DecisionTreeClassifier(max_leaf_nodes=12)

DTC.fit(X_train, y_train)
DTC.get_depth()

plot_tree(DTC, feature_names=X_train.columns)

y_pred = DTC.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
ConfusionMatrixDisplay(cm).plot()

DTC.predict_proba(X_train)



### Predicting mpg

mpg = pd.read_csv('mpg.csv')
plt.rcParams['figure.dpi'] = 150

# Create a dataframe X containing cylinders, weight, and mpg
X = mpg[["cylinders", "weight", "mpg"]]

# Create a dataframe y containing origin
y = mpg["origin"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Initialize the tree with `max_leaf_nodes=6`
DTC = DecisionTreeClassifier(max_leaf_nodes=6)

# Fit the tree to the training data
DTC.fit(X_train, y_train)

# Print the text summary of the tree
DTC_tree = export_text(DTC)
print(DTC_tree)

# Make predictions for the test data
y_pred = DTC.predict(X_test)
 
# Create a confusion matrix
cm = confusion_matrix(y_test,y_pred)

# Plot the confusion matrix
ConfusionMatrixDisplay(cm).plot()
plt.savefig('confMatrix.png')
