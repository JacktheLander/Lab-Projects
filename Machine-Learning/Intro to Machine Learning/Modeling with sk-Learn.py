# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Load the tips dataset
tips = pd.read_csv('tips.csv')

# Create a dataframe X containing total_bill and size
X = tips[['total_bill', 'size']]

# Create a dataframe y containing day
y = tips[['day']]

# Flatten y into an array
yNew = np.ravel(y)

# Initialize a GaussianNB() model
GNBModel = GaussianNB()

# Fit the model to X and yNew
GNBModel.fit(X, yNew)

# Determine the accuracy of the model GNBModel
accuracy = GNBModel.score(X, yNew)# Your code goes here

print(accuracy)
