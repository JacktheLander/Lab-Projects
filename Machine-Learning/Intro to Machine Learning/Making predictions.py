import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

# Load the dataset and drop instances with missing values
rides = pd.read_csv("cab_rides.csv").dropna()

# X = dataframe of input features
X = rides[['distance']]

# y = dataframe of the output feature
y = rides[['price']]

# Initialize the model
linearModel = LinearRegression()

# Fit the model
linearModel.fit(X, y)

# Make predictions for X
linearModel.predict(X)

# Make predictions for a new instance, a specified
# distance = 2.0 miles, 1.0, 5.0
Xnew = pd.DataFrame({'distance': [2.0, 1.0, 5.0]})
linearModel.predict(Xnew)

# Make predictions for X
linearModel.predict(X)

# Make predictions for a new instance, a specified
# distance = 2.0 miles, 1.0, 5.0
Xnew = pd.DataFrame({'distance': [2.0, 1.0, 5.0]})
print(linearModel.predict(Xnew))
