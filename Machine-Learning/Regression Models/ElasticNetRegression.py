import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression

# Load the dataset
rent_all = pd.read_csv('rent18.csv')

# Keep subset of features, drop missing values
rent = rent_all[['price', 'beds', 'baths', 'sqft']].dropna()
rent.head()


# Use elastic net regression to predict rental price from square footage

# Define input and output features for a simple linear model predicting price from sqft
X = rent[['sqft']]
y = rent[['price']]

# Scale the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize and fit model using elastic net regression
eNet = ElasticNet(alpha=1, l1_ratio=.5)
eNet.fit(X,y)

# Estimated intercept weight
eNet.intercept_

# Estimated weight for sqft
eNet.coef_


# Compare elastic net to least squares

# Fit using least squares

linRegModel=LinearRegression()
linRegModel.fit(X,y)

linRegModel.intercept_
linRegModel.coef_

# Plot the data and both fitted models

# Find predicted values
yPredictedENet = eNet.predict(X)
yPredictedLin = linRegModel.predict(X)

# Plot
plt.scatter(X, y, color='#1f77b4', s=10)
plt.plot(X, yPredictedENet, color='#ff7f0e', linewidth=2, label='Elastic net')
plt.plot(X, yPredictedLin, color='#3ca02c', linewidth=2, label='Least squares')
plt.xlabel('Standardized square footage', fontsize=14)
plt.ylabel('Price ($)', fontsize=14)
plt.legend(loc='upper left')
plt.show()


# Use elastic net regression to predict rental price from square fottage and number of bedrooms.

# Define input and output features
X = rent[['sqft', 'beds']]
y = rent[['price']]

# Scale the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize and fit elastic net
eNet2 = ElasticNet(alpha=1, l1_ratio=0.5)
eNet2.fit(X,y)

# Estimated intercept and weights
print(eNet2.intercept_)
print(eNet2.coef_)

# Plot the absolute value of the weights
importance = np.abs(eNet2.coef_)
names = np.array(['sqft', 'beds'])
sort = np.argsort(importance)[::-1]
plt.bar(x=names[sort], height=importance[sort])
plt.ylabel('Importance', fontsize=14)
plt.show()

# Set $\alpha$ = 1, show weight estimates for different values of the weight applied to the L1 norm, $\lambda$.

aVal = 1
l1vals = np.linspace(0.01, 1, 100,)
ENcoef = np.empty([100,3])

for i in range(len(l1vals)):
    EN = ElasticNet(alpha=aVal, l1_ratio=l1vals[i], max_iter=10000)
    EN.fit(X,y)
    ENcoef[i,0]=EN.intercept_[0]
    ENcoef[i,1]=EN.coef_[0]
    ENcoef[i,2]=EN.coef_[1]

# Plot
fig = plt.figure(figsize = (5.5,4))
plt.plot(l1vals, ENcoef[:,1], color='#1f77b4', linestyle='solid', linewidth=3, label=r"$w_1$")
plt.plot(l1vals, ENcoef[:,2], color='#ff7f0e', linestyle='dashed', linewidth=3, label=r"$w_2$")

plt.xlabel(r'Weight applied to L1 norm, $\lambda$', fontsize=14)
plt.ylabel('Estimate', fontsize=14)

plt.legend(loc='upper left', fontsize=14)

plt.show()

# Set the weight applied to the L1 norm at $\lambda$ = 0.5, show weight estimates for different values of the regularization strength, $\alpha$.
l1NormWeight = 0.5
alphaVals = np.logspace(-1, 2, 100)
ENcoef = np.empty([100,3])

for i in range(len(ENcoef)):
    EN = ElasticNet(alpha=alphaVals[i], l1_ratio=l1NormWeight, max_iter=2000)
    EN.fit(X,y)
    ENcoef[i,0]=EN.intercept_[0]
    ENcoef[i,1]=EN.coef_[0]
    ENcoef[i,2]=EN.coef_[1]


# Plot
fig = plt.figure(figsize = (6,4))
plt.plot(alphaVals, ENcoef[:,1], color='#1f77b4', linestyle='solid', linewidth=3, label=r"$w_1$")
plt.plot(alphaVals, ENcoef[:,2], color='#ff7f0e', linestyle='dashed', linewidth=3, label=r"$w_2$")


plt.xlabel(r'Regularization strength, $\alpha$', fontsize=14)
plt.ylabel('Estimate', fontsize=14)

plt.legend(loc='upper right', fontsize=14)

plt.show()

