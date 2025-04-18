import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# Load the dataset
rent_all = pd.read_csv('rent18.csv')

# Keep subset of features, drop missing values
rent = rent_all[rent_all['city']=='san jose']
rent = rent[['price', 'beds', 'baths', 'sqft']].dropna()
rent.head()

# Predict price from sqft
# Define input and output features
X = rent[['sqft']]
y = rent[['price']]


# Plot sqft and price
fig = plt.figure(figsize = (4,3))
plt.scatter(X, y, color='#1f77b4')
plt.xlabel('Square footage', fontsize=14)
plt.ylabel('Price ($)', fontsize=14)
plt.ylim([700,5000])
plt.xlim([200, 3000])

plt.show()

# Initiate and fit a k-nearest neighbors regression model with k=5
knnr = KNeighborsRegressor(n_neighbors=5)
knnrFit = knnr.fit(X,y)

# Define a new instance with 2000 square feet
Xnew = [[2000]]

# Predict price for new instance
neighbors = knnrFit.predict(Xnew)

# Find the 5 nearest neighbors for the new instance
neighbors = knnrFit.kneighbors(Xnew)

# Return only the distances between the new instance and each of the the 5 nearest neighbors
neighbors[0]

# Return the data frame instances of the 5 nearest neighbors
rent.iloc[neighbors[1][0]]

# Plot data with k-nearest neighbors prediction
Xvals=np.linspace(200, 3000, 100).reshape(-1, 1)
knnrPred = knnr.predict(Xvals)

fig = plt.figure(figsize = (4,3))
plt.scatter(X, y, color='#1f77b4')
plt.plot(Xvals, knnrPred, color='#ff7f0e', linewidth=2)
plt.xlabel('Square footage', fontsize=14)
plt.ylabel('Price ($)', fontsize=14)
plt.ylim([700,5000])
plt.xlim([200, 3000])

plt.show()

# Define input features as sqft, beds, and baths
X = rent[['sqft', 'beds', 'baths']]
y = rent[['price']]

# Scale the input features
scaler = StandardScaler()
Xscaled = scaler.fit_transform(X)

# Initiate and fit a k-nearest neighbors regression model with k=5 on unscaled input features
knnrUnscaled = KNeighborsRegressor(n_neighbors=5)
knnrUnscaledFit = knnrUnscaled.fit(X, y)

# Initiate and fit a k-nearest neighbors regression model with k=5 on unscaled input features
knnrScaled = KNeighborsRegressor(n_neighbors=5)
knnrScaledFit = knnrScaled.fit(Xscaled, y)

# Define new instance with 2000 square feet, 2 bedrooms, 1 bathroom
Xsqft = 2000
Xbeds = 2
Xbaths = 1
Xnew = [[Xsqft, Xbeds, Xbaths]]

# Predict price for new instance using unscaled input features
print("Prediction from unscaled input features: ", knnrUnscaledFit.predict(Xnew)[0][0])

# Predict price for new instance using scaled input features
# Find scaled input features for new instance
XsqftScaled = (Xsqft - rent['sqft'].mean())/(rent['sqft'].var()**.5)
XbedsScaled = (Xbeds - rent['beds'].mean())/(rent['beds'].var()**.5)
XbathsScaled = (Xbaths - rent['baths'].mean())/(rent['baths'].var()**.5)

XnewScaled = [[XsqftScaled, XbedsScaled, XbathsScaled]]

print("Prediction from scaled input features: ",knnrScaledFit.predict(XnewScaled)[0][0])

# Unscaled nearest neighbors
rent.iloc[knnrUnscaledFit.kneighbors(Xnew)[1][0]]

# Scaled nearest neighbors
rent.iloc[knnrScaledFit.kneighbors(XnewScaled)[1][0]]
