import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression

# Load the dataset
rent_all = pd.read_csv('rent18.csv')

# Keep subset of features, drop missing values
rent = rent_all[['price', 'beds', 'baths', 'sqft','room_in_apt']].dropna()
rent.head()

# Define input and output features for predicting price based on square footage
X = rent[['sqft']].values.reshape(-1, 1)
y = rent[['price']].values.reshape(-1, 1)

# Initialize and fit simple linear regression model
simpLinModel = LinearRegression()
simpLinModel.fit(X,y)

# Estimated intercept weight
simpLinModel.intercept_

# Estimated weight for sqft feature
simpLinModel.coef_

# Find predicted values
yPredicted = simpLinModel.predict(X)

# Plot
plt.scatter(X, y, color='#1f77b4')
plt.plot(X, yPredicted, color='#ff7f0e', linewidth=2)
plt.xlabel('Square footage', fontsize=14)
plt.ylabel('Price ($)', fontsize=14)
plt.show()

# Predict the price of a 2,500 square foot rental
simpLinModel.predict([[2500]])

# Use multiple linear regression to predict price from square footage and number of bedrooms
X = rent[['beds']].values.reshape(-1, 1)

# Find predicted values
yPredicted = simpLinModel.predict(X)

# Plot
plt.scatter(X, y, color='#1f77b4')
plt.plot(X, yPredicted, color='#ff7f0e', linewidth=2)
plt.xlabel('Bedrooms', fontsize=14)
plt.ylabel('Price ($)', fontsize=14)
plt.show()

# Define input and output features
X = rent[['sqft', 'beds']].values.reshape(-1, 2)
y = rent[['price']].values.reshape(-1, 1)

# Initialize and fit multiple regression model
multRegModel = LinearRegression()
multRegModel.fit(X,y)

# Estimated intercept weight
multRegModel.intercept_

# Estimated weights for sqft and beds features
multRegModel.coef_

# Predict the price of a 2,500 square foot rental with 2 bedrooms
multRegModel.predict([[2500, 2]])

# Plot data and fitted model

# Create grid for prediction surface
Xvals=np.linspace(min(rent['sqft']), max(rent['sqft']),20)
Yvals=np.linspace(min(rent['beds']), max(rent['beds']),20)
Xg, Yg = np.meshgrid(Xvals, Yvals)
Zvals = np.array(multRegModel.intercept_[0] + (Xg * multRegModel.coef_[0,0] +  Yg * multRegModel.coef_[0,1]))

# Plot data and surface
fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.grid()
ax.scatter(rent[['sqft']], rent[['beds']], rent[['price']], color='#1f77b4')
ax.set_xlabel('Square footage', fontsize=14)
ax.set_ylabel('Bedrooms', fontsize=14)
ax.set_zlabel('Price ($)', fontsize=14)
ax.plot_surface(Xg, Yg, Zvals, alpha=.25, color='grey')
plt.show()


# Predict price from square footage, bedrooms, and bathrooms using a multiple regression model

# Define input and output features
X = rent[['sqft', 'baths']].values.reshape(-1, 2)
y = rent[['price']].values.reshape(-1, 1)

# Initialize and fit multiple regression model
multRegModel = LinearRegression()
multRegModel.fit(X,y)

# Estimated intercept weight
multRegModel.intercept_

# Estimated weights for sqft and baths features
multRegModel.coef_

# Predict the price of a 2,500 square foot rental with 2 bathrooms
multRegModel.predict([[2500, 2]])

# Plot data and fitted model

# Create grid for prediction surface
Xvals=np.linspace(min(rent['sqft']), max(rent['sqft']),20)
Yvals=np.linspace(min(rent['baths']), max(rent['baths']),20)
Xg, Yg = np.meshgrid(Xvals, Yvals)
Zvals = np.array(multRegModel.intercept_[0] + (Xg * multRegModel.coef_[0,0] +  Yg * multRegModel.coef_[0,1]))

# Plot data and surface
fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.grid()
ax.scatter(rent[['sqft']], rent[['baths']], rent[['price']], color='#1f77b4')
ax.set_xlabel('Square footage', fontsize=14)
ax.set_ylabel('Bathrooms', fontsize=14)
ax.set_zlabel('Price ($)', fontsize=14)
ax.plot_surface(Xg, Yg, Zvals, alpha=.25, color='grey')
plt.show()

# Define input and output features
X = rent[['sqft', 'baths', 'beds']].values.reshape(-1, 3)
y = rent[['price']].values.reshape(-1, 1)

# Initialize and fit multiple regression model
multRegModel = LinearRegression()
multRegModel.fit(X,y)

# Estimated intercept weight
multRegModel.intercept_

# Estimated weights for sqft and baths features
multRegModel.coef_

# Predict the price of a 2,500 square foot rental with 2 bedrooms and 2 bathrooms
multRegModel.predict([[2500, 2, 2]])


# Predicting Happiness

# Load the world happiness dataset
happiness = pd.read_csv("world_happiness_2017.csv")

# Define input and output features
X = happiness[['economy_gdp_per_capita']]
y = happiness[['happiness_score']]

# Initialize a simple linear regression model
happinessModel = LinearRegression()

# Fit a simple linear regression model
happinessModel.fit(X, y)

# Estimated intercept weight
happinessModel.intercept_

# Estimated weight for economy_gdp_per_capita feature
happinessModel.coef_

X = happiness[['generosity']]
SLRModel = LinearRegression()

# Fit a simple linear regression model
SLRModel.fit(X,y)

# Estimated intercept and generosity feature weight
SLRModel.intercept_
SLRModel.coef_

# Predict the happiness score for a country with freedom = 0.4 and health_life_expectancy = 0.8
X = happiness[['freedom', 'health_life_expectancy']]
SLRModel = LinearRegression()
SLRModel.fit(X, y)
SLRModel.predict([[0.4, 0.8]])


## Predicting Well-Being
# Load the dataset
wellbeing = pd.read_csv('city_wellbeing.csv')
wellbeing.head()
X = wellbeing[['NAI']].values
y = wellbeing[['WBI_Physical']].values
LRModel = LinearRegression()
LRModel.fit(X,y)
print(LRModel.intercept_)
print(LRModel.coef_)
LRModel.predict([[-0.76]])
wellbeing.loc[[42]]
