import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

## Load the penguins dataset and drop instances with missing values
penguins = pd.read_csv('penguins.csv').dropna()

# Define input features and output features
X = penguins[['bill_length_mm', 'bill_depth_mm']]
y = penguins[['species']]

# Scale the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize a linear discriminant model
LDAmodel = LinearDiscriminantAnalysis(n_components=2, store_covariance=True)

# Fit the model
LDAmodel.fit(X, np.ravel(y))

# Calculate predictions
preds = LDAmodel.predict(X)

# Discriminant intercepts - w0
LDAmodel.intercept_

# Discriminant weights - w1 and w2
LDAmodel.coef_

# Class means
LDAmodel.means_

# Covariance matrix
LDAmodel.covariance_

# Inverse covariance matrix
# np.linalg.pinv calculates matrix inverses
np.linalg.pinv(LDAmodel.covariance_)

# Weights = Group means * Inverse covariance
LDAmodel.means_ @ np.linalg.pinv(LDAmodel.covariance_)

# Plot instances in original feature space
p = sns.scatterplot(data=penguins, x='bill_length_mm', 
                    y='bill_depth_mm', hue=preds)
p.set_xlabel('Bill length (mm)', fontsize=14)
p.set_ylabel('Bill depth (mm)', fontsize=14)
plt.legend(title='Species')
plt.show()

# Plot instances in transformed feature space
Xtransformed = LDAmodel.transform(X)
p = sns.scatterplot(x=Xtransformed[:, 0], y=Xtransformed[:, 1], 
                    hue=preds)
p.set_xlabel('Principal component 1', fontsize=14)
p.set_ylabel('Principcal component 2', fontsize=14)
plt.legend(title='Species')
plt.show()



# Load the wine_white dataset
wine = pd.read_csv('wine_white.csv')

# Create a dataframe X containing fixed_acidity, free_sulfur_dioxide, citric_acid, residual_sugar
X = wine[['fixed_acidity', 'free_sulfur_dioxide', 'citric_acid', 'residual_sugar']]

# Create a dataframe y containing quality
y = wine[['quality']]

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the model
LDAWine = LinearDiscriminantAnalysis(n_components=4, store_covariance=True)

# Print model parameters
print(LDAWine.get_params())


# Create a dataframe X containing free_sulfur_dioxide, density, alcohol
X = wine[['free_sulfur_dioxide', 'density', 'alcohol']]

# Create a dataframe y containing quality
y = wine[['quality']]

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the model
modelWine = LinearDiscriminantAnalysis(n_components=3, store_covariance=True)

# Fit the model to X and y
modelWine.fit(X_scaled, np.ravel(y))

# Print model predictions
print(modelWine.predict(X_scaled))


# Create a dataframe X containing volatile_acidity, alcohol, pH, total_sulfur_dioxide
X = wine[['volatile_acidity', 'alcohol', 'pH', 'total_sulfur_dioxide']]

# Create a dataframe y containing quality
y = wine[['quality']]

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the model
LDA = LinearDiscriminantAnalysis(n_components=4, store_covariance=True)
# Fit the model to X and y
LDA.fit(X_scaled, y)

# Print discriminant intercepts
print(LDA.intercept_)

# Print discriminant weights
# Your code goes here
print(LDA.coef_)

# Print covariance matrix
print(LDA.covariance_)
