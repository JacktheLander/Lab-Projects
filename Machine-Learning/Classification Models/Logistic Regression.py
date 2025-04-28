import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions

# Load the penguins dataset and drop instances with missing values
penguins = pd.read_csv('penguins.csv').dropna()

# Create binary feature for each species
penguins['Adelie'] = penguins['species'].replace(to_replace = ['Adelie','Chinstrap', 'Gentoo'],
                                                      value = [int(1), int(0), int(0)])
penguins['Chinstrap'] = penguins['species'].replace(to_replace = ['Adelie','Chinstrap', 'Gentoo'],
                                                      value = [int(0), int(1), int(0)])
penguins['Gentoo'] = penguins['species'].replace(to_replace = ['Adelie','Chinstrap', 'Gentoo'],
                                                      value = [int(0), int(0), int(1)])
penguins.head()

# Define input features and output features
X = penguins[['bill_length_mm']]
y = penguins[['Adelie']]

# Initialize a logistic regression model
logisticModel = LogisticRegression(penalty='l2', C=1/12)

# Fit the model
logisticModel.fit(X, np.ravel(y))

# Print the fitted model
print('w1:', logisticModel.coef_)
print('w0:', logisticModel.intercept_)

# Calculate predicted probabilities
logisticModel.predict_proba(X)[0:6]

# Classify instances in X
logisticModel.predict(X)[0:6]

# Calculate the proportion of instances correctly classified
logisticModel.score(X, np.ravel(y))

# Plot logistic regression model
plt.scatter(X, y, color='black')

xrange = np.linspace(X.min(), X.max(), 10000)
yhat = logisticModel.predict(X).reshape(-1, 1).astype(int)
yprob = logisticModel.predict_proba(xrange.reshape(-1, 1))[:, 1]

plt.plot(xrange, yprob, color='#4878d0', linewidth=2)
plt.xlabel('Bill length (mm)', fontsize=14)
plt.ylabel('Probability of Adelie', fontsize=14)

# Use additional input features
X = penguins[['bill_length_mm', 'bill_depth_mm']]
y = penguins[['Adelie']]

# Initialize a logistic regression model
logisticModel = LogisticRegression(penalty='None')

# Fit the model
logisticModel.fit(X, np.ravel(y))

# Print the fitted model
print('w1, w2:', logisticModel.coef_)
print('w0:', logisticModel.intercept_)

# Proportion of instances correctly classified
logisticModel.score(X, y)

# Decision boundary plot with two input features
# Set background opacity to 20%
contourf_kwargs = {'alpha': 0.2}

# Plot decision boundary regions
p = plot_decision_regions(X.to_numpy(), np.ravel(y),
                          clf=logisticModel, contourf_kwargs=contourf_kwargs,
                          colors='#7f7f7f,#1f77b4')

# Add title and axis labels
p.set_title('Decision boundary plot', fontsize=16)
p.set_xlabel('Bill length (mm)', fontsize=14)
p.set_ylabel('Bill depth (mm)', fontsize=14)

# Add legend
L = plt.legend()
L.get_texts()[0].set_text('Chinstrap or Gentoo')
L.get_texts()[1].set_text('Adelie')


# Load the heart dataset
heart = pd.read_csv('heart.csv')

# Create a dataframe X containing thalach and chol
X = heart[['thalach', 'chol']]

# Output feature: target
y = heart[['target']]

# Initialize the model
heartModel = LogisticRegression(penalty='l2')

# Fit the model
heartModel.fit(X, np.ravel(y))

# Calculate the predicted probabilities
probs = heartModel.predict_proba(X)

print('Probabilities: {}'.format(probs))

# Calculate the predicted classes
classes = heartModel.predict(X)

print('Classes: {}'.format(classes))
