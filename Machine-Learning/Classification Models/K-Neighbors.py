import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler

## Load the penguins dataset and drop instances with missing values
penguins = pd.read_csv('penguins.csv').dropna()

# Display first five rows
penguins.head()

## Scatterplot of bill length and bill depth
p = sns.scatterplot(data=penguins, x='bill_length_mm', y='bill_depth_mm',
                    hue='species', style='species')
p.set_xlabel('Bill length (mm)', fontsize=14)
p.set_ylabel('Bill depth (mm)', fontsize=14)
p.legend(title='Species')

# Create column with species saved as integer
penguins['species_int'] = penguins['species'].replace(to_replace = ['Adelie','Chinstrap', 'Gentoo'],
                                                      value = [int(0), int(1), int(2)])
penguins.head()

# Define input features and output features
X = penguins[['bill_length_mm', 'bill_depth_mm']]
y = penguins[['species_int']]

# Initialize a model with k=5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model
knn.fit(X, np.ravel(y))

# Calculate the predictions for each instance in X
knn.predict(X)

# Calculate the proportion of instances correctly classified
knn.score(X, np.ravel(y))

# Set background opacity to 20%
contourf_kwargs = {'alpha': 0.2}

# Plot decision boundary regions
p = plot_decision_regions(X.to_numpy(), np.ravel(y), clf=knn, contourf_kwargs=contourf_kwargs)

# Add title and axis labels
p.set_title('Decision boundary plot', fontsize=16)
p.set_xlabel('Bill length (mm)', fontsize=14)
p.set_ylabel('Bill depth (mm)', fontsize=14)

# Add legend 
L = plt.legend()
L.get_texts()[0].set_text('Adelie')
L.get_texts()[1].set_text('Chinstrap')
L.get_texts()[2].set_text('Gentoo')

# Use additional input features
X = penguins[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = penguins[['species_int']]

# Initialize a model with k=5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model
knn.fit(X, np.ravel(y))

# Calculate the predictions for each instance in X
knn.predict(X)

# Calculate the proportion of instances correctly classified
knn.score(X, np.ravel(y))

# Standardize features in X and re-fit the model

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled inputs back to a dataframe
X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

# Re-fit the model using X_scaled
knn.fit(X_scaled, np.ravel(y))

# Calculate the predictions for each instance in X
knn.predict(X_scaled)

# Calculate the proportion of instances correctly classified
knn.score(X_scaled, np.ravel(y))

# Load the heart dataset
heart = pd.read_csv('heart.csv')

# Create a dataframe X containing thalach and age
X = heart[['thalach', 'age']]

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Output feature: target
y = heart[['target']]

# Initialize the model
knnHeart = KNeighborsClassifier(n_neighbors=10)

# Fit the model to X and y
knnHeart.fit(X, np.ravel(y))

# Print model predictions
print(knnHeart.predict(X))

# Print proportion of instances classified correctly
print(knnHeart.score(X, y))
