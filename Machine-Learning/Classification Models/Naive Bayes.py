#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from mlxtend.plotting import plot_decision_regions


# In[2]:


## Load the penguins dataset and drop instances with missing values
penguins = pd.read_csv('penguins.csv').dropna()

# Display first five rows
penguins.head()


# In[3]:


# Create integer-valued species
penguins['species_int'] = penguins['species'].replace(to_replace = ['Adelie','Chinstrap', 'Gentoo'],
                                                      value = [int(0), int(1), int(2)])

# Calculate the number of penguins in each species
penguins.groupby('species').count()


# In[4]:


# Define input features and output features
X = penguins[['bill_length_mm']]
y = penguins[['species_int']]


# In[5]:


# Initialize a Gaussian naive Bayes model
NBModel = GaussianNB()

# Fit the model
NBModel.fit(X, np.ravel(y))

# Calculate the predictions for each instance in X
NBModel.predict(X)

# Calculate the proportion of instances correctly classified
NBModel.score(X, np.ravel(y))


# In[6]:


NBModel.predict_proba(X)[0:10]


# In[7]:


# Plot Gaussian naive Bayes model
xrange = np.linspace(X.min(), X.max(), 10000)
yhat = NBModel.predict(X).reshape(-1, 1).astype(int)
probAdelie = NBModel.predict_proba(xrange.reshape(-1, 1))[:, 0]
probChinstrap = NBModel.predict_proba(xrange.reshape(-1, 1))[:, 1]
probGentoo = NBModel.predict_proba(xrange.reshape(-1, 1))[:, 2]

plt.plot(xrange, probAdelie, color='#1f77b4', linewidth=2)
plt.plot(xrange, probChinstrap, color='#ff7f0e', linewidth=2)
plt.plot(xrange, probGentoo, color='#3ca02c', linewidth=2)

plt.xlabel('Bill length (mm)', fontsize=14)
plt.ylabel('Probability of each species', fontsize=14)


# In[8]:


# Use additional input features
X = penguins[['bill_length_mm', 'bill_depth_mm']]
y = penguins[['species_int']]


# In[9]:


# Initialize a Gaussian naive Bayes model
NBModel = GaussianNB()

# Fit the model
NBModel.fit(X, np.ravel(y))

# Calculate the predictions for each instance in X
NBModel.predict(X)

# Calculate the proportion of instances correctly classified
NBModel.score(X, np.ravel(y))


# In[10]:


# Decision boundary plot with two input features
# Set background opacity to 20%
contourf_kwargs = {'alpha': 0.2}

# Plot decision boundary regions
p = plot_decision_regions(X.to_numpy(), np.ravel(y),
                          clf=NBModel, contourf_kwargs=contourf_kwargs)

# Add title and axis labels
p.set_title('Decision boundary plot', fontsize=16)
p.set_xlabel('Bill length (mm)', fontsize=14)
p.set_ylabel('Bill depth (mm)', fontsize=14)

# Add legend
L = plt.legend()
L.get_texts()[0].set_text('Adelie')
L.get_texts()[1].set_text('Chinstrap')
L.get_texts()[2].set_text('Gentoo')

