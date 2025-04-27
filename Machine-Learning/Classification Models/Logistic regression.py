#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions


# In[2]:


# Load the penguins dataset and drop instances with missing values
penguins = pd.read_csv('penguins.csv').dropna()


# In[3]:


# Create binary feature for each species
penguins['Adelie'] = penguins['species'].replace(to_replace = ['Adelie','Chinstrap', 'Gentoo'],
                                                      value = [int(1), int(0), int(0)])
penguins['Chinstrap'] = penguins['species'].replace(to_replace = ['Adelie','Chinstrap', 'Gentoo'],
                                                      value = [int(0), int(1), int(0)])
penguins['Gentoo'] = penguins['species'].replace(to_replace = ['Adelie','Chinstrap', 'Gentoo'],
                                                      value = [int(0), int(0), int(1)])
penguins.head()


# In[4]:


# Define input features and output features
X = penguins[['bill_length_mm']]
y = penguins[['Adelie']]


# In[5]:


# Initialize a logistic regression model
logisticModel = LogisticRegression(penalty='l2', C=1/12)

# Fit the model
logisticModel.fit(X, np.ravel(y))

# Print the fitted model
print('w1:', logisticModel.coef_)
print('w0:', logisticModel.intercept_)


# In[6]:


# Calculate predicted probabilities
logisticModel.predict_proba(X)[0:6]


# In[7]:


# Classify instances in X
logisticModel.predict(X)[0:6]


# In[8]:


# Calculate the proportion of instances correctly classified
logisticModel.score(X, np.ravel(y))


# In[9]:


# Plot logistic regression model
plt.scatter(X, y, color='black')

xrange = np.linspace(X.min(), X.max(), 10000)
yhat = logisticModel.predict(X).reshape(-1, 1).astype(int)
yprob = logisticModel.predict_proba(xrange.reshape(-1, 1))[:, 1]

plt.plot(xrange, yprob, color='#4878d0', linewidth=2)
plt.xlabel('Bill length (mm)', fontsize=14)
plt.ylabel('Probability of Adelie', fontsize=14)


# In[10]:


# Use additional input features
X = penguins[['bill_length_mm', 'bill_depth_mm']]
y = penguins[['Adelie']]


# In[11]:


# Initialize a logistic regression model
logisticModel = LogisticRegression(penalty=None)

# Fit the model
logisticModel.fit(X, np.ravel(y))

# Print the fitted model
print('w1, w2:', logisticModel.coef_)
print('w0:', logisticModel.intercept_)


# In[12]:


# Proportion of instances correctly classified
logisticModel.score(X, y)


# In[13]:


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

