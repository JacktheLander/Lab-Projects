#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Using a random sample of rides speeds up visualization
rides = pd.read_csv('cab_rides.csv')
rides = rides.sample(1000, random_state=123)


# In[3]:


rides.head()


# In[4]:


# sns.countplot() creates a bar plot
p = sns.countplot(data=rides, x='cab_type')

# matplotlib functions are used to customize labels
p.set_xlabel('Cab type', fontsize=14)
p.set_ylabel('Count', fontsize=14)


# In[5]:


# sns.histplot() creates a histogram
p = sns.histplot(data=rides, x='distance', bins=20)
p.set_xlabel('Distance (miles)', fontsize=14)
p.set_ylabel('Count', fontsize=14)


# In[6]:


# sns.scatterplot() creates a scatter plot
p = sns.scatterplot(data=rides, x='distance', y='price', 
                    hue='cab_type', style='cab_type')
p.set_xlabel('Distance (miles)', fontsize=14)
p.set_ylabel('Price', fontsize=14)
p.legend(title='Cab type')

