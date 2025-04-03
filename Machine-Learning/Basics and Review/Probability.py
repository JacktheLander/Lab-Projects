import numpy as np
import pandas as pd

credit_fraud = pd.read_csv('credit_fraud.csv')
credit_fraud.head()

# Create spending bins
bins = [0, 1, 5, 10, 20, 50, 100, 500, np.inf]
names = ['<$1', '$1-5', '$5-10', '$10-20', '$20-50', '$50-100', 
         '$100-500', '>$500']

credit_fraud['SpendingRange'] = pd.cut(credit_fraud['Amount'], bins, labels=names)
credit_fraud.head()

# Count the number of instances that are fraud vs. not fraud
count = credit_fraud['Class'].value_counts()
n = len(credit_fraud)

# Calculate the empirical probability of fraud
print('P(True) =', count/n)
# Calculate the empirical probability of no fraud
print('P(False) =', 284315/n)

# Count the number of instances in each bin
print(credit_fraud['SpendingRange'].value_counts())

# Count the number of fraud vs. non-fraud transactions
print(credit_fraud['Class'].value_counts())

# Calculate the probability of a transaction between $1-10
print('P($1-10) = P($1-5) + P($5-10) =', 38540/n + 31232/n)

# Calculate P(True) and P($20-50)
print('P(True) =', count/n)
print('P($20-50) =', 51915/n)

# Calculate P(True and $20-50)
print('P(True and $20-50) =', 35/n)
print('P(True)*P($20-50) =', 0.001727485630620034*0.18228133437731517)

# Count the number of instances in each spending range/class combinations
print(credit_fraud.groupby(['SpendingRange', 'Class']).size())

n = len(credit_fraud)

# Calculate the conditional probability of fraud for transactions less than $1
print('P(True|$<1) = P(True and $<1)/P($<1) =', (154/n)/(28667/n))

# Calculate the conditional probability of fraud for transactions Between $5-10
print('P(True|$<1) = P(True and $5-10)/P($5-10) =', (27/n)/(31232/n))

# Dividing by n isn't necessary
print('P(True|$<1) = P(True and $<1)/P($<1) =', 154/28667)
print('P(True|$1-5) = P(True and $1-5)/P($1-5) =', 41/38540)
print('P(True|$>500) = P(True and $>500)/P($>500) =', 35/9142)
