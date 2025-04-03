from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

# Set the mean and standard deviation
mu, sigma = 5, 2

# Calculates the density curve's value at x=3
P = norm.pdf(3, mu, sigma)
print('P(X=3) =', P)

# Calculate P(3<=X<=4) = P(X<=4) - P(X<=3)
cp = norm.cdf(4, mu, sigma) - norm.cdf(3, mu, sigma)
print('P(3<=X<=4) =', cp)

# Calculate density curve for -10 to +15
x = np.arange(-10, 15, 0.1)
P = norm.pdf(x, mu, sigma)

# Plot probability density
plt.plot(x, P)
plt.xlabel('x', fontsize=14)
plt.ylabel('Density', fontsize=14)

# Plot multiple normal distributions
x = np.arange(-8, 8, 0.1)
plt.plot(x, norm.pdf(x, 0, 1), label='Standard normal', linestyle='solid')
plt.plot(x, norm.pdf(x, 5, 1), label='mean=5, sd=1', linestyle='dashed')
plt.plot(x, norm.pdf(x, 0, 2), label='mean=0, sd=2', linestyle='dotted')
plt.legend()

## Standard Normal Distribution
# Set the mean and standard deviation for a standard normal distribution
mu, sigma = 0, 1

# Calculate P(-1<=Z<1) = P(Z<=1 - P(Z<=-1)
cp = norm.cdf(1, mu, sigma) - norm.cdf(-1, mu, sigma)
print('P(-1<=Z<=1) =', cp)

# Set the mean and standard deviation for a 1,1 normal distribution
mu, sigma = 1, 1

# Calculate P(-1<=X<1) = P(X<=1 - P(X<=-1)
cp = norm.cdf(1, mu, sigma) - norm.cdf(-1, mu, sigma)
print('P(-1<=X<=1) =', cp)

# Calculate density curve for -1 to +1
x = np.arange(-10, 15, 0.1)
P = norm.pdf(x, mu, sigma)

# Plot multiple normal distributions
x = np.arange(-4, 5, 0.1)
plt.plot(x, norm.pdf(x, 0, 1), label='Standard normal', linestyle='solid')
plt.plot(x, norm.pdf(x, 1, 1), label='mean=1, sd=1', linestyle='dashed')
plt.legend()

