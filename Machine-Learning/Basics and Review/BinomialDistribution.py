from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np

# Set the sample size and probability of success
n, p = 25, 0.10

# Calculates the probability of x=3 successes given the defined n and p
P = binom.pmf(3, n, p)
print('P(X=3) =', P)

# Calculates the cumulative probability of x=3 or fewer successes
cp = binom.cdf(3, n, p)
print('P(X <=3) =', cp)

# Calculate probabilities of all possible values
x = np.arange(0, n+1, 1)
P = binom.pmf(x, n, p)

# Plot probability distribution
plt.bar(x, P)
plt.xlabel('x', fontsize=14)
plt.ylabel('P(X=x)', fontsize=14)

# Set the sample size and probability of success
n, p = 50, 0.6

# Calculates the probability of x=32 successes given the defined n and p
P = binom.pmf(32, n, p)
print('P(X=32) =', P)

## Now for a Binomial(50, 0.6) distribution

# Calculates the cumulative probability of x=32 or fewer successes
cp = binom.cdf(32, n, p)
print('P(X <=32) =', cp)

# Calculate probabilities of all possible values
x = np.arange(0, n+1, 1)
P = binom.pmf(x, n, p)

# Plot probability distribution
plt.bar(x, P)
plt.xlabel('x', fontsize=14)
plt.ylabel('P(X=x)', fontsize=14)
