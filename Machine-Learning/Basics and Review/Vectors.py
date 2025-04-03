import numpy as np

## A vector is a one-dimensional array
w = np.array([1, 2, 3, 4])

print("w:", w)
print("Shape:", w.shape)
print("Dimensions:", w.ndim)

v = np.array([2, 4, 6, 8, 10])
print("\nw:", w)
print("Shape: ", v.shape)
print("Dimensions: ", v.ndim)

## Vector Operations
# Scalar multiplication
print("\nScalar multiplication: 2w =", 2*w)

# Vector addition
print("\nVector addition: \nw + v =", w+v)

# Vector subtraction
print("\nVector addition: w + v =", w-v)

# Dot product
print("\nDot product: \nw.v =",np.dot(w, v))

# What does w * v return?
print("\nWhat does w * v return?\nw * v =", w*v)

## Mathematical and Statistical Functions
w = np.array([-3, -2, -1, 0, 1, 2, 3])
print("\n\nw:", w)

# Absolute Value
print("\nAbsolute value: \n|w| =", np.abs(w))

# Exponential
print("\nExponential: \ne^w =", np.exp(w))

# Standard Deviation
print("\nStandard Deviation:", np.std(w))

# Mean
a = np.array([4, 5, 1, 3, 10])
b = np.array([1, 6, 5, 6, -4])
print("\n\n a =",a)
print("\n b =",b)
print("\nMean(b):", np.mean(b))

# Log
print("\nLog(a):", np.log(a))

## Vector Norms
a = np.array([1, 7, 6, 9, 9])
print("a:", a)
b = np.array([9, 7, 3, 5, 9])
print("b:", b)

# L1 norm

# np.abs() calculates absolute values
w_abs = np.abs(a)
# np.sum() sums elements in a vector
L1 = np.sum(w_abs)

print("L1 norm:", L1)

# L2 norm

# w**2 squares all elements
w_sq = a**2

# np.sum() sums elements in a vector
# np.sqrt() calculates the square root
L2 = np.sqrt(np.sum(w_sq))

print("L2 norm:", L2)

# np.linalg.norm() calculates L1, L2, and other norms

L1 = np.linalg.norm(b, ord=1)
print("L1 norm:", L1)

L2 = np.linalg.norm(b, ord=2)
print("L2 norm:", L2)
