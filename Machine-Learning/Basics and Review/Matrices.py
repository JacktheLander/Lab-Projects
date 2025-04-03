import numpy as np

# np.matrix fills in elements row by row
A = np.matrix([[1, 2, 3, 4], [5, 6, 7, 8]])

print(A)
print(A.shape)

# Matrices are considered two-dimensional arrays
B = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

print(B)
print(B.shape)

# np.identity() creates an identity matrix
I = np.identity(n=3)

print(I)
print(I.shape)

# np.ones() creates a matrix of ones
O = np.ones(shape=(2, 3))

print(O)

# np.zeros() creates a matrix of zeros
Z = np.zeros(shape=(1, 4))

print(Z)

## Matrix Multiplication
A = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print("A:", A)
print(A.shape)

B = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
print("B:", B)
print(B.shape)

print("A multiplied by B:", np.matmul(A, B))
print("B multiplied by A:", np.matmul(B, A))

## Transposes and Inverses
C = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print("C:", C)
print(C.shape)

# np.transpose creates the transpose matrix
C_t = np.transpose(C)
print("C Transpose: ", C_t)
print(C_t.shape)

# D is a square matrix, so D^-1 exists
D = np.array([[1, 2], [3, 4]])
print("D:", D)
print("D Transpose: ", np.linalg.inv(D)

# Use np.linalg.pinv(C) to find the pseudoinverse of C
print("Pseudoinverse of C: ", np.linalg.pinv(C))

# CC^- = Identity matrix
print("Identity Matrix of C: ", np.matmul(C, np.linalg.pinv(C)))

## Eigenvalues and Eigenvectors
A = np.array([[4, -4, 0],
              [1, -2, 0],
              [0, 5, 3]])
print(A)

eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues: ", eigenvalues)
print("Eigenvectors: ", eigenvectors)
print("Determinant: ", np.linalg.det(A))
