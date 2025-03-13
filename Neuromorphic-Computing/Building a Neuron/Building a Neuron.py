import numpy
import scipy
import pandas
import networkx
import matplotlib


def countval(size):
    array = numpy.random.random(size)
    count = numpy.sum(array > 0.4)
    print("Count of elements greater than 0.4:", count)
    print("Shape of the array:", array.shape)
    return

def MatrixOper(size):
    A = numpy.random.random((size, size))
    B = numpy.random.random((size, size))
    Add = A + B
    Sub = A - B
    Multiply = A * B
    Divide = numpy.divide(A, B, out=numpy.zeros_like(A), where=B!=0)
    print("Shape of addition:", Add.shape)
    print("Shape of subtraction:", Sub.shape)
    print("Shape of multiplication:", Multiply.shape)
    print("Shape of division:", Divide.shape)
    return

def MatrixMult(A, B, C):
    Multiply = numpy.dot(A, B)
    Multiply = numpy.dot(Multiply, C)
    print("Output Matrix", Multiply)
    return

class MatrixObj:
    def __init__(self, input_dict):
        self.A = input_dict["A"]
        self.B = input_dict["B"]
        self.C = input_dict["C"]
        self.Str = input_dict["Str"]

    def MatriOper(self):
        Multiply = numpy.dot(self.A, self.B)
        Multiply = numpy.dot(Multiply, self.C)
        return Multiply

    def CountStr(self):
        char_counts = {}
        for char in self.Str.replace(" ", "").lower():
            if char in char_counts:
                char_counts[char] += 1
            else:
                char_counts[char] = 1
        return char_counts


def Sigmoid(Z):
    return 1 / (1 + numpy.exp(-Z))

def SigmoidDerivative(Z):
    return numpy.multiply(Sigmoid(Z), numpy.subtract(1.0, Sigmoid(Z)))

def Hyperbolic(Z):
    return numpy.tanh(Z)

def HyperbolicDerivative(Z):
    return 1 - numpy.square(numpy.tanh(Z))

def ReLU(Z):
    return numpy.maximum(0, Z)

def ReLUDerivative(Z):
    if Z < 0: Z = 0
    else: Z = 1
    return Z

def Softmax(Z):
    ExpVals = numpy.exp(numpy.subtract(Z, numpy.max(Z)))
    ExpValSum = numpy.sum(ExpVals)
    return numpy.divide(ExpVals, ExpValSum)

def SoftmaxDerivative(Z):
    # https://aimatters.wordpress.com/2019/06/17/the-softmax-function-derivative/
    S = numpy.array([Softmax(Z)])
    shape = S.shape
    S_vector = S.reshape(shape[0], 1)
    S_matrix = numpy.tile(S_vector, S.shape[0])
    return numpy.diag(S) - (S_matrix * numpy.transpose(S_matrix))

def Softplus(Z):
    return numpy.log(1 + numpy.exp(Z))

class Neuron:
    def __init__(self, W, x, b, PhiFunc):
        self.W = numpy.array(W)  # Weights
        self.x = numpy.array(x)  # Input
        self.b = numpy.array(b)  # Bias
        self.PhiFunc = PhiFunc

    def activate(self):
        z = numpy.dot(self.W, self.x) + self.b  # Linear Combination done before activation
        return self.PhiFunc(z)

if __name__ == "__main__":
    A = numpy.array([[1, 2]])
    B = numpy.array([[3], [4]])
    C = numpy.array([[5, 6]])
    test_string = "the quick brown fox jumps over the lazy dog"

    input_dict = {
        "A": A,
        "B": B,
        "C": C,
        "Str": test_string
    }

    W = [0.2, -0.5, 0.1]
    x = [1.0, 2.0, 3.0]
    b = 0.1
    activation_functions = {
        "Sigmoid": Sigmoid,
        "Sigmoid Derivative": SigmoidDerivative,
        "Hyperbolic": Hyperbolic,
        "Hyperbolic Derivative": HyperbolicDerivative,
        "ReLU": ReLU,
        "ReLU Derivative": ReLUDerivative,
        "Softmax": Softmax,
        "Softmax Derivative": SoftmaxDerivative,
        "Softplus": Softplus
    }

    for name, func in activation_functions.items():
        neuron = Neuron(W, x, b, func)
        output = neuron.activate()
        print(f"{name} Output: {output}")
