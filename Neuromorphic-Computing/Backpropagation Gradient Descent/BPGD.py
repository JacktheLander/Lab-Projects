# ******************************************************************************
# import modules
# ******************************************************************************
import numpy as np
import torch, os
from os.path import join, isfile
import pandas as pd

# ******************************************************************************
def DotSumAddBias(x, W, b):
	return np.add(np.dot(x, W), b)

# ******************************************************************************
def ReLU(Z):
	'''ReLU function
	Inputs:
		Z: a 2d matrix (mxn): m mini-batch, n output neurons
	Returns: ReLU values
	'''
	return np.maximum(0, Z)

# ******************************************************************************
def ReLUDerivative(Z):
	'''ReLU derivative function
	Inputs:
		Z: a 2d matrix (mxn): m mini-batch, n output neurons
	Returns: derivative of ReLU values
	'''
	Z[Z <= 0.0] = 0.0
	Z[Z > 0.0]  = 1.0
	return Z

# ******************************************************************************
def Sigmoid(Z):
	'''Sigmoid function
	Inputs:
		Z: a 2d matrix (mxn): m mini-batch, n output neurons
	Returns: Sigmoid values
	'''
	return np.divide(1.0, np.add(1.0, np.exp(-Z)))

# ******************************************************************************
def SigmoidDerivative(Z):
	'''Sigmoid derivative function
	Inputs:
		Z: a 2d matrix (mxn): m mini-batch, n output neurons
	Returns: derivative Sigmoid values
	'''
	return np.multiply(Sigmoid(Z), np.subtract(1.0, Sigmoid(Z)))

# ******************************************************************************
def Softmax(Z):
	'''Softmax function
	Inputs:
		Z: a 2d matrix (mxn): m mini-batch, n output neurons
	Returns: softmax values
	'''
	ExpVals     = np.exp(np.subtract(Z, np.max(Z)))
	ExpValSum   = np.sum(ExpVals)
	return np.divide(ExpVals, ExpValSum)

# ******************************************************************************
def SoftmaxDerivative(Z): # Best implementation (VERY FAST)
	'''Softmax derivative function
		Returns the jacobian of the Softmax function for the given set of inputs.
	Inputs:
		x: should be a 2d (mxn) matrix where m corresponds to the samples
			(or mini-batch), and n is the number of nodes.
	Returns: jacobian derivative of softmax
	reference: https://www.bragitoff.com/2021/12/efficient-implementation-of-softmax-\
			activation-function-and-its-derivative-jacobian-in-python/
	'''
	s     = Softmax(Z)
	a     = np.eye(s.shape[-1])
	Temp1 = np.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=np.float32)
	Temp2 = np.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=np.float32)
	Temp1 = np.einsum('ij,jk->ijk', s, a)
	Temp2 = np.einsum('ij,ik->ijk', s, s)
	return np.average(np.subtract(Temp1, Temp2), axis=2)

# ******************************************************************************
def CrossEntropy(Outputs, Targets):
	'''Cross entropy loss function
	Inputs:
		Outputs: a 2d matrix (mxn): m mini-batch, n output neurons
		Targets: a 2d matrix (mxn): m mini-batch, n expected values
	Returns: average derivative of cross entropy loss function
	'''
	# Loss    = -np.mean(Targets * np.log(Outputs) + (1 - Targets) * np.log(1 - Outputs))
	Loss    = -np.mean(np.add(np.multiply(Targets, np.log(Outputs)), np.multiply((1.0 - Targets), np.log(1.0 - Outputs))))
	return Loss

# ******************************************************************************
def CrossEntropyDeri(Outputs, Targets):
	'''Derivative of Cross entropy loss function
	Inputs:
		Outputs: a 2d matrix (mxn): m mini-batch, n output neurons
		Targets: a 2d matrix (mxn): m mini-batch, n expected values
	Returns: average derivative of cross entropy loss function
	'''
	DeriVector = np.add(np.divide(-Targets, (Outputs * np.log(10))), \
			np.divide((1.0 - Targets), (np.log(10) * (1 - Outputs))))
	return DeriVector

# ******************************************************************************
def LoadData(DataFolder):
	# **************************************************************************
	# display the information
	# **************************************************************************
	FunctionName = "LoadData()"

	# **************************************************************************
	# data file names
	# **************************************************************************
	TrainFile      = join(DataFolder, "Lab3_TrainVectors.csv")
	TrainLblFile   = join(DataFolder, "Lab3_TrainTargets.csv")
	TestFile       = join(DataFolder, "Lab3_TestVectors.csv")
	TestLblFile    = join(DataFolder, "Lab3_TestTargets.csv")

	# **************************************************************************
	# format the labels or targets
	# **************************************************************************
	TrainVectors   = np.loadtxt(TrainFile, delimiter=",", skiprows=1)
	TrainTargets   = np.loadtxt(TrainLblFile, delimiter=",", skiprows=1)
	TestData       = np.loadtxt(TestFile, delimiter=",", skiprows=1)
	TestTargets    = np.loadtxt(TestLblFile, delimiter=",", skiprows=1)
	return TrainVectors, TrainTargets, TestData, TestTargets

# ******************************************************************************
"""
 Feedforward Neural Network: http://neuralnetworksanddeeplearning.com/chap2.html
 Lab 3: Implementing Backpropagation Gradien Descent algorithm on a feedforward
 nework
"""
class BPFFNetwork:
	def __init__(self, InParams, Verbose=False):
		""" Initializes and constructs a feedforward network.
		Parameters are:
			- NetStructure  : a list of neurons [input, hidden, ..., outputs].
							  For example, [7, 3, 4, 3] means a FF network of 7x3x4x3
			- Epoch			: a number of trainings
			- Batch         : mimi-batch number. The default value should be 1.
			- Eta           : learning rate. The default value should be 0.001.
		"""
		# **********************************************************************
		# network name
		# **********************************************************************
		self.NetworkName  = "BPFFNetwork"

		# **********************************************************************
		# save the initial parameters
		# **********************************************************************
		self.NetStruct    = InParams["NetStruct"]
		self.Epoch        = InParams["Epoch"]
		self.Batch        = InParams["Batch"]
		self.LearnRateEta = InParams["LearnRateEta"]
		self.Verbose      = Verbose

		# Initialize Weights and Biases randomly - size determined by number of neurons in NetStruct

		self.W = [
			np.random.randn(self.NetStruct[0], self.NetStruct[1]),  # Weights for Layer 1
			np.random.randn(self.NetStruct[1], self.NetStruct[2]),  # Weights for Layer 2
			np.random.randn(self.NetStruct[2], self.NetStruct[3])  # Weights for Layer 3
		]

		self.B = [
			np.random.randn(1, self.NetStruct[1]),  # Biases for Layer 1
			np.random.randn(1, self.NetStruct[2]),  # Biases for Layer 2
			np.random.randn(1, self.NetStruct[3])  # Biases for Layer 3
		]
		...

	# **************************************************************************
	'''
		This function calculates a Forward pass of the network.
		Inputs: x[b, m] -> b: batch number, m: input neurons
		Output: o[b, n] -> b: batch number, n: output neurons
	'''
	def _Forward(self, x):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName    = "::_Forward():"

		# Layer 1 Input to Hidden1 using ReLU
		self.z1 = DotSumAddBias(x, self.W[0], self.B[0])
		a1 = ReLU(z1)

		# Layer 2 Hidden1 to Hidden2 using Sigmoid
		self.z2 = DotSumAddBias(a1, self.W[1], self.B[1])
		a2 = Sigmoid(z2)

		# Layer 3 Hidden2 to Output using Softmax
		self.z3 = DotSumAddBias(a2, self.W[2], self.B[2])
		a3 = Softmax(z3)

		return a3
		...

	# **************************************************************************
	'''
		This function calculates backpropagate errors:
			NablaW = nabla weights
			NablaB = nabla bias
	'''
	def _BackProp(self, Outputs, TrainTargets):

		# Error Calculations start with cost and pass back to each layer
		cost = CrossEntropyDeri(Outputs, TrainTargets)
		output_error = np.multiply(SoftmaxDerivative(np.sum(self.W[2], self.B[2])), cost)	#  Eq.s from module 3 pg. 55
		h2_error = np.dot(np.multiply(SigmoidDerivative(np.sum(self.W[1], self.B[1])), self.W[2].T), output_error)
		h1_error = np.dot(np.multiply(ReLUDerivative(np.sum(self.W[0], self.B[0])), self.W[1].T), h2_error)

		# Calculate Nablas
		NablaBiases = [h1_error, h2_error, output_error]
		NablaWeights = []
		NablaWeights[0] = np.multiply(self.z1.T, h1_error)
		NablaWeights[1] = np.multiply(self.z2.T, h2_error)
		NablaWeights[2] = np.multiply(self.z3.T, output_error)

		self._UpdateLayerWeights(NablaWeights, NablaBiases)
		...

	# **************************************************************************
	def _UpdateLayerWeights(self, NablaWeights, NablaBiases):
		# Update the weights by a factor of eta
		self.W -= np.multiply(self.LearnRateEta, NablaWeights)
		self.B -= np.multiply(self.LearnRateEta, NablaBiases)
		...

	# **************************************************************************
	'''
		This function trains the network
		Inputs:
			TrainData[l, m]		-> l: the number of input data, m: input neurons
			TrainTargets[l, n]	-> l: the number of input data, n: output neurons	
	'''
	def TrainNetwork(self, TrainData, TrainTargets):
		...

	# **************************************************************************
	def _CalPerf(self, Outputs, TestLbls):
		# **********************************************************************
		# get the total numbers of testings
		# **********************************************************************
		TestSize = len(Outputs)
		Results  = np.zeros(TestSize, dtype=int)
		Targets	 = np.zeros(TestSize, dtype=int)

		# **********************************************************************
		# reset the index
		# **********************************************************************
		for i in range(TestSize):
			# ******************************************************************
			# set the result array
			# ******************************************************************
			Results[i]	= np.argmax(Outputs[i])
			Targets[i]	= np.argmax(TestLbls[i])

		# **********************************************************************
		# calculate the classification
		# **********************************************************************
		NumFailures = float (np.count_nonzero(np.subtract(Targets, Results))) / (float (TestSize))
		return (1.0 - NumFailures) * 100.0

	# **************************************************************************
	'''
		This function test the network
		Inputs:
			TestData[k, l]		-> k: the number of input data, l: input neurons
			TestTargets[k, n]	-> k: the number of input data, m: output neurons
	'''
	def TestNetwork(self, TestData, TestTargets):
	   ...

# ******************************************************************************
if __name__ == "__main__":
	# **************************************************************************
	# load data to np array
	# **************************************************************************
	TrainData, TrainTargets, TestData, TestTargets = LoadData("Assets")
	# print("TrainData    = ", TrainData.shape)
	# print("TrainTargets    = ", TrainTargets.shape)

	# **************************************************************************
	# parameters for the network
	# **************************************************************************
	NetStruct    = [10, 15, 22, 4]
	Epoch        = 10
	Batch        = 1
	LearnRateEta = 0.001
	Verbose      = True

	# **************************************************************************
	# create a feedforward network
	# **************************************************************************
	NetFF  = BPFFNetwork(NetStruct=NetStruct, Epoch=Epoch, Batch=Batch,
			LearnRateEta=LearnRateEta, Verbose=Verbose)

	# **************************************************************************
	# train network
	# **************************************************************************
	NetFF.TrainNetwork(TrainData, TrainTargets)

	# **************************************************************************
	# test network
	# **************************************************************************
	Performance = NetFF.TestNetwork(TestData, TestTargets)
