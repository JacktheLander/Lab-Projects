# ******************************************************************************
# import modules
# ******************************************************************************
import sys

import numpy as np
import torch, os
import torch.nn as nn
# from progress.bar import Bar
from progressbar import progressbar

# ******************************************************************************
"""
* The reference of Echo State Network is from the website:
*	https://www.geeksforgeeks.org/echo-state-network-an-overview/
"""
class EchoStateNetwork(nn.Module):
	def __init__(self, Inputs=None, ReserSize= None, Outputs=None, SpecRadius=0.9, Verbose=False):
		# **********************************************************************
		super(EchoStateNetwork, self).__init__()

		# **********************************************************************
		# Save the network parameters
		# **********************************************************************
		self.Inputs		= Inputs
		self.ReserSize 	= ReserSize
		self.Outputs	= Outputs
		self.SpecRadius	= torch.tensor(SpecRadius, dtype=torch.float64)
		self.Verbose	= Verbose

		# **********************************************************************
		# Initializing input weight matrix in [0, 0.5]: [Inputs x ReserSize]
		# **********************************************************************
		# self.WIn	= np.random.rand(self.Inputs, self.ReserSize) - 0.5
		self.WIn	= torch.rand(self.Inputs, self.ReserSize, dtype=torch.float64) - 0.5

		# **********************************************************************
		# Initializing reservoir with random weights for reservoir matrix
		# **********************************************************************
		# self.WRes 	= np.random.rand(self.ReserSize, self.ReserSize) - 0.5
		self.WRes 	= torch.rand(self.ReserSize, self.ReserSize, dtype=torch.float64) - 0.5
		self.WRes  *= self.SpecRadius / torch.max(torch.abs(torch.linalg.eigvals(self.WRes)))

		# **********************************************************************
		# Initializing output maxtrix: [ReserSize x Outputs]
		# **********************************************************************
		# self.WOut  = np.random.rand(self.ReserSize, self.Outputs)
		self.WOut	= None

	# **************************************************************************
	def _ReservoirStates(self, TrainData):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName    = "ESN::_ReservoirStates()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			# display the information
			Msg = "...%-25s: get reservoir states ..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# Initialize reservoir states
		# **********************************************************************
		# ResStateX = np.zeros((len(TrainData), self.ReserSize))
		ResStateX = torch.zeros(len(TrainData), self.ReserSize, dtype=torch.float64)

		# **********************************************************************
		# set the reservoir states
		# **********************************************************************
		# with Bar(ProgressLabel) as bar:
		for t in progressbar(range(1, len(TrainData))):
			# ResStateX[t,:] = np.tanh(np.dot(self.WRes, ResStateX[t-1,:]) +
			# 		np.dot(self.WIn, TrainData[t]))
			ResStateX[t,:] = torch.tanh(torch.matmul(TrainData[t], self.WIn) +
					torch.matmul(ResStateX[t-1,:], self.WRes))
				# if (t % 5) == 0:
				# 	bar.next()
		return ResStateX

	# **************************************************************************
	def TrainReservoir(self, TrainData, TrainLabels):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName    = "ESN::TrainReservoir()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			# display the information
			Msg = "...%-25s: train ESN network ..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# Run reservoir with input Datasets
		# **********************************************************************
		ResStateX	= self._ReservoirStates(TrainData)

		# **********************************************************************
		# Train the output weights using pseudo-inverse
		# **********************************************************************
		# self.WOut	= np.dot(np.linalg.pinv(ResStateX), TrainLabels)
		self.WOut	= torch.matmul(torch.linalg.pinv(ResStateX), TrainLabels)

	# **************************************************************************
	def _CalPerf(self, Outputs, Targets):
		Predictions = np.argmax(Outputs, axis=1)
		# Targets 	= np.argmax(TestTargets, axis=1)
		Accuracy 	= np.mean(Predictions == Targets) * 100.0
		return Accuracy

	# **************************************************************************
	def TestReservoir(self, TestData, TestLabels):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName    = "ESN::TestReservoir()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			# display the information
			Msg = "...%-25s: test ESN network ..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# Run reservoir with input Datasets
		# **********************************************************************
		ResStateX	= self._ReservoirStates(TestData)

		# **********************************************************************
		# Make Predictions using the trained output weights
		# **********************************************************************
		ResOutputs	= torch.matmul(ResStateX, self.WOut)

		# print("ResOutputs	= ", ResOutputs.shape)
		# print("TestLabels	= ", TestLabels.shape)
		# exit()

		# **********************************************************************
		# calculate classification performance
		# **********************************************************************
		return self._CalPerf(ResOutputs.numpy(), TestLabels.numpy())

# ******************************************************************************
if __name__ == '__main__':
	# ******************************************************************************
	# import common module
	# ******************************************************************************
	DataFolder = os.path.abspath(os.path.join(os.path.dirname(__file__), r"C:\Users\Jaxba\PycharmProjects\Neuromorphic Computing\Datasets"))
	sys.path.append(DataFolder)
	import MNIST


	# **************************************************************************
	# a temporary fix for OpenMP
	# **************************************************************************
	os.environ["KMP_DUPLICATE_LIB_OK"]="True"

	# **************************************************************************
	# Device will determine whether to run the training on GPU or CPU.
	# **************************************************************************
	Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# **************************************************************************
	# Training and testing with the MNIST dataset
	# **************************************************************************
	NumTrains   = None
	NumTests    = None
	NumTrains   = 10000
	NumTests    = 1000
	ScaleVal    = 1.0
	ScaleFlag   = True
	TorchFlag   = True
	Verbose		= True

	# **************************************************************************
	# create a MNIST object
	# **************************************************************************
	InputDataSet = MNIST.MNIST(NumTrains=NumTrains, NumTests=NumTests, ScaleVal=ScaleVal, \
			Verbose=Verbose)
	InputDataSet.to(Device)

	# **************************************************************************
	# get training and testing Datasets
	# **************************************************************************
	TrainData, TrainLbls, TestData, TestLbls = InputDataSet.GetDataVectors(ScaleVal=1.0)

	# print("TrainData  = ", TrainData.shape)
	# print("TrainLbls    = ", TrainLbls.shape)
	# print("TestData   = ", TestData.shape)
	# print("TestLbls     = ", TestLbls.shape)
	# exit()

	# **************************************************************************
	# reservoir parameters
	# **************************************************************************
	NumTrains, Inputs	= TrainData.shape
	NumTests, Inputs	= TestData.shape
	NumTrains, Outputs	= TrainLbls.shape
	ReserSize	= Inputs * 2

	# **************************************************************************
	# instantiating an ESN
	# **************************************************************************
	ESN		= EchoStateNetwork(Inputs=Inputs, ReserSize=ReserSize, Outputs=Outputs,\
				Verbose=Verbose)
	ESN.to(Device)

	# **************************************************************************
	# Train ESN
	# **************************************************************************
	ESN.TrainReservoir(TrainData, TrainLbls)

	# **************************************************************************
	# Test ESN network
	# **************************************************************************
	Classification	= ESN.TestReservoir(TestData, TestLbls)
	print("Classification	= %.2f%%" % Classification)
