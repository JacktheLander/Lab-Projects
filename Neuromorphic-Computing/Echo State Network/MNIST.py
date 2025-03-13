# ******************************************************************************
# import modules
# ******************************************************************************
import sys, os
import struct
import torch
import torch.nn as nn
import numpy as np
from os.path import join, isdir, isfile
from MathUtils import LinearTransforming
from HelperFuncs import GetDataPath, CheckTempPath

# ******************************************************************************
# MNIST class
# ******************************************************************************
class MNIST(nn.Module):
	def __init__(self, NumTrains=None, NumTests=None, NumEpochs=1, ScaleVal=None, \
			ScaleFlag=True, ZeroMeanFlag=False, Force=False, Verbose=False):
		"""
		Python function for importing the MNIST Datasets set.  It returns an iterator
		of 2-tuples with the first element being the label and the second element
		being a numpy.uint8 2D array of pixel Datasets for the given image.
		"""
		# **********************************************************************
		super(MNIST, self).__init__()

		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNIST::__init__()"

		# **********************************************************************
		# set the class name
		# **********************************************************************
		self.ClassName = "Mnist"

		# **********************************************************************
		# save the parameters
		# **********************************************************************
		self.NumClasses = 10
		self.NumTrains  = NumTrains
		self.NumTests   = NumTests
		self.NumEpochs  = NumEpochs
		self.ScaleVal   = ScaleVal
		self.ScaleFlag  = ScaleFlag
		self.ZeroMeanFlag = ZeroMeanFlag
		self.Force		= Force
		self.Verbose    = Verbose

		# **********************************************************************
		# reset variables
		# **********************************************************************
		self.TotalTrainingImages = None
		self.TotalTrainingLabels = None
		self.TotalNumTrains      = 60000

		self.TotalTestingImages  = None
		self.TotalTestingLabels  = None
		self.TotalNumTests       = 10000

		self.TrainingImages      = None
		self.TrainingLabels      = None
		self.TestingImages       = None
		self.TestingLabels       = None
		self.VectorLength        = None

		self.DataPath			 = None
		self.ImageDataPath		 = None

		# **********************************************************************
		# maximum numbers of training and test Datasets
		# **********************************************************************
		self.MaxTrains  = self.TotalNumTrains
		self.MaxTests   = self.TotalNumTests

		# **********************************************************************
		# set the variables
		# **********************************************************************
		self.DatasetName = "MNIST"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			# display the information
			Msg = "\n==> Instantiating <%s>..." % (self.ClassName)
			print(Msg)

		# **********************************************************************
		# check the Datasets path
		# **********************************************************************
		self.DataPath = self._CheckPath()

		# **********************************************************************
		# set training and testing Datasets vectors
		# **********************************************************************
		self._BuildDataVectors()

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			# display the message
			Msg = "\n==> Instantiating <%s>..." % (self.ClassName)
			print(Msg)

			# display the information
			Msg = "...%-25s: Datasets location = %s" % (FunctionName, self.DataPath)
			print(Msg)

			# display the information
			Msg = "...%-25s: Classes = %d, Total trains = %d, total tests = %d" % \
					(FunctionName, self.NumClasses, self.TotalNumTrains, self.TotalNumTests)
			print(Msg)

			# display the information
			Msg = "...%-25s: DataFile = %s" % (FunctionName, self.DataFile)
			print(Msg)

			# display the information
			Msg = "...%-25s: Epochs = %d, Num Trains = %d, num tests = %d, vector length = %d" % \
					(FunctionName, self.NumEpochs, self.NumTrains, self.NumTests, self.VectorLength)
			print(Msg)

			# display the information
			Msg = "...%-25s: ScaleVal = %s, ZeroMeanFlag = %s" % \
					(FunctionName, str(self.ScaleVal), str(self.ZeroMeanFlag))
			print(Msg)

	# **************************************************************************
	# methods of the class
	# **************************************************************************
	def _CheckPath(self):
		# **********************************************************************
		# display the information
		# **********************************************************************
		FunctionName = "MNIST::_CheckPath()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			# display the information
			Msg = "...%-25s: checking the Datasets path ..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# Get the Datasets path
		# **********************************************************************
		# DataPath = GetDataPath(DatasetName=self.DatasetName)
		DataPath = GetDataPath()

		if not isdir(DataPath):
			# format error message
			Msg = "%s: Datasets path is invalid => <%s>" % (FunctionName, DataPath)
			raise ValueError(Msg)
		return DataPath

	# **************************************************************************
	def _LoadMNIST(self, InputType):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNIST::_LoadMNIST()"

		# **********************************************************************
		# check the flag
		# **********************************************************************
		if self.Verbose:
			# display the message
			Msg = "...%-25s: loading <%s> Datasets from <%s>..." % (FunctionName, InputType, \
					self.ImageDataPath)
			print(Msg)

		# **********************************************************************
		# check the input type
		# **********************************************************************
		if InputType == "Training":
			ImageFileName = join(self.ImageDataPath, "train-images-idx3-ubyte")
			LabelFileName = join(self.ImageDataPath, "train-labels-idx1-ubyte")

		elif InputType == "Testing":
			ImageFileName = join(self.ImageDataPath, "t10k-images-idx3-ubyte")
			LabelFileName = join(self.ImageDataPath, "t10k-labels-idx1-ubyte")
		else:
			raise ValueError("InputType must be <Testing> or <Training>")

		# **********************************************************************
		# Load everything in some numpy arrays
		# **********************************************************************
		with open(LabelFileName, "rb") as LabelFd:
			Magic, Num  = struct.unpack(">II", LabelFd.read(8))
			Labels      = np.fromfile(LabelFd, dtype=np.int8)

		# **********************************************************************
		# load the image files
		# **********************************************************************
		with open(ImageFileName, "rb") as ImageFd:
			Magic, Num, Rows, Cols = struct.unpack(">IIII", ImageFd.read(16))
			Images = np.fromfile(ImageFd, dtype=np.uint8).reshape(len(Labels), Rows, Cols)

		return Images, Labels

	# **************************************************************************
	def _SetTrainLabels(self, Labels):
		# **********************************************************************
		# get the number of labels
		# **********************************************************************
		NumLabels = len(Labels)

		# **********************************************************************
		# set the label matrix
		# **********************************************************************
		LabelMatrix = torch.zeros((NumLabels, self.NumClasses), dtype=torch.double)
		for i in range(NumLabels):
			# print(i, ": Labels[i] = ", Labels[i])
			LabelMatrix[i,Labels[i]] = 1

		return LabelMatrix

	# **************************************************************************
	def _ReadMNISTData(self):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNIST::_ReadMNISTData()"

		# **********************************************************************
		# check the flag
		# **********************************************************************
		if self.Verbose:
			# display the message
			Msg = "...%-25s: loading MNIST Datasets..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# loading training images and labels
		# **********************************************************************
		self.TotalTrainingImages, self.TotalTrainingLabels = self._LoadMNIST("Training")
		self.TotalNumTrains     = len(self.TotalTrainingImages)

		# **********************************************************************
		# loading testing images and labels
		# **********************************************************************
		self.TotalTestingImages, self.TotalTestingLabels   = self._LoadMNIST("Testing")
		self.TotalNumTests      = len(self.TotalTestingImages)

	# **************************************************************************
	def _GetClassGroupIndices(self, Vector):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNIST::_GetClassGroupIndices()"

		# **********************************************************************
		# check the flag
		# **********************************************************************
		if self.Verbose:
			# display the message
			Msg = "...%-25s: set class indices for <%s>..." % (FunctionName, str(Vector.shape))
			print(Msg)

		# **********************************************************************
		# set the list of class indices
		# **********************************************************************
		ClassGroupList	= [(Vector == i).nonzero()[0] for i in range(self.NumClasses)]
		return ClassGroupList

	# **************************************************************************
	def _SelectSubset(self, Dataset, Labels, NumSelect):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNIST::_SelectSubset()"

		# **********************************************************************
		# check the flag
		# **********************************************************************
		if self.Verbose:
			# display the message
			Msg = "...%-25s: set Datasets indices ..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# get class group index list
		# **********************************************************************
		ClassGroupList = self._GetClassGroupIndices(Labels)

		# **********************************************************************
		# set the selected number for each class
		# **********************************************************************
		ClassNoSelect	= int(NumSelect / self.NumClasses)

		# **********************************************************************
		# set the Datasets indices
		# **********************************************************************
		IndMatrix = np.zeros((self.NumClasses, ClassNoSelect), dtype=int)
		for i in range(self.NumClasses):
			Indices		 = np.random.choice(ClassGroupList[i], size=ClassNoSelect, replace=False)
			IndMatrix[i] = np.sort(Indices)

		Indices	= IndMatrix.flatten()
		np.random.shuffle(Indices)
		return Dataset[Indices], Labels[Indices]

	# **************************************************************************
	def _TrainingAndTestingSets(self):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNIST::_TrainingAndTestingSets()"

		# **********************************************************************
		# check the flag
		# **********************************************************************
		if self.Verbose:
			# display the message
			Msg = "...%-25s: loading MNIST Datasets..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# check the variable before reading from the file
		# **********************************************************************
		if (self.TotalTrainingImages is None) or (self.TotalTestingImages is None):
			# loading from file
			self._ReadMNISTData()

		# **********************************************************************
		# check the number of training digits
		# **********************************************************************
		if (self.NumTrains is None) or (self.NumTrains == self.TotalNumTrains):
			# use the total number of training digits
			self.TrainingImages = self.TotalTrainingImages
			self.TrainingLabels = self.TotalTrainingLabels
		else:
			# ******************************************************************
			# set the training Datasets set
			# ******************************************************************
			self.TrainingImages, self.TrainingLabels = self._SelectSubset(self.TotalTrainingImages, \
					self.TotalTrainingLabels, self.NumTrains)

		# **********************************************************************
		# check the number of testing digits
		# **********************************************************************
		if (self.NumTests is None) or (self.NumTests == self.TotalNumTests):
			# use the total number of training digits
			self.TestingImages  = self.TotalTestingImages
			self.TestingLabels  = self.TotalTestingLabels
		else:
			# ******************************************************************
			# set the testing Datasets set
			# ******************************************************************
			self.TestingImages, self.TestingLabels = self._SelectSubset(self.TotalTestingImages, \
					self.TotalTestingLabels, self.NumTests)

	# **************************************************************************
	def _BuildDataVectors(self):
		# **********************************************************************
		# set the funtion name
		# **********************************************************************
		FunctionName = "MNIST::_BuildDataVectors()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			# display the information
			Msg = "...%-25s: building MNIST training and testing Datasets..." % \
					(FunctionName)
			print(Msg)

		# **********************************************************************
		# check the number of trainings and testings
		# **********************************************************************
		if self.NumTrains is None:
			self.NumTrains  = self.TotalNumTrains
		if self.NumTests is None:
			self.NumTests   = self.TotalNumTests

		# **********************************************************************
		# set the file name
		# **********************************************************************
		FileName = "MNIST_Train_%d_Test_%d.npz" % (self.NumTrains, self.NumTests)

		# **********************************************************************
		# check the platform to set the Datasets path
		# **********************************************************************
		TempPath = CheckTempPath(FileName)
		if TempPath is not None:
			# ******************************************************************
			# update Datasets path
			# ******************************************************************
			self.DataPath   = TempPath

		# **********************************************************************
		# set the Datasets file
		# **********************************************************************
		self.DataFile = join(self.DataPath, FileName)

		# **********************************************************************
		# check the file status
		# **********************************************************************
		if isfile(self.DataFile) and not self.Force:
			# ******************************************************************
			# display the information
			# ******************************************************************
			Msg = "...%-25s: loading Datasets from file <%s>..." % \
					(FunctionName, self.DataFile)
			print(Msg)

			# ******************************************************************
			# loading Datasets from file
			# ******************************************************************
			Data                = np.load(self.DataFile)
			self.NumTrains      = Data["NumTrains"]
			self.TrainingImages = Data["Train"]
			self.TrainingLabels = Data["TrainLbls"]
			self.NumTests       = Data["NumTests"]
			self.TestingImages  = Data["Test"]
			self.TestingLabels  = Data["TestLbls"]

		else:
			# ******************************************************************
			# set the image Datasets path
			# ******************************************************************
			self.ImageDataPath	= join(self.DataPath, self.DatasetName)

			# ******************************************************************
			# build the training and testing set
			# ******************************************************************
			self._TrainingAndTestingSets()

			# ******************************************************************
			# display the message
			# ******************************************************************
			if self.Verbose:
				# display the information
				Msg = "...%-25s: saving Datasets to file <%s>..." % \
						(FunctionName, self.DataFile)
				print(Msg)

			# ******************************************************************
			# save to file
			# ******************************************************************
			np.savez_compressed(self.DataFile, NumTrains=self.NumTrains,
					Train=self.TrainingImages, TrainLbls=self.TrainingLabels,\
					NumTests=self.NumTests, Test=self.TestingImages, \
					TestLbls=self.TestingLabels)

		# **********************************************************************
		# set the vector length
		# **********************************************************************
		NumTrain, x, y      = self.TrainingImages.shape
		self.VectorLength   = x * y

		# **********************************************************************
		# check for torch flag
		# **********************************************************************
		if not torch.is_tensor(self.TrainingImages):
			self.TrainingImages = torch.from_numpy(self.TrainingImages)
			self.TrainingLabels = torch.from_numpy(self.TrainingLabels)
			self.TestingImages  = torch.from_numpy(self.TestingImages)
			self.TestingLabels  = torch.from_numpy(self.TestingLabels)

	# **************************************************************************
	def _ZeroMeanInputData(self, ScaleVal=None):
		# set the function name
		FunctionName = "MNIST::_ZeroMeanInputData()"

		# check the flag
		if self.Verbose:
			# display the message
			Msg = "...%-25s: loading zero-mean MNIST dataset..." % (FunctionName)
			print(Msg)

		# check the scale value
		if self.ScaleVal is None:
			# format error message
			Msg = "%s: ScaleVal = <%s> => invalid" % (FunctionName, str(self.ScaleVal))
			raise ValueError(Msg)

		# **********************************************************************
		# check the scale value
		# **********************************************************************
		if ScaleVal is None:
			if self.ScaleVal is None:
				# format error message
				Msg = "%s: ScaleVal = <%s> => invalid" % (FunctionName, str(self.ScaleVal))
				raise ValueError(Msg)
			else:
				AmpRange    = self.ScaleVal
		else:
			AmpRange    = ScaleVal

		# **********************************************************************
		# set the transforming y-range
		# **********************************************************************
		YRange      = [-AmpRange, AmpRange]
		BiasFlag    = True
		# print("AmpRange = ", AmpRange)

		# **********************************************************************
		# zero-mean the training images and linearly transforming Datasets
		# **********************************************************************
		InputData   = np.subtract(self.TrainingImages, np.mean(self.TrainingImages))

		# get the image information
		NumTrains, x, y = InputData.shape

		# calculate the image size
		ImageSize = self.VectorLength

		# linearly transforming dataset
		TrainingData, Alpha, Beta = LinearTransforming(InputData, YRange, BiasFlag=BiasFlag, \
				Verbose=self.Verbose)

		# print("Before   = ", self.TrainingImages)
		# print("ZeroMean = ", InputData)
		# print("After    = ", TrainingData)

		# set the number of epochs
		NumEpochs   = self.NumEpochs

		# check the number of epoch
		if NumEpochs > 1:
			# reset the TrainingImages
			TrainingImages = np.zeros((NumEpochs, NumTrains, ImageSize))

			# copy the Datasets
			for i in range(NumEpochs):
				TrainingImages[i] = TrainingData
		else:
			# set the training Images with epoch = 1
			TrainingImages = TrainingData.reshape(NumEpochs, NumTrains, ImageSize)

		# **********************************************************************
		# set the labels
		# **********************************************************************
		TrainingLabels = self._SetTrainLabels(self.TrainingLabels)

		# **********************************************************************
		# zero-mean the testing images and linearly transforming Datasets
		# **********************************************************************
		InputData   = np.subtract(self.TestingImages, np.mean(self.TestingImages))

		# get the image information
		NumTests, x, y = InputData.shape

		# calculate the image size
		ImageSize = self.VectorLength

		# linearly transforming dataset
		TestingData, Alpha, Beta = LinearTransforming(InputData, YRange, BiasFlag=BiasFlag, \
				Verbose=self.Verbose)

		# set the testing Images
		TestingImages = TestingData.reshape(NumTests, ImageSize)

		# set the testing labels
		TestingLabels   = self.TestingLabels

		# set the Datasets set
		return TrainingImages, TrainingLabels, TestingImages, TestingLabels

	# **************************************************************************
	def _RangeInputData(self, ScaleVal=None, BiasFlag=False, TrainLblFlag=True):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNIST::_RangeInputData()"

		# **********************************************************************
		# check the flag
		# **********************************************************************
		if self.Verbose:
			# display the message
			Msg = "...%-25s: loading range MNIST dataset..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# check the scale value
		# **********************************************************************
		if self.ScaleVal is None:
			# format error message
			Msg = "%s: ScaleVal = <%s> => invalid" % (FunctionName, str(self.ScaleVal))
			raise ValueError(Msg)

		# **********************************************************************
		# check the scale value
		# **********************************************************************
		if ScaleVal is None:
			if self.ScaleVal is None:
				# format error message
				Msg = "%s: ScaleVal = <%s> => invalid" % (FunctionName, str(self.ScaleVal))
				raise ValueError(Msg)
			else:
				AmpRange    = self.ScaleVal
		else:
			AmpRange    = ScaleVal

		# **********************************************************************
		# set the transforming y-range
		# **********************************************************************
		YRange      = [-AmpRange, AmpRange]

		# **********************************************************************
		# get the image information
		# **********************************************************************
		NumTrains, x, y = self.TrainingImages.shape
		NumTests, x, y 	= self.TestingImages.shape

		# **********************************************************************
		# calculate the image size
		# **********************************************************************
		ImageSize = self.VectorLength

		# **********************************************************************
		# linearly transforming Datasets
		# **********************************************************************
		TrainingImages, Alpha, Beta = LinearTransforming(self.TrainingImages, YRange, BiasFlag=BiasFlag, \
				Verbose=self.Verbose)
		TrainingImages	= torch.reshape(TrainingImages, (NumTrains,	ImageSize))

		# # **********************************************************************
		# # set the number of epochs
		# # **********************************************************************
		# NumEpochs   = self.NumEpochs

		# **********************************************************************
		# check the number of epoch
		# **********************************************************************
		# if self.TorchFlag:
		# 	TrainingInputs  = TrainingData
		# else:
		# 	if NumEpochs > 1:
		# 		# reset the TrainingImages
		# 		TrainingImages = np.zeros((NumEpochs, NumTrains, ImageSize))
		#
		# 		# copy the Datasets
		# 		for i in range(NumEpochs):
		# 			TrainingImages[i] = TrainingData
		# 	else:
		# 		# set the training Images with epoch = 1
		# 		TrainingImages = TrainingData.reshape(NumEpochs, NumTrains, ImageSize)

		# **********************************************************************
		# set the labels
		# **********************************************************************
		TrainingLabels = self._SetTrainLabels(self.TrainingLabels)

		# **********************************************************************
		# linearly transforming testing Datasets
		# **********************************************************************
		TestingImages, Alpha, Beta = LinearTransforming(self.TestingImages, YRange, BiasFlag=BiasFlag, \
				Verbose=self.Verbose)
		TestingImages	= torch.reshape(TestingImages, (NumTests, ImageSize))
		TestingLabels   = self.TestingLabels

		# **********************************************************************
		# return values
		# **********************************************************************
		# print("TrainingImages   = ", TrainingImages.shape)
		# print("TrainingLabels   = ", TrainingLabels.shape)
		# print("TestingImages    = ", TestingImages.shape)
		# print("TestingLabels    = ", TestingLabels.shape)
		# exit()

		# set the Datasets set
		return TrainingImages, TrainingLabels, TestingImages, TestingLabels

	# **************************************************************************
	def _NormalInputDataTorch(self, ScaleVal=None, TrainLblFlag=True):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNIST::_NormalInputDataTorch()"

		# **********************************************************************
		# check the flag
		# **********************************************************************
		if self.Verbose:
			# display the message
			Msg = "...%-25s: loading and scaling MNIST Datasets to <%s>..." % (FunctionName, \
					str(self.ScaleVal))
			print(Msg)

		# **********************************************************************
		# check the scale value
		# **********************************************************************
		if ScaleVal is None:
			if self.ScaleVal is None:
				# format error message
				Msg = "%s: ScaleVal = <%s> => invalid" % (FunctionName, str(self.ScaleVal))
				raise ValueError(Msg)
			else:
				AmpScale    = self.ScaleVal
		else:
			AmpScale    = ScaleVal

		# **********************************************************************
		# scaling the training images
		# **********************************************************************
		# print("AmpScale = ", AmpScale)
		if torch.is_tensor(self.TrainingImages):
			Images  = self.TrainingImages
		else:
			Images  = torch.from_numpy(self.TrainingImages)

		# **********************************************************************
		# check the flag
		# **********************************************************************
		if self.ScaleFlag:
			# calculate the input scaling
			InputScaling = AmpScale/torch.max(Images)
		else:
			InputScaling = 1.0 / self.ScaleVal

		# **********************************************************************
		# set the number of epochs
		# **********************************************************************
		NumEpochs   = self.NumEpochs

		# **********************************************************************
		# get the image information
		# **********************************************************************
		NumTrains, x, y = Images.shape

		# **********************************************************************
		# calculate the image size
		# **********************************************************************
		ImageSize = self.VectorLength

		# **********************************************************************
		# set the training Images
		# **********************************************************************
		TrainingImages  = torch.multiply(torch.reshape(Images, (NumTrains,\
				ImageSize)), InputScaling)
		# TrainingImages  = from_numpy(self.TrainingImages, InputScaling)
		TrainingImages  = TrainingImages.double()

		# **********************************************************************
		# set the labels
		# **********************************************************************
		if TrainLblFlag:
			TrainingLabels  = self._SetTrainLabels(self.TrainingLabels)
		else:
			TrainingLabels  = self.TrainingLabels

		# **********************************************************************
		# get the image information
		# **********************************************************************
		if torch.is_tensor(self.TestingImages):
			Images  = self.TestingImages
		else:
			Images  = torch.from_numpy(self.TestingImages)
		NumTests, x, y = Images.shape

		# **********************************************************************
		# calculate the image size
		# **********************************************************************
		ImageSize = self.VectorLength

		# **********************************************************************
		# set the testing Images
		# **********************************************************************
		TestingImages   = torch.multiply(torch.reshape(Images, (NumTests, \
				ImageSize)), InputScaling)
		TestingImages   = TestingImages.double()
		# TestingImages = torch.multiply(self.TestingImages, InputScaling)

		# **********************************************************************
		# set the testing labels
		# **********************************************************************
		# TestingLabels   = torch.reshape(self.TestingLabels, (NumEpochs, NumTests))
		TestingLabels   = self.TestingLabels

		# **********************************************************************
		# return values
		# **********************************************************************
		# print("TrainingImages   = ", TrainingImages.shape)
		# print("TrainingLabels   = ", TrainingLabels.shape)
		# print("TestingImages    = ", TestingImages.shape)
		# print("TestingLabels    = ", TestingLabels.shape)
		# exit()
		return TrainingImages, TrainingLabels, TestingImages, TestingLabels

	# **************************************************************************
	def _NormalInputDataNumpy(self, ScaleVal=None, TrainLblFlag=True):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNIST::_NormalInputDataNumpy()"

		# **********************************************************************
		# check the flag
		# **********************************************************************
		if self.Verbose:
			# display the message
			Msg = "...%-25s: loading and scaling MNIST Datasets to <%s>..." % (FunctionName, \
					str(self.ScaleVal))
			print(Msg)

		# **********************************************************************
		# check the scale value
		# **********************************************************************
		if ScaleVal is None:
			if self.ScaleVal is None:
				# format error message
				Msg = "%s: ScaleVal = <%s> => invalid" % (FunctionName, str(self.ScaleVal))
				raise ValueError(Msg)
			else:
				AmpScale    = self.ScaleVal
		else:
			AmpScale    = ScaleVal

		# **********************************************************************
		# scaling the training images
		# **********************************************************************
		# print("AmpScale = ", AmpScale)
		Images  = self.TrainingImages
		# check the flag
		if self.ScaleFlag:
			# calculate the input scaling
			InputScaling = AmpScale/np.amax(Images)
		else:
			InputScaling = 1.0 / self.ScaleVal

		# **********************************************************************
		# get the image information
		# **********************************************************************
		NumTrains, x, y = Images.shape

		# **********************************************************************
		# calculate the image size
		# **********************************************************************
		ImageSize = self.VectorLength

		# **********************************************************************
		# set the training Images
		# **********************************************************************
		TrainingData = np.multiply(Images.reshape(NumTrains, ImageSize), InputScaling)

		# print("Before   = ", self.TrainingImages)
		# print("ZeroMean = ", InputData)
		# print("After    = ", TrainingData)

		# **********************************************************************
		# set the number of epochs
		# **********************************************************************
		NumEpochs   = self.NumEpochs

		# **********************************************************************
		# check the number of epoch
		# **********************************************************************
		if NumEpochs > 1:
			# reset the TrainingImages
			TrainingImages = np.zeros((NumEpochs, NumTrains, ImageSize))

			# copy the Datasets
			for i in range(NumEpochs):
				TrainingImages[i] = TrainingData
		else:
			# set the training Images with epoch = 1
			TrainingImages = TrainingData.reshape(NumEpochs, NumTrains, ImageSize)

		# **********************************************************************
		# set the labels
		# **********************************************************************
		if TrainLblFlag:
			TrainingLabels  = self._SetTrainLabels(self.TrainingLabels)
		else:
			TrainingLabels  = self.TrainingLabels

		# **********************************************************************
		# scaling the testing images
		# **********************************************************************
		Images  = self.TestingImages

		# **********************************************************************
		# get the image information
		# **********************************************************************
		NumTests, x, y = Images.shape

		# **********************************************************************
		# calculate the image size
		# **********************************************************************
		ImageSize = self.VectorLength

		# **********************************************************************
		# set the testing Images
		# **********************************************************************
		TestingImages = np.multiply(Images.reshape(NumTests, ImageSize), InputScaling)

		# **********************************************************************
		# set the testing labels
		# **********************************************************************
		TestingLabels   = self.TestingLabels

		# **********************************************************************
		# set the Datasets set
		# **********************************************************************
		return TrainingImages, TrainingLabels, TestingImages, TestingLabels

	# **************************************************************************
	def _NormalInputData(self, ScaleVal=None, TrainLblFlag=True):
		return self._NormalInputDataTorch(ScaleVal=ScaleVal, TrainLblFlag=TrainLblFlag)

		# # **********************************************************************
		# # check the flag
		# # **********************************************************************
		# if self.TorchFlag:
		# 	return self._NormalInputDataTorch(ScaleVal=ScaleVal, TrainLblFlag=TrainLblFlag)
		# else:
		# 	return self._NormalInputDataNumpy(ScaleVal=ScaleVal, TrainLblFlag=TrainLblFlag)

	# **************************************************************************
	def GetDataVectors(self, ScaleVal=None, BiasFlag=False, TrainLblFlag=True):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNIST::GetDataVectors()"

		# **********************************************************************
		# check the flag
		# **********************************************************************
		if self.Verbose:
			# display the message
			Msg = "...%-25s: loading MNIST dataset, ZeroMean = <%s>..." % (FunctionName, \
					str(self.ZeroMeanFlag))
			print(Msg)

		# **********************************************************************
		# check the scale value
		# **********************************************************************
		if ScaleVal is None:
			ScaleVal    = self.ScaleVal

		# **********************************************************************
		# check the flag:
		# **********************************************************************
		if self.ZeroMeanFlag:
			# set the dataset to a zero mean and transforming it
			return self._ZeroMeanInputData(ScaleVal=ScaleVal)
		else:
			if BiasFlag:
				return self._RangeInputData(ScaleVal=ScaleVal, BiasFlag=BiasFlag, TrainLblFlag=TrainLblFlag)
			else:
				# scaling the dataset
				return self._NormalInputData(ScaleVal=ScaleVal, TrainLblFlag=TrainLblFlag)

	# **************************************************************************
	def GetMnistInformation(self):
		return self.VectorLength, self.NumClasses

	# **************************************************************************
	def GetDatasetInformation(self):
		return self.GetMnistInformation()

	# **************************************************************************
	def GetNumEpoch(self):
		return self.NumEpochs

	# **************************************************************************
	def LoadMNISTData(self, Verbose=False):
		return [self._LoadMNIST("Training"), self._LoadMNIST("Testing")]

	# **************************************************************************
	def GetTrainingData(self):
		return self.TrainingImages, self.TrainingLabels

	# **************************************************************************
	def GetTestingData(self):
		return self.TestingImages, self.TestingLabels

	# **************************************************************************
	def GetNumTrainsAndTests(self):
		return self.NumTrains, self.NumTests

	# **************************************************************************
	def GetScaleVal(self):
		return self.ScaleVal

	# **************************************************************************
	def GetAllData(self):
		# set the function name
		FunctionName = "MNIST::GetAllData()"

		# check the flag
		if self.Verbose:
			# display the message
			Msg = "...%-25s: getting all MNIST dataset ..." % (FunctionName)
			print(Msg)

		# check the variable
		if self.TotalTrainingImages is None:
			# loading training images and labels
			self.TotalTrainingImages, self.TotalTrainingLabels = self._LoadMNIST("Training")
			self.TotalNumTrains     = len(self.TotalTrainingImages)

		# check the variable
		if self.TotalTestingImages is None:
			# loading testing images and labels
			self.TotalTestingImages, self.TotalTestingLabels   = self._LoadMNIST("Testing")
			self.TotalNumTests      = len(self.TotalTestingImages)

		# flattening training Datasets
		NumTests, x, y  = self.TotalTrainingImages.shape
		TrainingData    = self.TotalTrainingImages.reshape(NumTests, x * y)

		# flattening testing Datasets
		NumTests, x, y  = self.TotalTestingImages.shape
		TestingData     = self.TotalTestingImages.reshape(NumTests, x * y)

		# combining training and testing Datasets
		AllData         = np.vstack((TrainingData, TestingData))
		AllDataLabels   = np.hstack((self.TotalTrainingLabels, self.TotalTestingLabels))
		return AllData, AllDataLabels

	# **************************************************************************
	def GetDataSetName(self):
		return self.ClassName

	# **************************************************************************
	def GetMaxTrainTest(self):
		return self.MaxTrains, self.MaxTests

	# **************************************************************************
	def ResetVerboseFlag(self):
		# reset verbose flag
		self.Verbose = False

	# **************************************************************************
	def GetDatasetInfo(self):
		return {"Inputs": self.VectorLength, "NumClass": self.NumClasses}

# ******************************************************************************
def _ProcessArguments():
	# process the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--Force", type=bool, default=False,
			help = "Directory for output")

	# # MemCapacitive model
	# ap.add_argument("-mc", "--McModel", type=str, default=None,
	# 		help = "Memcapacitive model.")
	#
	# # memristive model
	# ap.add_argument("-mr", "--MrModel", type=str, default=None,
	# 		help = "Memristive model.")

	# # add argument for graph name
	# ap.add_argument("-app", "--AppType", type=str, required=True,
	#         help = "Application type < SpokenDigits | Mnist | Cifar10 | Timit >.")

	# # add argument for graph name
	# ap.add_argument("-ip", "--ClusterIp", type=str, default=None,
	# 		help = "Cluster node IP address.")

	# # add argument for saving each step
	# ap.add_argument("-ncpu", "--Ncpu", type=int, default=None,
	#         help = "Number of CPUs for parrallel processing. The default value is none.")

	# # add argument for one device option
	# ap.add_argument("-per", "--PerMemC", type=float, default=None,
	# 		help = "either 0 or 100.")
	#
	# # add argument for number of training images
	# ap.add_argument("-ntr", "--NumTrains", type=int, default=None,
	# 		help = "Number of training images. Default is None.")
	#
	# # add argument for number of testing images
	# ap.add_argument("-nts", "--NumTests", type=int, default=None,
	# 		help = "Number of testing images. Default is None.")
	#
	# # add argument for a runtime in minutes
	# ap.add_argument("-tm", "--RunTimeM", type=float, default=0.0,
	# 		help = "Run time in minutes.")
	#
	# # add argument for a runtime in hours
	# ap.add_argument("-th", "--RunTimeH", type=float, default=0.0,
	# 		help = "Run time in hours.")
	#
	# # add argument for a runtime in days
	# ap.add_argument("-td", "--RunTimeD", type=float, default=0.0,
	# 		help = "Run time in days.")
	#
	# # add argument for simulation option
	# ap.add_argument("-dtLo", "--dtLo", type=float, default=None,
	# 		help = "Low limit of the timestep.")
	#
	# # add argument for simulation option
	# ap.add_argument("-dtHi", "--dtHi", type=float, default=None,
	# 		help = "High limit of the timestep.")
	#
	# # add argument for simulation option
	# ap.add_argument("-dt", "--TimeStep", type=float, default=None,
	# 		help = "Timestep pulse.")
	#
	# # add argument for simulation option
	# ap.add_argument("-AmpLo", "--AmpLo", type=float, default=None,
	# 		help = "Low limit of signal amplitude.")
	#
	# # add argument for simulation option
	# ap.add_argument("-AmpHi", "--AmpHi", type=float, default=None,
	# 		help = "High limit of signal amplitude.")
	#
	# # add argument for simulation option
	# ap.add_argument("-Pgpu", "--Pgpu", type=int, default=None,
	# 		help = "Number of processes per GPUDevice. The default value is none.")
	#
	# # add argument for simulation option
	# ap.add_argument("-nT", "--NumTrials", type=int, default=None,
	# 		help = "Number of trials. The default value is none.")
	#
	# # add argument for trial folder name
	# ap.add_argument("-tf", "--TrialFolder", type=str, default=None,
	# 		help = "Trial folder.")
	#
	# # add argument for instances
	# ap.add_argument("-nI", "--Instances", type=int, default=5,
	# 		help = "Number of instances for average results. Default value is 5.")

	# get the arguments
	args = ap.parse_args()
	return args

# ******************************************************************************
# def DisplayDigit():

# ******************************************************************************
if __name__ == '__main__':
	# from MathUtils import LinearTransforming
	# import matplotlib.pyplot as plt
	# from matplotlib import rcParams
	import pandas as pd
	import argparse
	# rcParams.update({"figure.autolayout": True})

	# **************************************************************************
	# process the arguments
	# **************************************************************************
	args = _ProcessArguments()

	# **************************************************************************
	# set the variable
	# **************************************************************************
	NumTrains   = None
	NumTests    = None
	NumTrains   = 5000
	NumTests    = 500
	# NumTrains   = 3000
	# NumTrains   = 6000
	# NumTests    = 1000

	Verbose     = True
	ScaleVal    = 1.0
	ScaleFlag   = True
	NumEpochs   = 1
	# ZeroMeanFlag = True
	ZeroMeanFlag = False
	Verbose     = True

	# **************************************************************************
	# get MNIST Datasets set
	# **************************************************************************
	Force = args.Force
	InputDataSet = MNIST(NumTrains=NumTrains, NumTests=NumTests, NumEpochs=NumEpochs, \
			ScaleVal=ScaleVal, ZeroMeanFlag=ZeroMeanFlag, Force=Force, Verbose=Verbose)
	ScaleVal     = InputDataSet.GetScaleVal()
	# print("Current ScaleVal = ", ScaleVal)

	# **************************************************************************
	# get training and testing Datasets
	# **************************************************************************
	BiasFlag	= False
	TrainInputs, TrainLbls, TestInputs, TestLbls = InputDataSet.GetDataVectors(ScaleVal=1.0, \
			BiasFlag=BiasFlag)

	print("Train input images   = ", TrainInputs.shape)
	print("Train input labels   = ", TrainLbls.shape)
	print("Test input images    = ", TestInputs.shape)
	print("Test input labels    = ", TestLbls.shape)

	# print(pd.DataFrame(TrainInputs))
	# print("TestInputs           = ", np.nonzero(TrainInputs))
	exit()

	# **************************************************************************
	# change the format of the training dataset
	# **************************************************************************
	# x, y, z     = TrainInputs.shape
	# TrainDataSet    = np.array([np.hstack((TrainInputs[0,i,:], TrainLbls[0,i,:])) for i in range(y)])

	s
	# **************************************************************************
	# change the format of the testing
	# **************************************************************************
	# NumTests    = len(TestLbls)
	# TestOutputs = np.zeros((NumTests, 10))

	# # set the testing label numpy arrays
	# for i in range(NumTests):
	#     ExptOutput = TestLbls[i]
	#     TestOutputs[i, ExptOutput]  = 1.0
	#
	# TestDataSet = np.asarray([np.hstack((TestInputs[i,:], TestOutputs[i,:])) for i in range(NumTests)])

	exit()

	# # get the training and testing dataset
	# # TrainImgs, TrainLbls, TestImgs, TestLbls = MNISTData.GetDataVectors(ScaleVal=ScaleVal, \
	# #         ScaleFlag=ScaleFlag, Verbose=Verbose)
	# TrainImgs, TrainLbls, TestImgs, TestLbls = MNISTData.GetDataVectors(ScaleVal=ScaleVal, \
	#         ScaleFlag=ScaleFlag, ZeroMeanFlag=ZeroMeanFlag, Verbose=Verbose)
	#
	# print("TrainImgs   = ", TrainImgs.shape)
	# print("TrainLbls   = ", TrainLbls.shape)
	# print("TestImgs    = ", TestImgs.shape)
	# print("TestLbls    = ", TestLbls.shape)

	AllData, AllDataLabels = MNISTData.GetAllData()
	print("AllData        = ", AllData.shape)
	print("AllDataLabels  = ", AllDataLabels.shape)

	exit()

	# set the line width
	LineWidth = 1.5
	# LineWidth = 2.0
	# LineWidth = 2.5

	# set the font family
	FontSize = 12
	# FontSize = 14
	# FontSize = 16
	# FontSize = 18
	# FontSize = 20
	# FontSize = 22
	# FontSize = 24
	# FontSize = 28
	font = {"family": "Times New Roman", "size": FontSize}
	plt.rc("font", **font)  # pass in the font dict as kwargs

	# # check platform
	# if (Platform == "linux2") or (Platform == "linux"):
	#     # set latex
	#     plt.rc("text", usetex=True)

	# set the title
	# FigureTitle = "MFCC frame %d" % (FramNum)
	# Fig = plt.figure(FigureTitle)
	# plt.grid(linestyle="dotted")
	# # plt.plot(t, RhoSpice, "-", label="RhoSpice", linewidth=LineWidth)
	# plt.plot(ResultMfcc[FramNum], "-", label="MFCC", linewidth=LineWidth)
	# plt.plot(Test, "--", label="MFCC[1]_Mean", linewidth=LineWidth)
	# plt.xlabel("Index")
	# plt.ylabel("MFCCs")
	# plt.axis("tight")
	# plt.legend(loc="best")

	# plt.show()
