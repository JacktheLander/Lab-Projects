# ******************************************************************************
# import modules
# ******************************************************************************
import tonic
from tonic import DiskCachedDataset
import tonic.transforms as transforms
import torch
from torch.utils.data import DataLoader

import numpy as np
from sklearn import linear_model
import time

from lsm_weight_definitions import initWeights1
from lsm_models import LSM

from progressbar import progressbar
from os.path import join, isdir
import os, sys


if __name__ == "__main__":
	# **********************************************************************
	# set the function name
	# **********************************************************************
	FunctionName    = "__main__"

	# **************************************************************************
	# Load dataset (Using NMNIST here)
	# **************************************************************************
	DataFolder = os.path.abspath(os.path.join(os.path.dirname(__file__), r"C:\Users\Jaxba\PycharmProjects\Neuromorphic Computing\Datasets"))

	Msg = "...%-25s: loading Datasets from <%s> ..." % (FunctionName, DataFolder)
	print(Msg)

	SensorSize    = tonic.datasets.NMNIST.sensor_size
	FrameTransform = transforms.Compose([transforms.Denoise(filter_time=3000),
										  transforms.ToFrame(sensor_size=SensorSize,time_window=1000)])

	TrainSet    = tonic.datasets.NMNIST(save_to=DataFolder, transform=FrameTransform, train=True)
	TestSet     = tonic.datasets.NMNIST(save_to=DataFolder, transform=FrameTransform, train=False)

	# **************************************************************************
	# caching Datasets
	# **************************************************************************
	TrainCacheFolder  = join("./cache", "nmnist", "train")
	TestCacheFolder   = join("./cache", "nmnist", "test")
	Msg = "...%-25s: caching Datasets to <%s> and <%s> ..." % (FunctionName, TrainCacheFolder, TestCacheFolder)
	print(Msg)

	CachedTrainSet = DiskCachedDataset(TrainSet, cache_path=TrainCacheFolder)
	CachedTestSet  = DiskCachedDataset(TestSet, cache_path=TestCacheFolder)

	# **************************************************************************
	# loading Datasets from cache folder
	# **************************************************************************
	BatchSize  	= 256
	TrainLoader = DataLoader(CachedTrainSet, batch_size=BatchSize, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
	TestLoader  = DataLoader(CachedTestSet, batch_size=BatchSize, collate_fn=tonic.collation.PadTensors(batch_first=False))

	# **************************************************************************
	# Set device
	# **************************************************************************
	device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	Msg = "...%-25s: running with <%s> device" % (FunctionName, device)
	print(Msg)

	# **************************************************************************
	# loading Datasets and targets
	# **************************************************************************
	data, targets = next(iter(TrainLoader))
	FlatData = torch.reshape(data, (data.shape[0], data.shape[1], -1))
	# print('Datasets shape: ', Datasets.shape)
	# print('flat Datasets shape: ', FlatData.shape)
	in_sz = FlatData.shape[-1]

	# **************************************************************************
	# Set neuron parameters
	# **************************************************************************
	tauV = 16.0
	tauI = 16.0
	th = 20
	curr_prefac = np.float32(1/tauI)
	alpha = np.float32(np.exp(-1/tauI))
	beta = np.float32(1 - 1/tauV)

	Nz = 10
	#Win, Wlsm = initWeights1(27, 2, 0.15, in_sz, Nz=Nz)
	Win, Wlsm = initWeights1(27, 2, 0.15, in_sz, Nz=Nz)
	abs_W_lsm = np.abs(Wlsm)

	Msg = "...%-25s: average fan out = <%.4f> ..." % (FunctionName, np.mean(np.sum(abs_W_lsm>0, axis=1)))
	print(Msg)

	# **************************************************************************
	N = Wlsm.shape[0]
	LSMNet = LSM(N, in_sz, np.float32(curr_prefac*Win), np.float32(curr_prefac*Wlsm), alpha=alpha, beta=beta, th=th).to(device)
	MumPartitions = 3
	LSMNet.eval()

	# **************************************************************************
	# Run with no_grad for LSM
	# **************************************************************************
	Msg = "...%-25s: running train batches ..." % (FunctionName)
	print(Msg)
	with torch.no_grad():
		start_time = time.time()
		for i, (data, targets) in progressbar(enumerate(iter(TrainLoader))):
			if i%25 == 24:
				print("train batches completed: ", i)
			FlatData    = torch.reshape(data, (data.shape[0], data.shape[1], -1)).to(device)
			PartSteps   = FlatData.shape[0]//MumPartitions
			SpkRec      = LSMNet(FlatData)
			if i==0:
				LSM_Parts = []
				for part in range(MumPartitions):
					LSM_Parts.append(torch.mean(SpkRec[part*PartSteps:(part+1)*PartSteps], dim=0))

				LSM_Out      = torch.cat(LSM_Parts, dim=1)
				InTrain     = torch.mean(FlatData, dim=0).cpu().numpy()
				LSM_OutTrain = LSM_Out.cpu().numpy()
				LSM_LabelTrain = np.int32(targets.numpy())
			else:
				LSM_Parts = []
				for part in range(MumPartitions):
					LSM_Parts.append(torch.mean(SpkRec[part*PartSteps:(part+1)*PartSteps], dim=0))

				LSM_Out      = torch.cat(LSM_Parts, dim=1)
				InTrain     = np.concatenate((InTrain, torch.mean(FlatData, dim=0).cpu().numpy()), axis=0)
				LSM_OutTrain = np.concatenate((LSM_OutTrain, LSM_Out.cpu().numpy()), axis=0)
				LSM_LabelTrain = np.concatenate((LSM_LabelTrain, np.int32(targets.numpy())), axis=0)

		end_time = time.time()
		Msg = "...%-25s: running time of training epoch = <%.4f> ..." % (FunctionName, end_time - start_time)
		print(Msg)

		# **********************************************************************
		# testing outputs of reservoirs
		# **********************************************************************
		Msg = "...%-25s: running test batches ..." % (FunctionName)
		print(Msg)
		for i, (data, targets) in progressbar(enumerate(iter(TestLoader))):
			if i%25 == 24:
				print("test batches completed: ", i)
			FlatData = torch.reshape(data, (data.shape[0], data.shape[1], -1)).to(device)
			PartSteps = FlatData.shape[0]//MumPartitions
			SpkRec = LSMNet(FlatData)
			if i==0:
				LSM_Parts = []
				for part in range(MumPartitions):
					LSM_Parts.append(torch.mean(SpkRec[part*PartSteps:(part+1)*PartSteps], dim=0))

				LSM_Out = torch.cat(LSM_Parts, dim=1)
				InTest  = torch.mean(FlatData, dim=0).cpu().numpy()
				LSM_OutTest   = LSM_Out.cpu().numpy()
				LSM_LabelTest = np.int32(targets.numpy())
			else:
				LSM_Parts = []
				for part in range(MumPartitions):
					LSM_Parts.append(torch.mean(SpkRec[part*PartSteps:(part+1)*PartSteps], dim=0))

				LSM_Out     = torch.cat(LSM_Parts, dim=1)
				InTest      = np.concatenate((InTest, torch.mean(FlatData, dim=0).cpu().numpy()), axis=0)
				LSM_OutTest = np.concatenate((LSM_OutTest, LSM_Out.cpu().numpy()), axis=0)
				LSM_LabelTest = np.concatenate((LSM_LabelTest, np.int32(targets.numpy())), axis=0)

	# **************************************************************************
	# display message
	# **************************************************************************
	# print("LSM_OutTrain    = ", LSM_OutTrain.shape)
	# print("LSM_OutTest     = ", LSM_OutTest.shape)
	#
	# print("InTrain     = ", InTrain.shape)
	# print("InTest     = ", InTest.shape)
	Msg = "...%-25s: mean in spiking (train) = <%.8f> ..." % (FunctionName, np.mean(InTrain))
	print(Msg)

	Msg = "...%-25s: mean in spiking (test) = <%.8f> ..." % (FunctionName, np.mean(InTest))
	print(Msg)

	Msg = "...%-25s: mean LSM spiking (train) = <%.8f> ..." % (FunctionName, np.mean(LSM_OutTrain))
	print(Msg)

	Msg = "...%-25s: mean LSM spiking (test) = <%.8f> ..." % (FunctionName, np.mean(LSM_OutTest))
	print(Msg)

	# **************************************************************************
	# training output layer
	# **************************************************************************
	Msg = "...%-25s: training output layer ..." % (FunctionName)
	print(Msg)
	clf = linear_model.SGDClassifier(max_iter=10000, tol=1e-6)
	clf.fit(LSM_OutTrain, LSM_LabelTrain)

	# **************************************************************************
	# testing the reservoir
	# **************************************************************************
	score = clf.score(LSM_OutTest, LSM_LabelTest)
	Msg = "...%-25s: test score = <%.4f> ..." % (FunctionName, score)
	print(Msg)
