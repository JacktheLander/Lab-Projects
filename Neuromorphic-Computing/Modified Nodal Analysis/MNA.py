# ******************************************************************************
# import modules
# ******************************************************************************
import numpy as np
import torch
import pprint
import sys

# ******************************************************************************
# import from other module
# ******************************************************************************
from MNAHelpers import ReadArguments, ReadAndExtractNetlist

# ******************************************************************************
# MNASym class to build Matrix A, vect X, and vector Z
# ******************************************************************************
import MNASym
import MNAClass

# ******************************************************************************
def Signal(Type, Amp, Freq, Offset, NoCycles=1, NpCycle=1000, Verbose=None):
	# **************************************************************************
	# set the function name
	# **************************************************************************
	FunctionName = "Signal()"

	# **************************************************************************
	# display the message
	# **************************************************************************
	if Verbose:
		Msg = "...%-25s: input sine signal ..." % (FunctionName)
		print(Msg)

	# **************************************************************************
	# set the information of the signal
	# **************************************************************************
	Period = 1/Freq
	T   = NoCycles * Period
	Num = NoCycles * NpCycle

	# **************************************************************************
	# check the signal
	# **************************************************************************
	if Type == "Square":
		t   = np.linspace(0, T, num=Num)
		v   = torch.from_numpy(Amp * signal.square(2*np.pi*Freq*t) + Offset)
		t   = torch.from_numpy(t)
	else:
		t   = torch.linspace(0, T, steps=Num)
		v   = Amp * torch.sin(2*torch.pi*Freq*t)  + Offset
	return v, t

# ******************************************************************************
if __name__ == '__main__':
	# **************************************************************************
	# import modules
	# **************************************************************************
	from scipy import signal
	from os.path import join
	import os
	import argparse
	import pandas as pd

	# **************************************************************************
	# Device will determine whether to run the training on GPU or CPU.
	# **************************************************************************
	Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# **************************************************************************
	# process the arguments
	# **************************************************************************
	ap = argparse.ArgumentParser()

	# Manually set sys.argv to simulate command-line arguments
	sys.argv = ["MNA.py", "-f", "Assets/Figure2.cir"]

	ap.add_argument("-f", "--Netlist", type=str, required=True, help="Netlist file for circuit.")
	args = ap.parse_args()
	Verbose = False

	# **************************************************************************
	# create input signal
	# **************************************************************************
	# Type    = "Square"
	Type    = "Sine"
	Amp     = 2.0
	Freq    = 1
	Offset  = 0
	Cycles  = 2
	Vs, ts  = Signal(Type, Amp, Freq, Offset, NoCycles=Cycles, Verbose=Verbose)
	Vins    = torch.reshape(Vs, (len(Vs), 1))

	# **************************************************************************
	# extract the component list from the netlist
	# **************************************************************************
	dt = ts[1] - ts[0]
	TorchFlag	= True
	InParams	= ReadAndExtractNetlist(args.Netlist, dt=dt, TorchFlag=TorchFlag, Verbose=True)
	InParams["MemResObj"] = None
	InParams["MemCapObj"] = None

	# **************************************************************************
	# Create MNASym object and build Matrix A, Vector X, and Vector Z
	# **************************************************************************
	Verbose     = False
	MNASymObj  	= MNASym.MNASym(InParams=InParams, Verbose=Verbose)

	# **************************************************************************
	# Get Matrix A, Vector X, and Vector Z
	# **************************************************************************
	ValDict = MNASymObj.GetMatrixAndVectors()
	MatrixA = ValDict["MatrixA"]
	VectorX = ValDict["VectorX"]
	VectorZ = ValDict["VectorZ"]

	# **************************************************************************
	# display the contents of Matrix A, Vector X, and Vector Z
	# **************************************************************************
	print("\nMatrix A")
	pprint.pprint(MatrixA)
	print("\nVector X")
	pprint.pprint(VectorX)
	print("\nVector Z")
	pprint.pprint(VectorZ)

	# **************************************************************************
	# Create MNA object and build Matrix A, Vector X, and Vector Z
	# **************************************************************************
	MNAObj  = MNAClass.MNAClass(InParams=InParams, Verbose=Verbose)

	# **************************************************************************
	# calculate a bias point
	# **************************************************************************
	MNAObj.CalBiasPoint()

	# **************************************************************************
	# get the matrix A, vector X, and Vector Z
	# **************************************************************************
	ObjDict = MNAObj.GetMatrixAndVectors()
	MatrixA = ObjDict["MatrixA"]
	VectorZ = ObjDict["VectorZ"]
	VectorX = ObjDict["VectorX"]

	# **************************************************************************
	# display the contents of Matrix A, Vector X, and Vector Z
	# **************************************************************************
	print("\nMatrix A")
	print(pd.DataFrame(MatrixA))
	print("\nVector Z")
	print(pd.DataFrame(VectorZ))
	print("\nVector X")
	print(pd.DataFrame(VectorX))

	# **************************************************************************
	# calculate node voltages
	# **************************************************************************
	VNodes  = MNAObj.CalNodeVoltages(Vins, dt)

	# **************************************************************************
	# setting for results
	# **************************************************************************
	import matplotlib.pyplot as plt
	from matplotlib import rcParams
	rcParams.update({"figure.autolayout": True})

	# **************************************************************************
	# set the font family
	# **************************************************************************
	# FontSize = 12
	FontSize = 14
	# FontSize = 16
	# FontSize = 18
	# FontSize = 20
	# FontSize = 22
	# FontSize = 24
	# FontSize = 28
	font = {"family": "Times New Roman", "size": FontSize}
	plt.rc("font", **font)  # pass in the font dict as kwargs
	plt.rcParams["font.family"] = "Times New Roman"
	plt.rcParams["mathtext.fontset"] = "cm"

	# **************************************************************************
	# default setting
	# **************************************************************************
	LineWidth = 1.5
	# Scale     = 1e3

	# **************************************************************************
	# set the legend labels
	# **************************************************************************
	VsLbl   = r"$v_s(t)$"
	V1Lbl = r"$v_1(t)$"
	V2Lbl   = r"$v_2(t)$"
	V3Lbl   = r"$v_3(t)$"
	V4Lbl = r"$v_4(t)$"

	# **************************************************************************
	# get the figure handle
	# **************************************************************************
	Fig = plt.figure("Input Signal")
	# plt.title("Biolek Q-V Plot")
	plt.grid(linestyle="dotted")
	plt.plot(ts, Vins[:,0], label=VsLbl, linewidth=LineWidth) 	# Plot Vs
	# plt.plot(ts, VNodes[:,1], label=V2Lbl, linewidth=LineWidth)		# Plot Node 1
	# plt.plot(ts, VNodes[:,2], label=V2Lbl, linewidth=LineWidth)		# Plot Node 2
	plt.plot(ts, VNodes[:,3], label=V3Lbl, linewidth=LineWidth) 	# Plot Node 3
	plt.plot(ts, VNodes[:,4], label=V4Lbl, linewidth=LineWidth)	# Plot Node 4
	plt.xlabel("time (s)")
	plt.ylabel("Voltage (V)")
	plt.legend(loc="best")
	plt.axis("tight")

	# **************************************************************************
	# set the file name
	# **************************************************************************
	FileName = "MNA.png"
	FileNamePng = join("Outputs", FileName)
	# FileName    = FileNameJpg
	FileName    = FileNamePng

	# **************************************************************************
	# save the figure
	# **************************************************************************
	print("...Saving figure to file = <%s> ..." % FileName)

	# **************************************************************************
	# save the figure
	# **************************************************************************
	plt.savefig(FileName)

	# **************************************************************************
	plt.show()
