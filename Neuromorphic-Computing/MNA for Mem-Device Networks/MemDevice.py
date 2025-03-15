import sys

# ******************************************************************************
if __name__ == '__main__':
	# **************************************************************************
	# import modules
	# **************************************************************************
	import os
	import numpy as np
	import torch

	# Get absolute paths for Lab 4 and Lab 5
	lab4_path = os.path.abspath(os.path.join(os.path.dirname(__file__), r"C:\Users\Jaxba\PycharmProjects\Neuromorphic Computing\Lab 4 - Biolek R2 Memristor"))
	lab5_path = os.path.abspath(os.path.join(os.path.dirname(__file__), r"C:\Users\Jaxba\PycharmProjects\Neuromorphic Computing\Lab 5 - Modified Nodal Analysis"))
	cap_path = os.path.abspath(os.path.join(os.path.dirname(__file__), r"C:\Users\Jaxba\PycharmProjects\Neuromorphic Computing\Modeling a Memcapacitor"))

	# Add them to sys.path
	sys.path.append(lab4_path)
	sys.path.append(lab5_path)
	sys.path.append(cap_path)

	# **************************************************************************
	# import memristive models
	# **************************************************************************
	"""
	Import your memristive/memcapacitive model here:
		Format: <FileName>.<ClassName>

	"""

	from BiolekR2 import BiolekR2Memristor
	from BiolekC4 import BiolekC4Memcapacitor

	# **************************************************************************
	# MRMC device model
	# **************************************************************************
	"""
	Import your MNA class here:
		Format: <FileName>.<ClassName>
	"""

	from MNAClass import MNAClass


	# **************************************************************************
	# import module library
	# **************************************************************************
	import SmallWorldPowerLaw
	from MathUtils import FindFactors
	import NetList
	# import MNAClass

	# **************************************************************************
	# a temporary fix for OpenMP
	# **************************************************************************
	os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

	# **************************************************************************
	# check for GPU
	# **************************************************************************
	GPU_Flag    = torch.cuda.is_available()
	Device      = torch.device("cuda:0" if GPU_Flag else "cpu")

	# **************************************************************************
	# create mem-device objects for MNA circuit
	# **************************************************************************
	MemResObj = BiolekR2Memristor()  # Creates an instance, not just a reference
	MemCapObj = BiolekC4Memcapacitor()

	if GPU_Flag:
		if MemResObj is not None:
			MemResObj = MemResObj.to(Device)
		if MemCapObj is not None:
			MemCapObj = MemCapObj.to(Device)

	# **************************************************************************
	# create input sine wave signals for 20 inputs
	# **************************************************************************
	InNodes  = 20
	Ampl     = torch.distributions.uniform.Uniform(0.5, 2.0).sample((InNodes,1))
	Freq     = 1
	Phase    = torch.distributions.uniform.Uniform(0, torch.pi).sample((InNodes,1))
	NoCycles = 2
	EndTime  = NoCycles * 1/Freq
	NumPnts  = 1000 * NoCycles
	ts      = torch.linspace(0, EndTime, steps=NumPnts)
	dt      = ts[1] - ts[0]
	Vins    = torch.zeros((len(ts), InNodes), dtype=torch.double)

	# **************************************************************************
	# fill the inputs
	# **************************************************************************
	for i in range(InNodes):
		Vins[:,i]   = Ampl[i]*torch.sin(2*torch.pi*Freq*ts + Phase[i])
	Vins    = Vins.to(Device)

	# **************************************************************************
	# parameters for small-world power-law graph
	# **************************************************************************
	Nodes       = 200
	InitGraph   = "Grid_Graph"
	Alpha       = 1.390525834
	Beta        = 82.42180956
	Gamma       = 16.33022077
	Delta       = 0
	BoundaryFlag = False
	GndNodeFlag = True
	L           = FindFactors(Nodes)
	Verbose     = True

	# **************************************************************************
	# Create netlist using SWPL network
	# **************************************************************************
	NetClassObj = SmallWorldPowerLaw.SmallWorldPowerLaw(InitGraph=InitGraph, L=L, Beta=Beta, \
			Alpha=Alpha, Gamma=Gamma, Delta=Delta, BoundaryFlag=BoundaryFlag, \
			GndNodeFlag=GndNodeFlag, Verbose=Verbose)
	DevList = NetClassObj.GetEdgeList()
	del NetClassObj

	# **************************************************************************
	# create a netlist object
	# **************************************************************************
	PerMemC     = 0
	MRMCFlag    = False
	NetParams   = {"DevList": DevList, "InNodes": InNodes, "PerMemC": PerMemC, \
			"MRMCFlag": MRMCFlag, "Verbose": Verbose}

	NetListObj  = NetList.NetList(NetParams)
	NetParams   = NetListObj.GetNetParams()
	del NetListObj

	# **************************************************************************
	# adding extra values
	# **************************************************************************
	NetParams["dt"]          = dt
	NetParams["MemResObj"]   = MemResObj
	NetParams["MemCapObj"]   = MemCapObj


	# **************************************************************************
	# create a MNA class object
	# **************************************************************************
	print("NetParams keys:", NetParams.keys()) # Debug the MNAClass inputs
	NetParams["KeyVals"] = NetParams.keys()

	MNAObj = MNAClass(InParams=NetParams, Verbose=Verbose)
	if GPU_Flag:
		MNAObj = MNAObj.to(Device)

	# **************************************************************************
	# calculate node voltages
	# **************************************************************************
	VNodes  = MNAObj.CalNodeVoltages(Vins, dt)
	print("VNodes   = ", VNodes.shape)

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
	# VsLbl   = r"$v_s(t)$"
	V3Lbl   = r"$v_{3}(t)$"
	V30Lbl  = r"$v_{30}(t)$"
	V50Lbl  = r"$v_{50}(t)$"
	V100Lbl = r"$v_{100}(t)$"

	# **************************************************************************
	# get the figure handle
	# **************************************************************************
	Fig = plt.figure("Reservoir States")
	# plt.title("Biolek Q-V Plot")
	plt.grid(linestyle="dotted")
	plt.plot(ts, VNodes[:,3], label=V3Lbl, linewidth=LineWidth)
	plt.plot(ts, VNodes[:,30], label=V30Lbl, linewidth=LineWidth)
	plt.plot(ts, VNodes[:,50], label=V50Lbl, linewidth=LineWidth)
	plt.plot(ts, VNodes[:,100], label=V100Lbl, linewidth=LineWidth)
	plt.xlabel("time (s)")
	plt.ylabel("Voltage (V)")
	plt.legend(loc="best")
	plt.axis("tight")

	# **************************************************************************
	# set the file name
	# **************************************************************************
	#FileNameEps = join("Figures", "MNAStates.eps")
	#FileNameJpg = join("Figures", "MNAStates.jpg")
	#FileNamePng = join("Figures", "MNAStates.png")
	# FileName    = FileNameJpg
	FileName    = "FileNamePng"

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
