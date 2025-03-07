# ******************************************************************************
# import modules
# ******************************************************************************
import numpy as np
import torch
import torch.nn as nn
from progressbar import progressbar

# ******************************************************************************
# import from other module
# ******************************************************************************
from MNAHelpers import ReadArguments, ReadAndExtractNetlist

# ******************************************************************************
"""
# ******************************************************************************
# MNA class to build Matrix A, vect X, and vector Z
# ******************************************************************************
"""
class MNAClass(nn.Module):
	"""
	https://lpsa.swarthmore.edu/Systems/Electrical/mna/MNA1.html
	1. Creating a netlist of memcapacitor based on the random graph
	2. Creating matrixA and Z for MNA
	3. Calculating voltages at nodes for outputs.
	"""
	def __init__(self, InParams, GPUFlag=False, GPUDevice=None, Verbose=False):
		# **********************************************************************
		super(MNAClass, self).__init__()

		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNA::__init__()"

		# **********************************************************************
		# Save the network parameters
		# **********************************************************************
		self.InParams	= InParams
		self.KeyVals	= InParams["KeyVals"]
		self.dt      	= InParams["dt"]
		self.NodeList	= InParams["NodeList"]
		self.MemResObj  = InParams["MemResObj"]
		self.MemCapObj  = InParams["MemCapObj"]
		self.GPUFlag    = GPUFlag
		self.GPUDevice  = GPUDevice
		self.Verbose    = Verbose

		# **********************************************************************
		# initial values
		# **********************************************************************
		self.InitMRStates = 0.5
		self.InitMCStates = 0.5

		# **********************************************************************
		# reset variables
		# **********************************************************************
		self.SaveMatrixA = None
		self.SaveVectorZ = None

		# **********************************************************************
		# reset flags
		# **********************************************************************
		self.VsFlag = False
		self.IsFlag = False
		self.RFlag	= False
		self.CFlag  = False
		self.LFlag	= False
		self.MRFlag = False
		self.MCFlag = False

		# **********************************************************************
		# reset all variables
		# **********************************************************************
		self.NumVs  = 0
		self.NumIs  = 0
		self.NumR   = 0
		self.NumC   = 0
		self.NumL	= 0
		self.NumMR  = 0
		self.NumMC  = 0

		# **********************************************************************
		# build all lists
		# **********************************************************************
		self._BuildAllLists()

		# **********************************************************************
		# set number of voltage sources
		# **********************************************************************
		if self.VsFlag:
			self.NumVs	= len(self.VsList)

		# **********************************************************************
		# reset all variables
		# **********************************************************************
		self.NumNodes =  len(self.NodeList) - 1
		self.Size    = self.NumNodes + self.NumVs
		self.MatrixA = torch.zeros((self.Size, self.Size), dtype=torch.double)
		self.VectorX = None
		self.VectorZ = torch.zeros(self.Size, dtype=torch.double)

		# **********************************************************************
		# set matrix A from RList
		# **********************************************************************
		if self.RFlag:
			self.NumR   = len(self.RList)
			self._SetMatrixAfromRList(self.RList)

		# **********************************************************************
		# set matrix A from VList
		# **********************************************************************
		if self.VsFlag:
			self._SetMatrixAVectorZfromVsList(self.VsList)

		# **********************************************************************
		# set Vector Z from IList
		# **********************************************************************
		if self.IsFlag:
			self.NumIs  = len(self.IsList)
			self._SetVectorZfromIsList(self.IsList)

			# ******************************************************************
			# save vector Z
			# ******************************************************************
			if self.SaveVectorZ is None:
				self.SaveVectorZ = torch.clone(self.VectorZ[:self.NumNodes])

		# **********************************************************************
		# set matrix A from CList
		# **********************************************************************
		if self.CFlag:
			self.NumC    = len(self.CList)
			self._SetCapMatrix(self.CList)
			self._SetMatrixAfromCList()
			self._SetVectorZfromCList()

		# **********************************************************************
		# save the matrix A
		# **********************************************************************
		if self.SaveMatrixA is None:
			self.SaveMatrixA = torch.clone(self.MatrixA)

		# **********************************************************************
		# set matrix A from MRList
		# **********************************************************************
		if self.MRFlag:
			self.NumMR  = len(self.MRList)
			self._SetMRMatrix(self.MRList)
			self._SetMatrixAFromMRList()

		# **********************************************************************
		# set matrix A from MCList
		# **********************************************************************
		if self.MCFlag:
			self.NumMC  = len(self.MCList)
			self._SetMCMatrix(self.MCList)
			self._SetMatrixAfromMCList()
			self._SetVectorZfromMCList()

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			# ******************************************************************
			# display the information
			# ******************************************************************
			Msg = "\n==> MNA Class ..."
			print(Msg)

			# ******************************************************************
			# display the information
			# ******************************************************************
			Msg = "...%-25s: Vs = %d, Is = %d, R = %d, MR = %d, C = %d, MC = %d" % \
					(FunctionName, self.NumVs, self.NumIs, self.NumR, self.NumMR, \
							self.NumC, self.NumMC)
			print(Msg)

			# ******************************************************************
			# display the information
			# ******************************************************************
			Msg = "...%-25s: matrix A = <%s>, Vector Z = <%s>" % (FunctionName, \
					self.MatrixA.shape, self.VectorZ.shape)
			print(Msg)

	# **************************************************************************
	def _BuildAllLists(self):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNA::_BuildAllList()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			# ******************************************************************
			# display the information
			# ******************************************************************
			Msg = "...%-25s: building all lists..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# build each list
		# **********************************************************************
		for AKey in self.KeyVals:
			# ******************************************************************
			# check the key for a particular list
			# ******************************************************************
			ListType	= self.InParams[AKey]
			if ListType is None:
				NotEmptyFlag = False
			else:
				List	= ListType["List"]
				NotEmptyFlag = True

			if AKey == "VsList":
				self.VsFlag = NotEmptyFlag
				if self.VsFlag:
					self.VsList		= List
				else:
					self.VsList		= None

			elif AKey == "IsList":
				self.IsFlag = NotEmptyFlag
				if self.IsFlag:
					self.IsList		= List
				else:
					self.IsList		= None

			elif AKey == "RList":
				self.RFlag = NotEmptyFlag
				if self.RFlag:
					self.RList		= List
				else:
					self.RList		= None

			elif AKey == "CList":
				self.CFlag = NotEmptyFlag
				if self.CFlag:
					self.CList		= List
				else:
					self.CList		= None

			elif AKey == "LList":
				self.LFlag = NotEmptyFlag
				if self.LFlag:
					self.LList		= List
				else:
					self.LList		= None

			elif AKey == "MRList":
				self.MRFlag = NotEmptyFlag
				if self.MRFlag:
					self.MRList		= List
				else:
					self.MRList		= None

			elif AKey == "MCList":
				self.MCFlag = NotEmptyFlag
				if self.MCFlag:
					self.MCList		= List
				else:
					self.MCList		= None
			else:
				# **************************************************************
				# format error message
				# **************************************************************
				Msg = "%s: unknown device => <%s>" % (FunctionName, AKey)
				raise ValueError(Msg)

	# **************************************************************************
	"""
	****************************************************************************
	   Functions setup Matrix A and Vector Z
	****************************************************************************
	"""
	# **************************************************************************
	def _SetMatrixAfromRList(self, RList):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNA::_SetMatrixAfromRList()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			Msg = "...%-25s: set Matrix A from RList..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# set the matrix A for each node
		# **********************************************************************
		for i in range(self.NumNodes):
			Node    = i + 1
			# ******************************************************************
			# list of nodes on N1 and N2
			# ******************************************************************
			N1_Ind  = RList[:,0] == Node
			N2_Ind  = RList[:,1] == Node

			# ******************************************************************
			# set the flag for valid entries
			# ******************************************************************
			N1Flag  = torch.any(N1_Ind)
			N2Flag  = torch.any(N2_Ind)

			# ******************************************************************
			# resistors at N1 and N2
			# ******************************************************************
			if N1Flag:
				N1_InvRVals = 1/RList[N1_Ind,2]
				N1Total = torch.sum(N1_InvRVals)
			else:
				N1Total = 0.0

			if N2Flag:
				N2_InVRVals = 1/RList[N2_Ind,2]
				N2Total = torch.sum(N2_InVRVals)
			else:
				N2Total = 0.0

			# ******************************************************************
			# set the element of Matrix A
			# ******************************************************************
			self.MatrixA[i,i]  += (N1Total + N2Total)

			# ******************************************************************
			# adjacent nodes of N1
			# ******************************************************************
			if N1Flag:
				# **************************************************************
				# update adjacent nodes of i
				# **************************************************************
				N1Entries   = RList[N1_Ind]
				N2NonZero   = N1Entries[:,1] != 0
				if torch.any(N2NonZero):
					N2Nodes     = N1Entries[N2NonZero,1].to(int) - 1
					N2Vals      = 1/N1Entries[N2NonZero,2]
					self.MatrixA[i,N2Nodes] -= N2Vals
					self.MatrixA[N2Nodes,i] -= N2Vals

	# **************************************************************************
	def _SetMatrixAVectorZfromVsList(self, VList):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNA::_SetMatrixAVectorZfromVsList()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			Msg = "...%-25s: set Matrix A from VList..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# set the matrix A for each node
		# **********************************************************************
		for i in range(self.NumNodes):
			Node    = i + 1
			# ******************************************************************
			# list of nodes on N1 and N2
			# ******************************************************************
			N1_Ind  = VList[:,0] == Node
			N2_Ind  = VList[:,1] == Node

			# ******************************************************************
			# set the flag for valid entries
			# ******************************************************************
			N1Flag  = torch.any(N1_Ind)
			N2Flag  = torch.any(N2_Ind)

			# ******************************************************************
			# check the flag and set Matrix A
			# ******************************************************************
			RowA    = self.MatrixA[i,self.NumNodes:]
			ColA    = self.MatrixA[self.NumNodes:,i]
			if N1Flag:
				RowA[N1_Ind]    = 1
				ColA[N1_Ind]    = 1
			if N2Flag:
				RowA[N2_Ind]    = -1
				ColA[N2_Ind]    = -1

		# **********************************************************************
		# set the vector Z
		# **********************************************************************
		self.VectorZ[self.NumNodes:] = VList[:,2]

	# **************************************************************************
	def _SetVectorZfromIsList(self, IList):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNA::_SetVectorZfromIsList()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			Msg = "...%-25s: set Vector Z from IList..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# set the vector Z for each node
		# **********************************************************************
		for i in range(self.NumNodes):
			Node    = i + 1
			# ******************************************************************
			# list of nodes on N1 and N2
			# ******************************************************************
			N1_Ind  = IList[:,0] == Node
			N2_Ind  = IList[:,1] == Node

			# ******************************************************************
			# set the flag for valid entries
			# ******************************************************************
			N1Flag  = torch.any(N1_Ind)
			N2Flag  = torch.any(N2_Ind)

			# ******************************************************************
			# check the flag and set vector Z
			# ******************************************************************
			if N1Flag:
				N1Vals  = IList[N1_Ind,2]
				self.VectorZ[i] -= torch.sum(N1Vals)
			if N2Flag:
				N2Vals  = IList[N2_Ind,2]
				self.VectorZ[i] += torch.sum(N2Vals)

	# **************************************************************************
	def _SetCapMatrix(self, CList):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNA::_SetCapMatrix()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			# display the information
			Msg = "...%-25s: Set Cap Matrix from C List..." % (FunctionName)
			print(Msg)

		"""
		# **********************************************************************
		# setup capacitor matrix for calculating MNA
		# setting capacitor matrix for updating Matrix A
		# **********************************************************************
		# parameters for capacitive matrix
		# **********************************************************************
		#      0     ,    1      , 2,   3
		# **********************************************************************
		#   plus node, minus node, C, PreV
		# **********************************************************************
		"""
		Fields = 4
		self.CHder      = ["N1", "N2", "C", "PreV"]
		self.CapMatrix  = torch.zeros((self.NumC, Fields), dtype=torch.double)

		# **********************************************************************
		# save the values to Cap matrix
		# **********************************************************************
		self.CapMatrix[:,:3] = CList
		self.CapNodes   = self.CapMatrix[:,:2].to(torch.int)

	# **************************************************************************
	def _SetMatrixAfromCList(self):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNA::_SetMatrixAfromCList()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			Msg = "...%-25s: set Matrix A from CList..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# set the matrix A for each node
		# **********************************************************************
		for i in range(self.NumNodes):
			Node    = i + 1
			# ******************************************************************
			# list of nodes on N1 and N2
			# ******************************************************************
			N1_Ind  = self.CapMatrix[:,0] == Node
			N2_Ind  = self.CapMatrix[:,1] == Node

			# ******************************************************************
			# set the flag for valid entrie
			# ******************************************************************
			N1Flag  = torch.any(N1_Ind)
			N2Flag  = torch.any(N2_Ind)

			# ******************************************************************
			# resistors at N1 and N2
			# ******************************************************************
			if N1Flag:
				# C/dt
				N1_InvRVals = self.CapMatrix[N1_Ind,2]/self.dt
				N1Total = torch.sum(N1_InvRVals)
			else:
				N1Total = 0.0

			if N2Flag:
				N2_InVRVals = self.CapMatrix[N2_Ind,2]/self.dt
				N2Total = torch.sum(N2_InVRVals)
			else:
				N2Total = 0.0

			# ******************************************************************
			# set the element of Matrix A
			# ******************************************************************
			self.MatrixA[i,i]  += N1Total + N2Total

			# ******************************************************************
			# adjacent nodes of N1
			# ******************************************************************
			if N1Flag:
				# **************************************************************
				# update adjacent nodes of i
				# **************************************************************
				N1Entries   = self.CapMatrix[N1_Ind]
				N2NonZero   = N1Entries[:,1] != 0
				N2Nodes     = N1Entries[N2NonZero,1].to(int) - 1
				N2Vals      = N1Entries[N2NonZero,2]/self.dt
				self.MatrixA[i,N2Nodes] -= N2Vals
				self.MatrixA[N2Nodes,i] -= N2Vals

	# **************************************************************************
	def _UpdateVectorZfromC(self, i, N1Flag, N1_Ind, N2Flag, N2_Ind):
		# **********************************************************************
		# check the flag and set vector Z
		# **********************************************************************
		if N1Flag:
			N1Vals  = torch.mul(self.CapMatrix[N1_Ind,2]/self.dt, self.CapMatrix[N1_Ind,3])
			self.VectorZ[i] += torch.sum(N1Vals)
		if N2Flag:
			N2Vals  = torch.mul(self.CapMatrix[N2_Ind,2]/self.dt, self.CapMatrix[N1_Ind,3])
			self.VectorZ[i] -= torch.sum(N2Vals)

	# **************************************************************************
	def _SetVectorZfromCList(self):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNA::_SetVectorZfromCList()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			Msg = "...%-25s: set Vector Z from CList..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# create a cap information dictionary to keep information
		# **********************************************************************
		N1IndMatrix = torch.zeros((self.NumNodes, self.NumC), dtype=torch.bool)
		N2IndMatrix = torch.zeros((self.NumNodes, self.NumC), dtype=torch.bool)
		N1FlagVec   = torch.zeros(self.NumNodes, dtype=torch.bool)
		N2FlagVec   = torch.zeros(self.NumNodes, dtype=torch.bool)
		self.CapInfo = {"NumC": self.NumC, "N1FlagVec": N1FlagVec, "N1IndMatrix": N1IndMatrix,\
				"N2FlagVec": N2FlagVec, "N2IndMatrix": N2IndMatrix}

		# **********************************************************************
		# set the vector Z for each node
		# **********************************************************************
		for i in range(self.NumNodes):
			Node    = i + 1
			# ******************************************************************
			# list of nodes on N1 and N2
			# ******************************************************************
			N1_Ind  = self.CapMatrix[:,0] == Node
			N2_Ind  = self.CapMatrix[:,1] == Node

			# ******************************************************************
			# set the flag for valid entries
			# ******************************************************************
			N1Flag  = torch.any(N1_Ind)
			N2Flag  = torch.any(N2_Ind)

			# ******************************************************************
			# save the information
			# ******************************************************************
			N1IndMatrix[i]  = N1_Ind
			N2IndMatrix[i]  = N2_Ind
			N1FlagVec[i]    = N1Flag
			N2FlagVec[i]    = N2Flag

			# ******************************************************************
			# check the flag and set vector Z
			# ******************************************************************
			self._UpdateVectorZfromC(i, N1Flag, N1_Ind, N2Flag, N2_Ind)

	# **************************************************************************
	def _SetMRMatrix(self, MRList):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNA::_SetMRMatrix()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			Msg = "...%-25s: set MR matrix from MRList ..." % (FunctionName)
			print(Msg)

		"""
		# **********************************************************************
		# setup memristor matrix for calculating MNA
		# setting memristor matrix for updating Matrix A
		# **********************************************************************
		# **********************************************************************
		# parameters for memcapacitive matrix
		# **********************************************************************
		#    0,  1, 2, 3
		# **********************************************************************
		#   N1, N2, R, CurrV
		# **********************************************************************
		"""
		Fields = 4
		self.MRHder       = ["N1", "N2", "R", "CurV"]
		self.MemResMatrix = torch.zeros((self.NumMR, Fields), dtype=torch.double)
		self.MemResMatrix[:,:3] = MRList

		# **********************************************************************
		# reset initial state variables
		# **********************************************************************
		InitStates = torch.zeros(self.NumMR, dtype=torch.double)
		InitStates[:] = self.InitMRStates
		self.MemResObj.ResetInitVals(InitStates)

		# **********************************************************************
		# get the initial resistance
		# **********************************************************************
		# NewMRList   = torch.zeros(self.NumMR, dtype=torch.double)
		self.MemResMatrix[:,2] = self.MemResObj.GetInitVals(InitStates)
		self.MemResNodes = self.MemResMatrix[:,:2].to(torch.int)

	# **************************************************************************
	def _SetDiagAdjElmMatrixAFromMR(self, i, N1Flag, N1_Ind, N2Flag, N2_Ind):
		# ******************************************************************
		# resistors at N1 and N2
		# ******************************************************************
		if N1Flag:
			N1_InvRVals = 1/self.MemResMatrix[N1_Ind,2]
			N1Total = torch.sum(N1_InvRVals)
		else:
			N1Total = 0.0

		if N2Flag:
			N2_InVRVals = 1/self.MemResMatrix[N2_Ind,2]
			N2Total = torch.sum(N2_InVRVals)
		else:
			N2Total = 0.0

		# ******************************************************************
		# set the element of Matrix A
		# ******************************************************************
		self.MatrixA[i,i]  += N1Total + N2Total

		# ******************************************************************
		# adjacent nodes of N1
		# ******************************************************************
		if N1Flag:
			# **************************************************************
			# update adjacent nodes of i
			# **************************************************************
			N1Entries   = self.MemResMatrix[N1_Ind]
			N2NonZero   = N1Entries[:,1] != 0
			N2Nodes     = N1Entries[N2NonZero,1].to(int) - 1
			N2Vals      = 1/N1Entries[N2NonZero,2]
			self.MatrixA[i,N2Nodes] -= N2Vals
			self.MatrixA[N2Nodes,i] -= N2Vals

	# **************************************************************************
	def _SetMatrixAFromMRList(self):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNA::_SetMatrixAFromMRList()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			Msg = "...%-25s: set Matrix A from MRList..." % \
					(FunctionName)
			print(Msg)

		# **********************************************************************
		# create a cap information dictionary to keep information
		# **********************************************************************
		N1IndMatrix = torch.zeros((self.NumNodes, self.NumMR), dtype=torch.bool)
		N2IndMatrix = torch.zeros((self.NumNodes, self.NumMR), dtype=torch.bool)
		N1FlagVec   = torch.zeros(self.NumNodes, dtype=torch.bool)
		N2FlagVec   = torch.zeros(self.NumNodes, dtype=torch.bool)
		self.MRInfo = {"NumMR": self.NumMR, "N1FlagVec": N1FlagVec, "N1IndMatrix": N1IndMatrix,\
				"N2FlagVec": N2FlagVec, "N2IndMatrix": N2IndMatrix}

		# **********************************************************************
		# set the matrix A for each node
		# **********************************************************************
		for i in range(self.NumNodes):
			Node    = i + 1
			# ******************************************************************
			# list of nodes on N1 and N2
			# ******************************************************************
			N1_Ind  = self.MemResMatrix[:,0] == Node
			N2_Ind  = self.MemResMatrix[:,1] == Node

			# ******************************************************************
			# set the flag for valid entries
			# ******************************************************************
			N1Flag  = torch.any(N1_Ind)
			N2Flag  = torch.any(N2_Ind)

			# ******************************************************************
			# save the information
			# ******************************************************************
			N1FlagVec[i]    = N1Flag
			N1IndMatrix[i]  = N1_Ind
			N2FlagVec[i]    = N2Flag
			N2IndMatrix[i]  = N2_Ind

			# ******************************************************************
			# set the diagonal and adjacent entries of Matrix A
			# ******************************************************************
			self._SetDiagAdjElmMatrixAFromMR(i, N1FlagVec[i], N1IndMatrix[i], N2FlagVec[i], \
					N2IndMatrix[i])

	# **************************************************************************
	def _SetMCMatrix(self, MCList):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNA::_SetMCMatrix()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			# display the information
			Msg = "...%-25s: Set MC Matrix from MC List..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# setup capacitor matrix for calculating MNA
		# setting capacitor matrix for updating Matrix A
		# **********************************************************************
		# **********************************************************************
		# parameters for memcapacitive matrix
		# **********************************************************************
		#   0 , 1 , 2,  3,    4
		# **********************************************************************
		#   N1, N2, C, dC, PreV
		# **********************************************************************

		Fields = 5
		self.MCHder       = ["N1", "N2", "C", "dC", "PreV"]
		self.MemCapMatrix = torch.zeros((self.NumMC, Fields), dtype=torch.double)
		self.MemCapMatrix[:,:3] = MCList

		# **********************************************************************
		# reset initial state variables
		# **********************************************************************
		InitStates = torch.zeros(self.NumMC, dtype=torch.double)
		InitStates[:] = self.InitMCStates
		self.MemCapObj.ResetInitVals(InitStates)

		# **********************************************************************
		# get the initial capacitance
		# **********************************************************************
		self.MemCapMatrix[:,2] = self.MemCapObj.GetInitVals(InitStates)
		self.MemCapNodes = self.MemCapMatrix[:,:2].to(torch.int)

	# **************************************************************************
	def _SetDiagAdjElmMatrixAFromMC(self, i, N1Flag, N1_Ind, N2Flag, N2_Ind):
		# **********************************************************************
		# memcapacitors at N1 and N2
		# **********************************************************************
		if N1Flag:
			# (C + dC)/dt
			N1_InvRVals = torch.add(self.MemCapMatrix[N1_Ind,2], self.MemCapMatrix[N1_Ind,3])/self.dt
			N1Total = torch.sum(N1_InvRVals)
		else:
			N1Total = 0.0

		if N2Flag:
			# (C + dC)/dt
			N2_InVRVals = torch.add(self.MemCapMatrix[N2_Ind,2], self.MemCapMatrix[N2_Ind,3])/self.dt
			N2Total = torch.sum(N2_InVRVals)
		else:
			N2Total = 0.0

		# **********************************************************************
		# set the element of Matrix A
		# **********************************************************************
		self.MatrixA[i,i]  += N1Total + N2Total

		# **********************************************************************
		# adjacent nodes of N1
		# **********************************************************************
		if N1Flag:
			# ******************************************************************
			# update adjacent nodes of i
			# ******************************************************************
			N1Entries   = self.MemCapMatrix[N1_Ind]
			N2NonZero   = N1Entries[:,1] != 0
			N2Nodes     = N1Entries[N2NonZero,1].to(int) - 1

			# (C + dC)/dt
			N2Vals      = torch.add(N1Entries[N2NonZero,2], N1Entries[N2NonZero,3])/self.dt
			self.MatrixA[i,N2Nodes] -= N2Vals
			self.MatrixA[N2Nodes,i] -= N2Vals

	# **************************************************************************
	def _SetMatrixAfromMCList(self):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNA::_SetMatrixAfromMCList()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			Msg = "...%-25s: set Matrix A from MCList..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# create a cap information dictionary to keep information
		# **********************************************************************
		N1IndMatrix = torch.zeros((self.NumNodes, self.NumMC), dtype=torch.bool)
		N2IndMatrix = torch.zeros((self.NumNodes, self.NumMC), dtype=torch.bool)
		N1FlagVec   = torch.zeros(self.NumNodes, dtype=torch.bool)
		N2FlagVec   = torch.zeros(self.NumNodes, dtype=torch.bool)
		self.MCMatrixAInfo = {"NumMR": self.NumMC, "N1FlagVec": N1FlagVec, "N1IndMatrix": N1IndMatrix,\
				"N2FlagVec": N2FlagVec, "N2IndMatrix": N2IndMatrix}

		# **********************************************************************
		# set the matrix A for each node
		# **********************************************************************
		for i in range(self.NumNodes):
			Node    = i + 1
			# ******************************************************************
			# list of nodes on N1 and N2
			# ******************************************************************
			N1_Ind  = self.MemCapMatrix[:,0] == Node
			N2_Ind  = self.MemCapMatrix[:,1] == Node

			# ******************************************************************
			# set the flag for valid entrie
			# ******************************************************************
			N1Flag  = torch.any(N1_Ind)
			N2Flag  = torch.any(N2_Ind)

			# ******************************************************************
			# save the information
			# ******************************************************************
			N1FlagVec[i]    = N1Flag
			N1IndMatrix[i]  = N1_Ind
			N2FlagVec[i]    = N2Flag
			N2IndMatrix[i]  = N2_Ind

			# ******************************************************************
			# set the diagonal and adjacent entries of Matrix A
			# ******************************************************************
			self._SetDiagAdjElmMatrixAFromMC(i, N1FlagVec[i], N1IndMatrix[i], \
					N2FlagVec[i], N2IndMatrix[i])

	# **************************************************************************
	def _UpdateVectorZfromMC(self, i, N1Flag, N1_Ind, N2Flag, N2_Ind):
		# **********************************************************************
		# check the flag and set vector Z
		# **********************************************************************
		if N1Flag:
			N1Vals  = torch.mul(self.MemCapMatrix[N1_Ind,2]/self.dt, self.MemCapMatrix[N1_Ind,4])
			self.VectorZ[i] += torch.sum(N1Vals)
		if N2Flag:
			N2Vals  = torch.mul(self.MemCapMatrix[N2_Ind,2]/self.dt, self.MemCapMatrix[N1_Ind,4])
			self.VectorZ[i] -= torch.sum(N2Vals)

	# **************************************************************************
	def _SetVectorZfromMCList(self):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNA::_SetVectorZfromMCList()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			Msg = "...%-25s: set Vector Z from MCList..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# save the information for updating Vector Z
		# **********************************************************************
		N1IndMatrix = torch.zeros((self.NumNodes, self.NumMC), dtype=torch.bool)
		N2IndMatrix = torch.zeros((self.NumNodes, self.NumMC), dtype=torch.bool)
		N1FlagVec   = torch.zeros(self.NumNodes, dtype=torch.bool)
		N2FlagVec   = torch.zeros(self.NumNodes, dtype=torch.bool)
		self.MCVectorZInfo = {"NumMR": self.NumMC, "N1FlagVec": N1FlagVec, "N1IndMatrix": N1IndMatrix,\
				"N2FlagVec": N2FlagVec, "N2IndMatrix": N2IndMatrix}

		# **********************************************************************
		# set the vector Z for each node
		# **********************************************************************
		for i in range(self.NumNodes):
			Node    = i + 1
			# ******************************************************************
			# list of nodes on N1 and N2
			# ******************************************************************
			N1_Ind  = self.MemCapMatrix[:,0] == Node
			N2_Ind  = self.MemCapMatrix[:,1] == Node

			# ******************************************************************
			# set the flag for valid entries
			# ******************************************************************
			N1Flag  = torch.any(N1_Ind)
			N2Flag  = torch.any(N2_Ind)

			# ******************************************************************
			# save the information
			# ******************************************************************
			N1FlagVec[i]    = N1Flag
			N1IndMatrix[i]  = N1_Ind
			N2FlagVec[i]    = N2Flag
			N2IndMatrix[i]  = N2_Ind

			# ******************************************************************
			# update Vector Z
			# ******************************************************************
			self._UpdateVectorZfromMC(i, N1FlagVec[i], N1IndMatrix[i], N2FlagVec[i], \
					N2IndMatrix[i])

	# **************************************************************************
	def GetMatrixAndVectors(self):
		return {"MatrixA": self.MatrixA, "VectorX": self.VectorX, "VectorZ": self.VectorZ}

	# **************************************************************************
	"""
	****************************************************************************
	   Functions update Matrix A and Vector Z after calculating X = A^-1*X each
	   timestep
	****************************************************************************
	"""
	# **************************************************************************
	def _UpdateVectorZ_C_AfterCal(self):
		# **********************************************************************
		# extract information a cap information dictionary to keep information
		# **********************************************************************
		N1FlagVec    = self.CapInfo["N1FlagVec"]
		N1IndMatrix  = self.CapInfo["N1IndMatrix"]
		N2FlagVec    = self.CapInfo["N2FlagVec"]
		N2IndMatrix  = self.CapInfo["N2IndMatrix"]

		# **********************************************************************
		# set the vector Z for each node
		# **********************************************************************
		for i in range(self.NumNodes):
			# ******************************************************************
			# check the flag and set vector Z
			# ******************************************************************
			self._UpdateVectorZfromC(i, N1FlagVec[i], N1IndMatrix[i], N2FlagVec[i], \
					N2IndMatrix[i])

	# **************************************************************************
	def _UpdateCapMatrix_VectorZ_AfterCal(self, VectorX):
		"""
		# **********************************************************************
		# setup capacitor matrix for calculating MNA
		# setting capacitor matrix for updating Matrix A
		# **********************************************************************
		# parameters for capacitive matrix
		# **********************************************************************
		#      0     ,    1      , 2,  3
		# **********************************************************************
		#   plus node, minus node, C, PreV
		# **********************************************************************
		"""
		# **********************************************************************
		# update Capacitance Matrix
		# **********************************************************************
		N1Ind   = self.CapNodes[:,0]
		N2Ind   = self.CapNodes[:,1]
		VN1     = VectorX[N1Ind]
		VN2     = VectorX[N2Ind]
		self.CapMatrix[:,3] = torch.subtract(VN1, VN2)

		# **********************************************************************
		# update Vector Z
		# **********************************************************************
		self._UpdateVectorZ_C_AfterCal()

	# **************************************************************************
	def _UpdateMatrixA_MR_AfterCal(self):
		# **********************************************************************
		# create a cap information dictionary to keep information
		# **********************************************************************
		N1FlagVec   = self.MRInfo["N1FlagVec"]
		N1IndMatrix = self.MRInfo["N1IndMatrix"]
		N2FlagVec   = self.MRInfo["N2FlagVec"]
		N2IndMatrix = self.MRInfo["N2IndMatrix"]

		# **********************************************************************
		# set the matrix A for each node
		# **********************************************************************
		for i in range(self.NumNodes):
			# ******************************************************************
			# set the diagonal and adjacent entries of Matrix A
			# ******************************************************************
			self._SetDiagAdjElmMatrixAFromMR(i, N1FlagVec[i], N1IndMatrix[i], \
					N2FlagVec[i], N2IndMatrix[i])

	# **************************************************************************
	def _UpdateMatrixMR_MatrixA_AfterCal(self, VectorX):
		"""
		# **********************************************************************
		# setup memristor matrix for calculating MNA
		# setting memristor matrix for updating Matrix A
		# **********************************************************************
		# **********************************************************************
		# parameters for memcapacitive matrix
		# **********************************************************************
		#      0     ,    1      , 2,    3     , 4
		# **********************************************************************
		#   plus node, minus node, R, PrevVcaps, In
		# **********************************************************************
		"""
		# **********************************************************************
		# update PreV
		# **********************************************************************
		N1Ind   = self.MemResNodes[:,0]
		N2Ind   = self.MemResNodes[:,1]
		VN1     = VectorX[N1Ind]
		VN2     = VectorX[N2Ind]
		VMemRes = torch.subtract(VN1, VN2)
		self.MemResMatrix[:,3] = VMemRes

		# **********************************************************************
		# update memristance of MemResMatrix
		# **********************************************************************
		# print(pd.DataFrame(self.MemResMatrix))
		# print("...update MemResMatrix")
		self.MemResMatrix[:,2] = self.MemResObj.GetVals(VMemRes, self.dt)
		# print(pd.DataFrame(self.MemResMatrix))
		# exit()

		# print(pd.DataFrame(self.MatrixA))
		# print("...update MemResMatrix")
		self._UpdateMatrixA_MR_AfterCal()
		# print(pd.DataFrame(self.MatrixA))
		# exit()

	# **************************************************************************
	def _UpdateMatrixA_MC_AfterCal(self):
		# **********************************************************************
		# create a cap information dictionary to keep information
		# **********************************************************************
		N1FlagVec   = self.MCMatrixAInfo["N1FlagVec"]
		N1IndMatrix = self.MCMatrixAInfo["N1IndMatrix"]
		N2FlagVec   = self.MCMatrixAInfo["N2FlagVec"]
		N2IndMatrix = self.MCMatrixAInfo["N2IndMatrix"]

		# **********************************************************************
		# set the matrix A for each node
		# **********************************************************************
		for i in range(self.NumNodes):
			# ******************************************************************
			# set the diagonal and adjacent entries of Matrix A
			# ******************************************************************
			self._SetDiagAdjElmMatrixAFromMC(i, N1FlagVec[i], N1IndMatrix[i], \
					N2FlagVec[i], N2IndMatrix[i])

	# **************************************************************************
	def _UpdateVectorZ_MC_AfterCal(self):
		# **********************************************************************
		# extract information a cap information dictionary to keep information
		# **********************************************************************
		N1FlagVec    = self.MCVectorZInfo["N1FlagVec"]
		N1IndMatrix  = self.MCVectorZInfo["N1IndMatrix"]
		N2FlagVec    = self.MCVectorZInfo["N2FlagVec"]
		N2IndMatrix  = self.MCVectorZInfo["N2IndMatrix"]

		# **********************************************************************
		# set the vector Z for each node
		# **********************************************************************
		for i in range(self.NumNodes):
			# ******************************************************************
			# check the flag and set vector Z
			# ******************************************************************
			self._UpdateVectorZfromMC(i, N1FlagVec[i], N1IndMatrix[i], N2FlagVec[i], \
					N2IndMatrix[i])

	# **************************************************************************
	def _UpdateMatrixMC_MatrixA_VectorZ_AfterCal(self, VectorX):
		"""
		# **********************************************************************
		# setup capacitor matrix for calculating MNA
		# setting capacitor matrix for updating Matrix A
		# **********************************************************************
		# **********************************************************************
		# parameters for memcapacitive matrix
		# **********************************************************************
		#   0 , 1 , 2,  3,    4
		# **********************************************************************
		#   N1, N2, C, dC, PreV
		# **********************************************************************
		"""
		# **********************************************************************
		# update PreV
		# **********************************************************************
		N1Ind   = self.MemCapNodes[:,0]
		N2Ind   = self.MemCapNodes[:,1]
		VN1     = VectorX[N1Ind]
		VN2     = VectorX[N2Ind]
		VMemCap = torch.subtract(VN1, VN2)
		self.MemCapMatrix[:,4] = VMemCap

		# **********************************************************************
		# update memcapacitance of MemCapMatrix
		# **********************************************************************
		NewMemC, dC   = self.MemCapObj.GetVals(VMemCap, self.dt)
		self.MemCapMatrix[:,2] = NewMemC
		self.MemCapMatrix[:,3] = dC

		# **********************************************************************
		# update entries of Matrix A
		# **********************************************************************
		self._UpdateMatrixA_MC_AfterCal()

		# **********************************************************************
		# update entries of Vector Z
		# **********************************************************************
		self._UpdateVectorZ_MC_AfterCal()

	# **************************************************************************
	#   Functions caculate X = A^-1*X for all timesteps.
	# **************************************************************************
	def _CalOneStep(self, Vins, TempVectorX):
		# **********************************************************************
		# update voltage values in the vector Z
		# **********************************************************************
		self.VectorZ[self.NumNodes:] = Vins

		# **********************************************************************
		# solve for the solutions
		# **********************************************************************
		try:
			# ******************************************************************
			# solve for X = A^-1 * Z
			# ******************************************************************
			TempVectorX[1:] = torch.linalg.solve(self.MatrixA, self.VectorZ)

		except Exception as Error:
			# ******************************************************************
			# raise exception
			# ******************************************************************
			raise ValueError(Error)

		# **********************************************************************
		# check for overflow error => NaN. Pytorch doesn't generate exception when
		# an overflow error occurs. This is an attempt to detect errors.
		# **********************************************************************
		if torch.isnan(TempVectorX[1:]).any():
			# # ******************************************************************
			# # clean up variables
			# # ******************************************************************
			# self._DeleteGPUVars()

			# ******************************************************************
			# setup error message
			# ******************************************************************
			ErrorMsg    = "%s: error <NaN>" % "MNA::_CalOneStep()"
			raise ValueError(ErrorMsg)

		# **********************************************************************
		# reset Matrix A
		# **********************************************************************
		if self.RFlag or self.CFlag:
			# ******************************************************************
			# reload matrix A with the constant values of R's and C's
			# ******************************************************************
			self.MatrixA[:] = self.SaveMatrixA[:]
		else:
			self.MatrixA[:self.NumNodes,:self.NumNodes] = 0.0

		# **********************************************************************
		# update vector Z
		# **********************************************************************
		if self.IsFlag:
			# ******************************************************************
			# reset the initial values of Vector Z
			# ******************************************************************
			self.VectorZ[:self.NumNodes] = self.SaveVectorZ[:self.NumNodes]
		else:
			self.VectorZ[:self.NumNodes] = 0.0

		# **********************************************************************
		# check the flag for updating CapMatrix and vector Z
		# **********************************************************************
		if self.CFlag:
			self._UpdateCapMatrix_VectorZ_AfterCal(TempVectorX)

		# **********************************************************************
		# check the flag for updating MemResMatrix and matrix A
		# **********************************************************************
		if self.MRFlag:
			self._UpdateMatrixMR_MatrixA_AfterCal(TempVectorX)

		# **********************************************************************
		# check the flag for updating MemCapMatrix and matrix A
		# **********************************************************************
		if self.MCFlag:
			self._UpdateMatrixMC_MatrixA_VectorZ_AfterCal(TempVectorX)

		return TempVectorX[1:]

	# **************************************************************************
	def _CalNStepVerbose(self, Vins, NumSteps, TempVectorX):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNA::_CalNStepVerbose()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			# display the information
			Msg = "...%-25s: calculating node voltages ..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# calculate node voltages
		# **********************************************************************
		for i in progressbar(range(NumSteps)):
			self.VectorX[i,1:] = self._CalOneStep(Vins[i], TempVectorX)

	# **************************************************************************
	def _CalNSteps(self, Vins, NumSteps, TempVectorX):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNA::_CalNSteps()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			# display the information
			Msg = "...%-25s: calculating node voltages ..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# calculate node voltages
		# **********************************************************************
		for i in range(NumSteps):
			self.VectorX[i,1:] = self._CalOneStep(Vins[i], TempVectorX)

	# **************************************************************************
	def CalBiasPoint(self):
		self.VectorX = torch.linalg.solve(self.MatrixA, self.VectorZ)
		return self.VectorX

	# **************************************************************************
	def CalNodeVoltages(self, Vins, dt, Verbose=None):
		"""
		Vins = [m x n]
			m: the number of time steps
			n: the number of voltage sources
		return: the node voltages
		"""
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNA::CalNodeVoltages()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			# display the information
			Msg = "...%-25s: calculating node voltages ..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# check the validity of input signals
		# **********************************************************************
		NumSteps, NumVs = Vins.shape
		if NumVs != self.NumVs:
			# ******************************************************************
			# setup error message
			# ******************************************************************
			ErrorMsg    = "%s: error <NumVs != self.NumVs>" % (FunctionName)
			raise ValueError(ErrorMsg)

		# **********************************************************************
		# check the validity of input signals
		# **********************************************************************
		self.dt         = dt
		self.VectorX    = torch.zeros((NumSteps, self.Size+1), dtype=torch.double)
		TempVectorX     = torch.zeros(self.Size+1, dtype=torch.double)

		# **********************************************************************
		# check the verbose flag
		# **********************************************************************
		if self.Verbose:
			self._CalNStepVerbose(Vins, NumSteps, TempVectorX)
		else:
			self._CalNSteps(Vins, NumSteps, TempVectorX)

		# **********************************************************************
		# return the node voltages
		# **********************************************************************
		return self.VectorX[:,:self.NumNodes+1]

# ******************************************************************************
"""
# ******************************************************************************
# Class decorator derived from the MNAClass
# ******************************************************************************
"""
class MNAGeneal(MNAClass):
	pass

# ******************************************************************************
"""
********************************************************************************
   Extra Functions for the MNA module
********************************************************************************
"""
# ******************************************************************************
def CreateMemDeviceModel(MemResName=None, MemCapName=None, DecayEffect=True, Verbose=None):
	# **************************************************************************
	# set the function name
	# **************************************************************************
	FunctionName = "CreateMemDeviceModel()"

	# **************************************************************************
	# display the message
	# **************************************************************************
	if Verbose:
		Msg = "...%-25s: create mem-device <%s, %s>..." % (FunctionName, \
				MemResName, MemCapName)
		print(Msg)

	# **************************************************************************
	# check memristive model
	# **************************************************************************
	MRMCFlag  = False
	if MemResName is not None:
		# check the model
		if (MemResName == "WeiLu") or (MemResName == "Chang"):
			# set the Weilu memristor model
			# self.MemResObj = WeiLuMem.WeiLuMemristor(InitVals=0.0, \
			#         DecayEffect=DecayEffect, Verbose=self.Verbose, \
			#         Theta=self.Theta, Vth=self.Vth)
			MemResObj = WeiLuMem.WeiLuMemristor(InitVals=0.0, \
					DecayEffect=DecayEffect, Verbose=Verbose)

		elif MemResName == "Oblea":
			# set the Oblea memristor model
			# self.MemResObj = ObleaMem.ObleaMemristor(InitVals=0.0, \
			#         DecayEffect=DecayEffect, Verbose=self.Verbose, \
			#         Theta=self.Theta, Vth=self.Vth)
			MemResObj = ObleaMem.ObleaMemristor(InitVals=0.0, \
					DecayEffect=DecayEffect, Verbose=Verbose)

		elif MemResName == "MemDeviceMR":
			MemResObj = MemDevMemristor(InitVals=0.5, \
					DecayEffect=DecayEffect, Verbose=Verbose)
			MRMCFlag  = True

		else:
			# set error message
			ErrMsg = "%s: <%s> => unknown memristive model in [%s, %s]" % (FunctionName, \
					MemResName, "WeiLu", "Oblea")
			raise ValueError(ErrMsg)

		# **********************************************************************
		# get the mem model name
		# **********************************************************************
		MemResName = MemResObj.GetModelName()
	else:
		# **********************************************************************
		# set the default object
		# **********************************************************************
		MemResObj = None

	# **************************************************************************
	# the memcapacitive model
	# **************************************************************************
	if MemCapName is not None:
		# check the model
		if MemCapName == "BiolekC4":
			# set the Oblea memristor model
			MemCapObj = BiolekC4Memcap.BiolekC4Memcapacitor(InitVals=0.0, \
					DecayEffect=DecayEffect, Verbose=Verbose)

		elif MemCapName == "Mohamed":
			# set the Oblea memristor model
			MemCapObj = MohamedMemcap.MohamedMemcapacitor(InitVals=0.0, \
					DecayEffect=DecayEffect, Verbose=Verbose)

		elif MemCapName == "Najem":
			# set the Oblea memristor model
			MemCapObj = NajemMemcap.NajemMemcapacitor(InitVals=0.0, \
					DecayEffect=DecayEffect, Verbose=Verbose)

		elif MemCapName == "MemDeviceMC":
			# set the Oblea memristor model
			MemCapObj = MemDevMemcapacitor(InitVals=MRInits, \
					DecayEffect=DecayEffect, Verbose=self)
			MRMCFlag  = True
		else:
			# set error message
			ErrMsg = "%s: <%s> => unknown memcapacitive model in [%s, %s]" % (FunctionName, \
					self.MemCapObj, "BiolekC4", "Mohamed")
			raise ValueError(ErrMsg)

		# **********************************************************************
		# get the mem model name
		# **********************************************************************
		MemCapName = MemCapObj.GetModelName()

	else:
		# **********************************************************************
		# set the default object
		# **********************************************************************
		MemCapObj = None

	return MemResObj, MemCapObj, MRMCFlag

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
	# import memristive models
	# **************************************************************************
	import MemristiveModels.WeiLuMemristorTorch as WeiLuMem
	import MemristiveModels.ObleaMemristorTorch as ObleaMem

	# **************************************************************************
	# import memcapacitive models
	# **************************************************************************
	import MemcapacitiveModels.BiolekC4MemcapacitorTorch as BiolekC4Memcap
	import MemcapacitiveModels.MohamedMemcapacitorTorch as MohamedMemcap
	import MemcapacitiveModels.NajemMemcapacitorTorch as NajemMemcap

	# **************************************************************************
	# MRMC device model
	# **************************************************************************
	from MemDeviceTorch import Memristor as MemDevMemristor
	from MemDeviceTorch import Memcapacitor as MemDevMemcapacitor

	# **************************************************************************
	# Device will determine whether to run the training on GPU or CPU.
	# **************************************************************************
	Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# **************************************************************************
	# process the arguments
	# **************************************************************************
	args = ReadArguments()

	# **************************************************************************
	# parameters for MNA circuit
	# **************************************************************************
	Verbose     = True
	# set the memcapacitor model
	MemCapName  = "BiolekC4"
	# MemCapName  = "Mohamed"
	# MemCapName  = "Najem"
	# MemCapName = None

	# set the memristor model
	MemResName  = "Chang"
	# MemResName  = "Oblea"
	# MemResName = None

	# **************************************************************************
	# create mem-device objects for MNA circuit
	# **************************************************************************
	MemResObj, MemCapObj, MRMCFlag = CreateMemDeviceModel(MemResName=MemResName, \
			MemCapName=MemCapName, Verbose=Verbose)

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
	# extract the component list
	# **************************************************************************
	dt = ts[1] - ts[0]
	TorchFlag	= True
	InParams	= ReadAndExtractNetlist(args.Netlist, dt=dt, TorchFlag=TorchFlag, Verbose=Verbose)

	# **************************************************************************
	# read components from the netlist
	# **************************************************************************
	InParams["MemResObj"] = MemResObj
	InParams["MemCapObj"] = MemCapObj

	# **************************************************************************
	# Create MNA object and build Matrix A, Vector X, and Vector Z
	# **************************************************************************
	print(" ")
	Verbose         = True
	# MNAObj          = MNAClass(InParams=InParams, Verbose=Verbose)
	MNAObj          = MNAGeneal(InParams=InParams, Verbose=Verbose)
	MNAObj.to(Device)

	exit()
	# **************************************************************************
	# calculate node voltages
	# **************************************************************************
	# Vins      = torch.zeros((1, 1), dtype=torch.double)
	# Vins[0,0] = 40
	# dt = 0

	# Vins    = torch.reshape(v, (len(v), 1))
	# dt      = t[1] - t[0]
	VNodes  = MNAObj.CalNodeVoltages(Vins, dt)

	# **************************************************************************
	# Get the matrix and vectors
	# **************************************************************************
	ObjDict = MNAObj.GetMatrixAndVectors()
	MatrixA = ObjDict["MatrixA"]
	VectorZ = ObjDict["VectorZ"]
	VectorX = ObjDict["VectorX"]

	print("Matrix A")
	print(pd.DataFrame(MatrixA))
	print("Vector Z")
	print(pd.DataFrame(VectorZ))
	print("Vector X")
	print(pd.DataFrame(VectorX))
	#
	# print("VNodes")
	# print(pd.DataFrame(VNodes))
	# exit()

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
	V2Lbl   = r"$v_2(t)$"
	V3Lbl   = r"$v_3(t)$"

	# **************************************************************************
	# get the figure handle
	# **************************************************************************
	Fig = plt.figure("Input Signal")
	# plt.title("Biolek Q-V Plot")
	plt.grid(linestyle="dotted")
	plt.plot(ts, Vins[:,0], label=VsLbl, linewidth=LineWidth)
	plt.plot(ts, VNodes[:,1], label=V2Lbl, linewidth=LineWidth)
	plt.plot(ts, VNodes[:,2], label=V3Lbl, linewidth=LineWidth)
	plt.xlabel("time (s)")
	plt.ylabel("Voltage (V)")
	plt.legend(loc="best")
	plt.axis("tight")

	# **************************************************************************
	# set the file name
	# **************************************************************************
	if args.Netlist == "Ex2_RNetlist.cir":
		FileName    = "Ex2_MNA_R_%s.png" % (Type)
	elif args.Netlist == "Ex3_RNetlist.cir":
		FileName    = "Ex3_MNA_R_%s.png" % (Type)
	elif args.Netlist == "Ex4_CNetlist.cir":
		FileName    = "Ex4_MNA_C_%s.png" % (Type)
	elif args.Netlist == "Ex5_CNetlist.cir":
		FileName    = "Ex5_MNA_C_%s.png" % (Type)
	elif args.Netlist == "Ex6_MRNetlist.cir":
		FileName    = "Ex6_MNA_MR_%s.png" % (Type)
	elif args.Netlist == "Ex7_MCNetlist.cir":
		FileName    = "Ex7_MNA_MC_%s.png" % (Type)
	elif args.Netlist == "Ex18_RNetlist.cir":
		FileName    = "Ex18_MNA_R_%s.png" % (Type)

	elif args.Netlist == "Ex7_8_RNetlist.cir":
		FileName    = "Ex7_8_MNA_R_%s.png" % (Type)

	# FileNameEps = join("Figures", "MemRes%sRV.eps" % ModelName)
	# FileNameJpg = join("Figures", "MemRes%sRV.jpg" % ModelName)
	FileNamePng = join("Figures", "Module7", FileName)
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
