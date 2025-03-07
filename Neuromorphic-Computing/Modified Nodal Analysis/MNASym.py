# ******************************************************************************
# import modules
# ******************************************************************************
import numpy as np
import torch
import torch.nn as nn
import sympy
import pprint

# ******************************************************************************
# import from other module
# ******************************************************************************
from MNAHelpers import ReadArguments, ReadAndExtractNetlist

# ******************************************************************************
# MNASym class to build Matrix A, vect X, and vector Z
# ******************************************************************************
class MNASym(nn.Module):
	def __init__(self, InParams=None, Verbose=False):
		# **********************************************************************
		super(MNASym, self).__init__()

		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNASym::__init__()"

		# **********************************************************************
		# Save the network parameters
		# **********************************************************************
		self.InParams	= InParams
		self.KeyVals	= InParams["KeyVals"]
		self.dt      	= InParams["dt"]
		self.NodeList	= InParams["NodeList"]
		self.Verbose 	= Verbose

		# **********************************************************************
		# reset symbolic matrices
		# **********************************************************************
		self.SymVsList 	= None
		self.SymIsList 	= None
		self.SymRList	= None
		self.SymCList	= None
		self.SymLList	= None
		self.SymMRList	= None
		self.SymMCList	= None

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
		# reset all variables
		# **********************************************************************
		self.NumNodes		=  len(self.NodeList) - 1

		# **********************************************************************
		# build all lists
		# **********************************************************************
		self._BuildAllLists()

		# **********************************************************************
		# check the flag
		# **********************************************************************
		if self.VsFlag:
			self.NumVs	= len(self.VsList)
		if self.IsFlag:
			self.NumIs	= len(self.IsList)
		if self.RFlag:
			self.NumR	= len(self.RList)
		if self.CFlag:
			self.NumC	= len(self.CList)
		if self.MRFlag:
			self.NumMR	= len(self.MRList)
		if self.MCFlag:
			self.NumMC	= len(self.MCList)

		# **********************************************************************
		# set the matrix size and indices
		# **********************************************************************
		self.MatrixSize = self.NumNodes + self.NumVs
		self.Indices    = sympy.Array(range(self.MatrixSize))

		# **********************************************************************
		# initializing symbolic Matrix A, vector X, vector Z
		# **********************************************************************
		self.SymMatrixA = sympy.zeros(self.MatrixSize, self.MatrixSize)
		self.SymVectorX = sympy.zeros(1, self.MatrixSize)
		self.SymVectorZ = sympy.zeros(1, self.MatrixSize)

		# **********************************************************************
		# build symbolic Matrix A and vector Z
		# **********************************************************************
		if self.VsFlag:
			self._BuildSymAandZFromVList(self.SymVsList, self.VsList)

		if self.IsFlag:
			self._BuildSymZFromIsList(self.SymIsList, self.IsList)

		if self.RFlag:
			self._BuildSymAFromRList(self.SymRList, self.RList)

		if self.CFlag:
			self._BuildSymAFromCList(self.SymCList, self.CList)
			self._BuildSymZFromCList(self.SymCList, self.CList)

		if self.MRFlag:
			self._BuildSymAFromRList(self.SymMRList, self.MRList)

		if self.MCFlag:
			self._BuildSymAFromMCList(self.SymMCList, self.MCList)
			self._BuildSymZFromCList(self.SymMCList, self.MCList)

		# **********************************************************************
		# set the vector X
		# **********************************************************************
		self._BuildSymVectorX()

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			# ******************************************************************
			# display the information
			# ******************************************************************
			Msg = "\n==> MNASym Class ..."
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
			Msg = "...%-25s: matrix size = %d x %d" % (FunctionName, self.MatrixSize, \
					self.MatrixSize)
			print(Msg)

			# ******************************************************************
			# display the information
			# ******************************************************************
			Msg = "...%-25s: Matrix A = %s, Vector X = %s, Vector Z = %s" % (FunctionName, \
					str(self.SymMatrixA.shape), str(self.SymVectorX.shape), str(self.SymVectorZ.shape))
			print(Msg)

	# **************************************************************************
	def _ConvertSym(self, Dev, List):
		return np.asarray([sympy.symbols(AVal) for AVal in Dev])

	# **************************************************************************
	def _ConvertSymCList(self, Dev, List):
		# **********************************************************************
		# convert to device name to symbols
		# **********************************************************************
		NumDev  = len(Dev)
		AMatrix = sympy.zeros(NumDev, 3)
		for i in range(NumDev):
			AMatrix[i,0] = sympy.Symbol("%s/dt" % (Dev[i]))

			# ******************************************************************
			# set the device node voltage
			# ******************************************************************
			N1  = int(List[i,0])
			N2  = int(List[i,1])
			if N1 != 0:
				AMatrix[i,1] = sympy.Symbol("V%dPrev" % (N1))
			if N2 != 0:
				AMatrix[i,2] = sympy.Symbol("V%dPrev" % (N2))
		return AMatrix

	# **************************************************************************
	def _ConvertSymMCList(self, Dev, List):
		# **********************************************************************
		# convert to device name to symbols
		# **********************************************************************
		NumDev  = len(Dev)
		AMatrix = sympy.zeros(NumDev, 4)
		for i in range(NumDev):
			AMatrix[i,0] = sympy.Symbol("%s/dt" % (Dev[i]))
			AMatrix[i,3] = sympy.Symbol("d%s/dt" % (Dev[i]))

			# ******************************************************************
			# set the device node voltage
			# ******************************************************************
			N1  = int(List[i,0])
			N2  = int(List[i,1])
			if N1 != 0:
				AMatrix[i,1] = sympy.Symbol("V%dPrev" % (N1))
			if N2 != 0:
				AMatrix[i,2] = sympy.Symbol("V%dPrev" % (N2))

		return AMatrix

	# **************************************************************************
	def _BuildAllLists(self):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNASym::_BuildAllList()"

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
				Dev		= ListType["Dev"]
				List	= ListType["List"]
				if torch.is_tensor(List):
					List	= List.numpy()
				NotEmptyFlag = True

			if AKey == "VsList":
				self.VsFlag = NotEmptyFlag
				if self.VsFlag:
					self.VsList		= List
					self.SymVsList 	= self._ConvertSym(Dev, List)
				else:
					self.VsList		= None
					self.SymVsList 	= None

			elif AKey == "IsList":
				self.IsFlag = NotEmptyFlag
				if self.IsFlag:
					self.IsList		= List
					self.SymIsList 	= self._ConvertSym(Dev, List)
				else:
					self.IsList		= None
					self.SymIsList 	= None

			elif AKey == "RList":
				self.RFlag = NotEmptyFlag
				if self.RFlag:
					self.RList		= List
					self.SymRList 	= self._ConvertSym(Dev, List)
				else:
					self.RList		= None
					self.SymRList 	= None

			elif AKey == "CList":
				self.CFlag = NotEmptyFlag
				if self.CFlag:
					self.CList		= List
					self.SymCList 	= self._ConvertSymCList(Dev, List)
				else:
					self.CList		= None
					self.SymCList 	= None

			elif AKey == "LList":
				self.LFlag = NotEmptyFlag
				if self.LFlag:
					self.LList		= List
					self.SymLList 	= self._ConvertSym(Dev, List)
				else:
					self.LList		= None
					self.SymLList 	= None

			elif AKey == "MRList":
				self.MRFlag = NotEmptyFlag
				if self.MRFlag:
					self.MRList		= List
					self.SymMRList 	= self._ConvertSym(Dev, List)
				else:
					self.MRList		= None
					self.SymMRList 	= None

			elif AKey == "MCList":
				self.MCFlag = NotEmptyFlag
				if self.MCFlag:
					self.MCList		= List
					self.SymMCList 	= self._ConvertSymMCList(Dev, List)
				else:
					self.MCList		= None
					self.SymMCList 	= None

			else:
				# **************************************************************
				# format error message
				# **************************************************************
				Msg = "%s: unknown device => <%s>" % (FunctionName, AKey)
				raise ValueError(Msg)

	# **************************************************************************
	def _BuildSymAandZFromVList(self, Dev, VsList):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNA::_BuildSymAandZFromVList()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			Msg = "...%-25s: set Symbolic Matrix A and Vector Z from VsList..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# set the Matrix A with voltage source list
		# **********************************************************************
		for i in range(len(VsList)):
			# ******************************************************************
			# set row and column indices
			# ******************************************************************
			RowIndex = self.NumNodes + i
			ColIndex = self.NumNodes + i

			# ******************************************************************
			# set the coefficients for voltage sources
			# ******************************************************************
			N1	= int(VsList[i, 0])
			N2	= int(VsList[i, 1])
			Val = VsList[i, 2]
			if N1 > 0:
				N1 -= 1
				self.SymMatrixA[N1, ColIndex] = 1
				self.SymMatrixA[RowIndex, N1] = 1

			if N2 > 0:
				N2 -= 1
				self.SymMatrixA[N2, ColIndex] = -1
				self.SymMatrixA[RowIndex, N2] = -1

		# **********************************************************************
		# set the vector Z
		# **********************************************************************
		for i in range(len(Dev)):
			SetIndex = self.NumNodes + i
			self.SymVectorZ[SetIndex] = Dev[i]

	# **************************************************************************
	def _BuildSymZFromIsList(self, Dev, IsList):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNA::_BuildSymZFromIsList()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			Msg = "...%-25s: set vector Z from IsList..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# set the vector Z for each node
		# **********************************************************************
		for i in range(self.NumNodes):
			Node    = i + 1
			# ******************************************************************
			# list of nodes on N1 and N2
			# ******************************************************************
			N1_Ind  = IsList[:,0] == Node
			N2_Ind  = IsList[:,1] == Node

			# ******************************************************************
			# set the flag for valid entries
			# ******************************************************************
			N1Flag  = np.any(N1_Ind)
			N2Flag  = np.any(N2_Ind)

			# ******************************************************************
			# check the flag and set vector Z
			# ******************************************************************
			if N1Flag:
				N1Vals  = Dev[N1_Ind]
				self.SymVectorZ[i] -= np.sum(N1Vals)
			if N2Flag:
				N2Vals  = Dev[N2_Ind]
				self.SymVectorZ[i] += np.sum(N2Vals)

	# **************************************************************************
	def _BuildSymAFromRList(self, Dev, RList):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNA::_BuildSymAFromRList()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			Msg = "...%-25s: set Symbolic Matrix A from RList..." % (FunctionName)
			print(Msg)


		# **********************************************************************
		# set the matrix A for each node
		# **********************************************************************
		for i in range(self.NumNodes):
			# ******************************************************************
			# set the number number since i start at 0
			# ******************************************************************
			Node    = i + 1

			# ******************************************************************
			# list of nodes on N1 and N2
			# ******************************************************************
			N1_Ind  = RList[:,0] == Node
			N2_Ind  = RList[:,1] == Node

			# ******************************************************************
			# set the flag for valid entries
			# ******************************************************************
			N1Flag  = np.any(N1_Ind)
			N2Flag  = np.any(N2_Ind)

			# ******************************************************************
			# resistors at N1 and N2
			# ******************************************************************
			if N1Flag:
				N1_InvRVals = 1/Dev[N1_Ind]
				N1Total = np.sum(N1_InvRVals)
			else:
				N1Total = 0

			if N2Flag:
				N2_InVRVals = 1/Dev[N2_Ind]
				N2Total = np.sum(N2_InVRVals)
			else:
				N2Total = 0

			# ******************************************************************
			# set the element of Matrix A
			# ******************************************************************
			self.SymMatrixA[i,i]  += (N1Total + N2Total)

			# print("SymMatrixA")
			# pprint.pprint(self.SymMatrixA)

			# ******************************************************************
			# adjacent nodes of N1
			# ******************************************************************
			if N1Flag:
				# **************************************************************
				# update adjacent nodes of i
				# **************************************************************
				RDev		= Dev[N1_Ind]
				N1Entries   = RList[N1_Ind]
				N2NonZero   = N1Entries[:,1] != 0

				# **************************************************************
				# check for any value
				# **************************************************************
				if np.any(N2NonZero):
					N2Nodes = N1Entries[N2NonZero, 1] - 1
					N2Vals  = 1/RDev[N2NonZero]

					# **********************************************************
					# set the adjecent cells
					# **********************************************************
					for j in range(len(N2Nodes)):
						Index	= int(N2Nodes[j])
						self.SymMatrixA[i,Index]	-= N2Vals[j]
						self.SymMatrixA[Index, i] 	-= N2Vals[j]

	# **************************************************************************
	def _BuildSymAFromCList(self, Dev, CList):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNA::_BuildSymAFromCList()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			Msg = "...%-25s: set Symbolic Matrix A from CList..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# set the matrix A for each node
		# **********************************************************************
		for i in range(self.NumNodes):
			# ******************************************************************
			# set the number number since i start at 0
			# ******************************************************************
			Node    = i + 1

			# ******************************************************************
			# list of nodes on N1 and N2
			# ******************************************************************
			N1_Ind  = CList[:,0] == Node
			N2_Ind  = CList[:,1] == Node

			# ******************************************************************
			# set the flag for valid entries
			# ******************************************************************
			N1Flag  = np.any(N1_Ind)
			N2Flag  = np.any(N2_Ind)

			# ******************************************************************
			# resistors at N1 and N2
			# ******************************************************************
			if N1Flag:
				Indices	= np.nonzero(N1_Ind)[0].astype(int)
				N1Total	= 0
				for k in Indices:
					N1Total	+= Dev[k, 0]
			else:
				N1Total = 0

			if N2Flag:
				Indices	= np.nonzero(N2_Ind)[0].astype(int)
				N2Total	= 0
				for k in Indices:
					N2Total	+= Dev[k, 0]
			else:
				N2Total = 0

			# ******************************************************************
			# set the element of Matrix A
			# ******************************************************************
			self.SymMatrixA[i,i]  += (N1Total + N2Total)

			# ******************************************************************
			# adjacent nodes of N1
			# ******************************************************************
			if N1Flag:
				# **************************************************************
				# update adjacent nodes of i
				# **************************************************************
				N1Entries   = CList[N1_Ind]
				N2NonZero   = N1Entries[:,1] != 0

				# **************************************************************
				# check for any value
				# **************************************************************
				if np.any(N2NonZero):
					N2Nodes = N1Entries[N2NonZero, 1].astype(int) - 1
					Indices	= np.nonzero(N2NonZero)[0].astype(int)
					N2Vals	= 0
					for k in Indices:
						N2Vals	+= Dev[k, 0]

					# **********************************************************
					# set the adjecent cells
					# **********************************************************
					for j in range(len(N2Nodes)):
						Index	= int(N2Nodes[j])
						self.SymMatrixA[i,Index]	-= N2Vals
						self.SymMatrixA[Index, i] 	-= N2Vals

	# **************************************************************************
	def _BuildSymAFromMCList(self, Dev, MCList):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNA::_BuildSymAFromMCList()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			Msg = "...%-25s: set Symbolic Matrix A from MCList..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# set the matrix A for each node
		# **********************************************************************
		for i in range(self.NumNodes):
			# ******************************************************************
			# set the number number since i start at 0
			# ******************************************************************
			Node    = i + 1

			# ******************************************************************
			# list of nodes on N1 and N2
			# ******************************************************************
			N1_Ind  = MCList[:,0] == Node
			N2_Ind  = MCList[:,1] == Node

			# ******************************************************************
			# set the flag for valid entries
			# ******************************************************************
			N1Flag  = np.any(N1_Ind)
			N2Flag  = np.any(N2_Ind)

			# ******************************************************************
			# resistors at N1 and N2
			# ******************************************************************
			if N1Flag:
				Indices	= np.nonzero(N1_Ind)[0].astype(int)
				N1Total	= 0
				for k in Indices:
					N1Total	+= (Dev[k, 0] + Dev[k, 3])
			else:
				N1Total = 0

			if N2Flag:
				Indices	= np.nonzero(N2_Ind)[0].astype(int)
				N2Total	= 0
				for k in Indices:
					N2Total	+= (Dev[k, 0] + Dev[k, 3])
			else:
				N2Total = 0

			# ******************************************************************
			# set the element of Matrix A
			# ******************************************************************
			self.SymMatrixA[i,i]  += (N1Total + N2Total)

			# ******************************************************************
			# adjacent nodes of N1
			# ******************************************************************
			if N1Flag:
				# **************************************************************
				# update adjacent nodes of i
				# **************************************************************
				N1Entries   = MCList[N1_Ind]
				N2NonZero   = N1Entries[:,1] != 0

				# **************************************************************
				# check for any value
				# **************************************************************
				if np.any(N2NonZero):
					N2Nodes = N1Entries[N2NonZero, 1].astype(int) - 1
					Indices	= np.nonzero(N2NonZero)[0].astype(int)
					N2Vals	= 0
					for k in Indices:
						N2Vals	+= (Dev[k, 0] + Dev[k, 3])

					# **********************************************************
					# set the adjecent cells
					# **********************************************************
					for j in range(len(N2Nodes)):
						Index	= int(N2Nodes[j])
						self.SymMatrixA[i,Index]	-= N2Vals
						self.SymMatrixA[Index, i] 	-= N2Vals

	# **************************************************************************
	def _BuildSymZFromCList(self, Dev, CList):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNA::_BuildSymZFromCList()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			Msg = "...%-25s: set symbolic Vector Z from CList..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# build the vector z
		# **********************************************************************
		for i in range(len(CList)):
			ADevInfo = CList[i]
			N1 = int(ADevInfo[0])
			N2 = int(ADevInfo[1])

			# ******************************************************************
			# extract values
			# ******************************************************************
			CVal = Dev[i, 0]
			VN1  = Dev[i, 1]
			VN2  = Dev[i, 2]

			# ******************************************************************
			# check the node N1
			# ******************************************************************
			if N1 > 0:
				N1 -= 1
				# **************************************************************
				# calculate capacitive currents
				# **************************************************************
				if N2 > 0:
					iCVals = (CVal) * (VN1 - VN2)
				else:
					iCVals = (CVal) * VN1

				# **************************************************************
				# update Vector Z
				# **************************************************************
				self.SymVectorZ[N1] += iCVals

			# ******************************************************************
			# check the node N2
			# ******************************************************************
			if N2 > 0:
				N2 -= 1
				# **************************************************************
				# calculate capacitive currents
				# **************************************************************
				if N1 > 0:
					iCVals = (CVal) * (VN2 - VN1)
				else:
					iCVals = -(CVal) * VN2

				# **************************************************************
				# update Vector Z
				# **************************************************************
				self.SymVectorZ[N2] += iCVals

	# **************************************************************************
	def _BuildSymVectorX(self):
		# **********************************************************************
		# set the function name
		# **********************************************************************
		FunctionName = "MNASym::_BuildSymVectorX()"

		# **********************************************************************
		# display the message
		# **********************************************************************
		if self.Verbose:
			# display the information
			Msg = "...%-25s: Building Vector X..." % (FunctionName)
			print(Msg)

		# **********************************************************************
		# set the voltage symbols
		# **********************************************************************
		j = 0
		for i in range(self.NumNodes):
			j = i + 1
			VSymbol     =  sympy.Symbol("V%d" % (j))
			self.SymVectorX[i]   = VSymbol

		# **********************************************************************
		# set the voltage current symbols
		# **********************************************************************
		for i in range(self.NumVs):
			IVSymbol = sympy.Symbol("IV%d" % (i+1))
			self.SymVectorX[j] = IVSymbol
			j += 1

	# **************************************************************************
	def GetMatrixAndVectors(self):
		return {"MatrixA": self.SymMatrixA, "VectorX": self.SymVectorX, "VectorZ": self.SymVectorZ}

# ******************************************************************************
if __name__ == '__main__':
	# **************************************************************************
	# process the arguments
	# **************************************************************************
	args = ReadArguments()

	# **************************************************************************
	# default values
	# **************************************************************************
	dt		= 1e-6
	Verbose	= True

	# **************************************************************************
	# read components from the netlist
	# **************************************************************************
	InParams	= ReadAndExtractNetlist(args.Netlist, dt=dt, Verbose=Verbose)
	# print(InParams)
	# exit()
	# **************************************************************************
	# Create MNASym object and build Matrix A, Vector X, and Vector Z
	# **************************************************************************
	print("\n")
	MNASymObj  = MNASym(InParams=InParams, Verbose=Verbose)

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
	print("\n\nMatrix A")
	pprint.pprint(MatrixA)
	print("\n\nVector X")
	pprint.pprint(VectorX)
	print("\n\nVector Z")
	pprint.pprint(VectorZ)
