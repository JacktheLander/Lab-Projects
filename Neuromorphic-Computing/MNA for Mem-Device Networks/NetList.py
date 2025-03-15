# **************************************************************************# **************************************************************************# ******************************************************************************
# import modules
# ******************************************************************************
import numpy as np

# ******************************************************************************
# torch
# ******************************************************************************
import torch
import torch.nn as nn

# ******************************************************************************
class NetList(nn.Module):
    """
    Creating NetLists of different components based on the input device list.
    """
    # def __init__(self, DevList=None, VsList=None, VsInits=None, IsList=None, IsInits=None,\
    #         RList=None, CList=None,\
    #         MemResObj=None, MRList=None, MRInits=None, \
    #         MemCapObj=None, MCList=None, MCInits=None, \
    #         TrapezoidFlag=False, dt=1e-9, PerMemC=100.0,\
    #         CalPower=False, DecayEffect=False, NumInputs=0, NumOutputs=0,\
    #         InputNodes=None, OutputNodes=None, \
    #         Theta=1.0, \
    #         LeakageOpt=False, NanoWireFlag=False, \
    #         Vth=0.0, SetMCVth=False, PowerMRFlag=False, \
    #         InputRFlag=False, GPU_Flag=False, Device=False, Verbose=False):
    def __init__(self, NetParams=None):
        # **********************************************************************
        # set the dimension of the super class
        # **********************************************************************
        super(NetList, self).__init__()

        # **********************************************************************
        """ Initializes and constructs a random reservoir.
        Parameters are:
            - DevList       : predefined device list.

        """
        # **********************************************************************
        # set the function name
        # **********************************************************************
        FunctionName = "NetList::__init__()"

        # **********************************************************************
        # memristor model name
        # **********************************************************************
        self.ClassName = "NetList"

        # **********************************************************************
        # save parameters
        # **********************************************************************
        """
          The format of the device list
        # * ********************************************************************
        # * 0, 1, 2 , 3
        # **********************************************************************
        # * a, b, Rw, Distance
        """
        self.DevList    = NetParams["DevList"]
        self.InNodes    = NetParams["InNodes"]
        self.PerMemC    = NetParams["PerMemC"]
        self.MRMCFlag   = NetParams["MRMCFlag"]
        self.Verbose    = NetParams["Verbose"]

        # **********************************************************************
        # reset variables
        # **********************************************************************
        self.VsList     = None
        self.VsInits    = 0

        self.IsList     = None
        self.NumIs      = 0

        self.RList      = None
        self.NumR       = 0

        self.CList      = None
        self.NumC       = 0

        self.MRList     = None
        self.NumMRes    = 0

        self.MCList     = None
        self.NumMCap    = 0

        self.InputNodes = None
        self.AvaiOutNodes = None
        # self.InputNodes = None

        # **********************************************************************
        # Ground resistor for dangling nodes
        # **********************************************************************
        self.GndResistor = 100e6

        # **********************************************************************
        # display the message
        # **********************************************************************
        if self.Verbose:
            # display the information
            Msg = "\n==> Instantiating <%s>..." % (self.ClassName)
            print(Msg)

        # **********************************************************************
        # remove one-node device
        # **********************************************************************
        self.DevList    = self._RemoveOneNodeDevice(self.DevList)
        self.NumDev     = len(self.DevList)

        # **********************************************************************
        # get the node list
        # **********************************************************************
        self.NodeList   = np.unique(self.DevList[:,:2]).astype(int)

        # **********************************************************************
        # check the node list for missing nodes
        # **********************************************************************
        MaxNodeNum  = self.NodeList[-1]
        if MaxNodeNum >= len(self.NodeList):
            # there are missing nodes from the node list, rebuild the node list
            self.NodeList   = np.arange(MaxNodeNum + 1)

        # **********************************************************************
        # save the number of nodes in the reservoir
        # **********************************************************************
        self.NumNodes  = len(self.NodeList)

        # **********************************************************************
        # copy the node list
        # **********************************************************************
        self.AvaiOutNodes = np.copy(self.NodeList)

        # **********************************************************************
        # check for the ground node
        # **********************************************************************
        if 0 in self.NodeList:
            self.NumNodes = len(self.NodeList) - 1

            # ******************************************************************
            # delete the ground node
            # ******************************************************************
            self.AvaiOutNodes = np.delete(self.AvaiOutNodes, 0)
        else:
            self.NumNodes = len(self.NodeList)

        # **********************************************************************
        # create check the check list of connections numbers
        # **********************************************************************
        self.NumConAtEachNodeList  = self._CreateNodeList(NetList=self.DevList, VsList=None)
        # print("NumConAtEachNodeList")
        # print(pd.DataFrame(self.NumConAtEachNodeList))
        # exit()

        # **********************************************************************
        # grounding dangling nodes: number of connection (NumCon) = 0
        # **********************************************************************
        self.DanglingNodes = self._CheckNodeList(self.DevList, NumCon=0)

        # print("DanglingNodes")
        # print(pd.DataFrame(self.DanglingNodes))
        # print(self.NumConAtEachNodeList[self.DanglingNodes])

        # **********************************************************************
        # add ground resistance for dangling nodes (0 connection nodes)
        # **********************************************************************
        if self.DanglingNodes is not None:
            # set the ground resistors
            self._SetRGndList(self.DanglingNodes)

            # remove the dangling nodes from the node list
            self.AvaiOutNodes = np.setdiff1d(self.AvaiOutNodes, self.DanglingNodes)

            # print("DanglingNodes")
            # print(pd.DataFrame(self.DanglingNodes))
            # print(pd.DataFrame(self.RList))

        # **********************************************************************
        # set the input nodes and voltage source list
        # **********************************************************************
        if self.InputNodes is None:
            self.InputNodes = self._SelectRandomNodes(self.AvaiOutNodes, self.InNodes)
        else:
            # ******************************************************************
            # display the message
            # ******************************************************************
            if self.Verbose:
                # display the information
                Msg = "...%-25s: input node list is loaded" % (FunctionName)
                print(Msg)

        # print("self.InNodes = ", self.InNodes)
        # print("InputNodes   = ", self.InputNodes)
        # exit()

        # **********************************************************************
        # create the Vs list from the input nodes
        # **********************************************************************
        if self.VsList is None:
            self.VsList = self._CreatVsList(self.InputNodes, VsInits=self.VsInits)

            # update the number of connections for input nodes
            self.NumConAtEachNodeList[self.InputNodes,1] += 1

        # **********************************************************************
        # remove the input nodes from the node list
        # **********************************************************************
        self.AvaiOutNodes = np.setdiff1d(self.AvaiOutNodes, self.InputNodes)

        # **********************************************************************
        # select 1-connection nodes: number of connection (NumCon) = 1
        # **********************************************************************
        self.OneConnectionNodes = self._CheckNodeList(self.DevList, NumCon=1)

        # **********************************************************************
        # add ground resistance for 1-connection nodes (0 connection nodes)
        # **********************************************************************
        if self.OneConnectionNodes is not None:
            self._SetRGndList(self.OneConnectionNodes)

        # **********************************************************************
        # set the connection list for mem-devices based on percentage of MC
        # **********************************************************************
        self._GenerateMemLists(self.DevList, self.PerMemC, MRMCFlag=self.MRMCFlag)

        # print("DevList  = ", self.DevList.shape)
        # if self.MRList is not None:
        #     print("self.MRList    = ", self.MRList.shape)
        #     print(pd.DataFrame(self.MRList))
        #
        # if self.MCList is not None:
        #     print("self.MCList    = ", self.MCList.shape)
        #     print(pd.DataFrame(self.MCList))
        # exit()

        # **********************************************************************
        # display the message
        # **********************************************************************
        if self.Verbose:
            # display the information
            Msg = "\n==> Instantiating <%s>..." % (self.ClassName)
            print(Msg)

        # **********************************************************************
        # display the message
        # **********************************************************************
        if self.Verbose:
            # display the information
            Msg = "...%-25s: NumDev = %d, Input nodes = %d, PerMemC = %.1f, MRMC = %s" % \
                    (FunctionName, self.NumDev, self.InNodes, self.PerMemC, self.MRMCFlag)
            print(Msg)

            # display the information
            Msg = "...%-25s: VsList = %s, RList = %s, GndR = %.2f Mohm" % \
                    (FunctionName, str(self.VsList.shape), str(self.RList.shape), self.GndResistor/1e6)
            print(Msg)

            # display the information
            Msg = "...%-25s: invalid nodes = %s" % (FunctionName, str(self.DanglingNodes))
            print(Msg)

            # display the information
            Msg = "...%-25s: One connection nodes = %s" % (FunctionName, str(self.OneConnectionNodes))
            print(Msg)

            # display the information
            if self.NumIs > 0:
                Msg = "...%-25s: IsList = %s" % (FunctionName, str(self.IsList.shape))
                print(Msg)

            # display the information
            if self.NumC > 0:
                Msg = "...%-25s: CList = %s" % (FunctionName, str(self.CList.shape))
                print(Msg)

            # display the information
            if self.NumMRes > 0:
                Msg = "...%-25s: MRList = %s" % (FunctionName, str(self.MRList.shape))
                print(Msg)

            # display the information
            if self.NumMCap > 0:
                Msg = "...%-25s: MCList = %s" % (FunctionName, str(self.MCList.shape))
                print(Msg)

    # **************************************************************************
    def _FindOneNodeDevices(self, List=None):
        # **********************************************************************
        # set the function name
        # **********************************************************************
        FunctionName = "NetList::_FindOneNodeDevices()"

        # **********************************************************************
        # finding the one node device indices
        # **********************************************************************
        OneNodeDevInd = np.asarray([i for i in range(len(List)) if List[i,0] == List[i,1]])

        # **********************************************************************
        # display the message
        # **********************************************************************
        if self.Verbose:
            # ******************************************************************
            # display the information
            # ******************************************************************
            Msg = "...%-25s: finding one node devices, NumDev = <%d> ..." % (FunctionName, \
                    OneNodeDevInd.size)
            print(Msg)

        return OneNodeDevInd

    # **************************************************************************
    def _RemoveOneNodeDevice(self, List=None):
        # **********************************************************************
        # set the function name
        # **********************************************************************
        FunctionName = "NetList::_RemoveOneNodeDevice()"

        # **********************************************************************
        # display the message
        # **********************************************************************
        if self.Verbose:
            # display the information
            Msg = "...%-25s: remove one node devices, len = <%d>..." % \
                    (FunctionName, len(List))
            print(Msg)

        # **********************************************************************
        # get the indices of one node devices
        # **********************************************************************
        OneNodeDevIndices   = self._FindOneNodeDevices(List)

        # **********************************************************************
        # check the list
        # **********************************************************************
        if OneNodeDevIndices.size > 0:
            # total indices
            TotalIndices    = np.arange(0, len(List))

            # result indices
            ResultIndices   = np.delete(TotalIndices, OneNodeDevIndices)

            # set the new list
            NewList         = np.copy(List[ResultIndices])
        else:
            NewList         = List
        return NewList

    # # **************************************************************************
    # def _GenerateMemLists(self, DevList, PercentMC, MRMCFlag=False):
    #     # **********************************************************************
    #     # reset the variable
    #     # **********************************************************************
    #     NanoWireList    = None
    #     MResList        = None
    #     MCapList        = None
    #
    #     # **********************************************************************
    #     # get total device in the connection list
    #     # **********************************************************************
    #     TotalDevs       = len(DevList)
    #
    #     # **********************************************************************
    #     # calculate the number of mem-devices
    #     # **********************************************************************
    #     self.NumMCap    = int ((PercentMC * TotalDevs ) / 100.0)
    #
    #     # **********************************************************************
    #     # check the flag for combining mem-device (memR and memC)
    #     # **********************************************************************
    #     if MRMCFlag:
    #         self.NumMRes    = self.NumMCap
    #     else:
    #         self.NumMRes    = TotalDevs - self.NumMCap
    #
    #     # **********************************************************************
    #     # check the nano wire flag
    #     # **********************************************************************
    #     if self.NanoWireFlag:
    #         NanoWireList    = self.Rwires
    #
    #     # **********************************************************************
    #     # check the number of devices
    #     # **********************************************************************
    #     if self.NumMCap == 0:
    #         MCapList = None
    #         MResList = ConnectionList
    #
    #         # ******************************************************************
    #         # check the nano wire flag
    #         # ******************************************************************
    #         if self.NanoWireFlag:
    #             self.RRSeries   = NanoWireList
    #
    #     elif self.NumMRes == 0:
    #         MResList = None
    #         MCapList = ConnectionList
    #
    #         # ******************************************************************
    #         # check the nano wire flag
    #         # ******************************************************************
    #         if self.NanoWireFlag:
    #             # save the RC series
    #             self.RCSeries   = NanoWireList
    #
    #     else:
    #         # ******************************************************************
    #         # check the flag
    #         # ******************************************************************
    #         if MRMCFlag:
    #             MResList    = np.copy(ConnectionList)
    #             MCapList    = np.copy(ConnectionList)
    #         else:
    #             # slice the list
    #             MResList    = ConnectionList[0:self.NumMRes,:]
    #             MCapList    = ConnectionList[self.NumMRes:,:]
    #
    #         # ******************************************************************
    #         # check the nano wire flag
    #         # ******************************************************************
    #         if self.NanoWireFlag:
    #             self.RRSeries  = NanoWireList[0:self.NumMRes,:]
    #             self.RCSeries  = NanoWireList[self.NumMRes:,:]
    #
    #     return MResList, MCapList

    # **************************************************************************
    def _GenerateMemLists(self, DevList, PercentMC, MRMCFlag=False):
        # **********************************************************************
        # reset the variable
        # **********************************************************************
        MResList        = None
        MCapList        = None

        # **********************************************************************
        # get total device in the connection list
        # **********************************************************************
        TotalDevs       = len(DevList)

        # **********************************************************************
        # check the percentage of MC
        # **********************************************************************
        self.MRList     = None
        self.NumMRes    = 0
        self.MCList     = None

        if PercentMC == 0.0:
            self.MRList  = DevList
            self.NumMRes = TotalDevs

        elif PercentMC == 100.0:
            self.MCList  = DevList
            self.NumMCap = TotalDevs
        else:
            # ******************************************************************
            # calculate the number of mem-devices
            # ******************************************************************
            self.NumMCap = int ((PercentMC * TotalDevs ) / 100.0)
            self.NumMRes = TotalDevs - self.NumMCap

            # ******************************************************************
            # indices of device list
            # ******************************************************************
            DevIndices  = np.arange(TotalDevs)
            MCIndices   = self._SelectRandomNodes(DevIndices, self.NumMCap)
            MRIndices   = np.setdiff1d(DevIndices, MCIndices)

            # ******************************************************************
            # set the MR and MC lists
            # ******************************************************************
            self.MRList = DevList[MRIndices]
            self.MCList = DevList[MCIndices]

        # **********************************************************************
        # check for MRMC device
        # **********************************************************************
        if MRMCFlag:
            self.MRList  = self.MCList
            self.NumMRes = self.NumMCap

    # **************************************************************************
    def _SetRGndList(self, NodeList):
        # **********************************************************************
        # set the function name
        # **********************************************************************
        FunctionName = "NetList::_SetRGndList()"

        # **********************************************************************
        # display the message
        # **********************************************************************
        if self.Verbose:
            # ******************************************************************
            # format the message
            # ******************************************************************
            Msg = "...%-25s: Adding resistors for <%s> = <%d>..." % (FunctionName, \
                    str(NodeList), len(NodeList))
            print(Msg)

        # **********************************************************************
        # get the resistor list
        # **********************************************************************
        RGndList = self._CreatRList(NodeList)

        # **********************************************************************
        # check the current resistor list
        # **********************************************************************
        if self.RList is None:
            # ******************************************************************
            # set the resistor list
            # ******************************************************************
            self.RList = RGndList
        else:
            # ******************************************************************
            # add to the resistor list
            # ******************************************************************
            self.RList = np.concatenate((self.RList, RGndList), axis=0)

    # **************************************************************************
    def _CreateNodeList(self, NetList, VsList=None):
        # **********************************************************************
        # set the function name
        # **********************************************************************
        FunctionName = "NetList::_CreateNodeList()"

        # **********************************************************************
        # display the message
        # **********************************************************************
        if self.Verbose:
            # ******************************************************************
            # display the information
            # ******************************************************************
            Msg = "...%-25s: node list connections ..." % (FunctionName)
            print(Msg)

        # **********************************************************************
        # number of nodes
        # **********************************************************************
        Nodes = self.NumNodes + 1

        # **********************************************************************
        # reset the check list
        # **********************************************************************
        CheckList = np.zeros((Nodes, 2), dtype=int)

        # **********************************************************************
        # filing in node number
        # **********************************************************************
        CheckList[:,0] = np.arange(0, Nodes)

        # **********************************************************************
        # set the check list
        # **********************************************************************
        for row in NetList:
            a = int(row[0])
            b = int(row[1])
            CheckList[a,1] += 1
            CheckList[b,1] += 1

        # ******************************************************************
        # check the input signal connection list
        # ******************************************************************
        if VsList is not None:
            # **************************************************************
            # set the check list
            # **************************************************************
            for row in VsList:
                a = int(row[0])
                b = int(row[1])
                CheckList[a,1] += 1
                CheckList[b,1] += 1

        return CheckList

    # **************************************************************************
    def _CheckNodeList(self, NetList, VsList=None, NumCon=-1):
        # **********************************************************************
        # set the function name
        # **********************************************************************
        FunctionName = "NetList::_CheckNodeList()"

        # **********************************************************************
        # display the message
        # **********************************************************************
        if self.Verbose:
            # ******************************************************************
            # display the information
            # ******************************************************************
            if NumCon == 0:
                Note = "invalid nodes"
            elif NumCon == 1:
                Note = "dangling nodes"
            else:
                Note = "%d" % NumCon
            Msg = "...%-25s: checking node connections = <%s>..." % (FunctionName, \
                    Note)
            print(Msg)

        # **********************************************************************
        # make copy of the check list
        # **********************************************************************
        CheckList   = np.copy(self.NumConAtEachNodeList)

        # **********************************************************************
        # check for dangling nodes
        # **********************************************************************
        InvalidNodes = np.argwhere(CheckList[:,1] == NumCon).ravel().astype(int)

        # **********************************************************************
        # check the dangling nodes
        # **********************************************************************
        if InvalidNodes.size == 0:
            # reset the node list
            InvalidNodes = None

        return InvalidNodes

    # **************************************************************************
    def _SetVsRs(self):
        # create the voltage source resistance list
        self.RsVs       = np.copy(self.VsList)
        self.RsVs[:,2]  = self.RsVsVal
        self.NumRsNodes = self.InNodes

        # internal node list
        RsNodes = np.arange(self.InNodes) + self.NumNodes

        # adjust the connecting nodes
        self.RsVs[:,0]   = RsNodes
        self.RsVs[:,1]   = self.VsList[:,0]
        self.VsList[:,0] = RsNodes

        # adjust the input nodes
        self.InputNodes  = RsNodes.astype(int)

    # **************************************************************************
    def _SelectRandomNodes(self, NodeList, NumNodes):
        # **********************************************************************
        # set the function name
        # **********************************************************************
        FunctionName = "NetList::_SelectRandomNodes()"

        # **********************************************************************
        # display the message
        # **********************************************************************
        if self.Verbose:
            # display the information
            Msg = "...%-25s: selecting random nodes from <%d nodes>..." % (FunctionName, \
                    NumNodes)
            print(Msg)

        # **********************************************************************
        # make a copy of the array list
        # **********************************************************************
        RemainNodes = np.copy(NodeList)

        # **********************************************************************
        # check the lists
        # **********************************************************************
        if NumNodes > len(RemainNodes):
            # set the error message
            ErrMsg = "%s: num nodes = <%d> > available nodes = <%d>" % (FunctionName, \
                    NumNodes, len(RemainNodes))
            raise ValueError(ErrMsg)

        # **********************************************************************
        # select input nodes
        # **********************************************************************
        SelectNodeList  = np.random.choice(NodeList, size=NumNodes, replace=False)
        return SelectNodeList

    # # **************************************************************************
    # def _CreatMRList(self, ConnectionList=None):
    #     # **********************************************************************
    #     # set the function name
    #     # **********************************************************************
    #     FunctionName = "NetList::_CreatMRList()"
    #
    #     # **********************************************************************
    #     # check the list
    #     # **********************************************************************
    #     if ConnectionList is None:
    #         return ConnectionList
    #
    #     # **********************************************************************
    #     # display the message
    #     # **********************************************************************
    #     if self.Verbose:
    #         # display the information
    #         Msg = "...%-25s: creating memristive NetList <MResDev = %d>..." % \
    #                 (FunctionName, len(ConnectionList))
    #         print(Msg)
    #
    #     # **********************************************************************
    #     # Number of memcapacitive device
    #     # **********************************************************************
    #     NumDev = len(ConnectionList)
    #
    #     """
    #     # * ********************************************************************
    #     # * 0, 1, 2 , 3
    #     # **********************************************************************
    #     # * a, b, Rw, Distance
    #     self.ConnectionList
    #
    #     # **********************************************************************
    #     # set the memristive device list
    #     # **********************************************************************
    #     #      0     ,    1      , 2,  3 , 4
    #     # **********************************************************************
    #     #   plus node, minus node, R,  Rw, d
    #     # **********************************************************************
    #     """
    #     Fields = 5
    #     MRList = np.zeros((NumDev, Fields))
    #
    #     # **********************************************************************
    #     # set the memristive list
    #     # **********************************************************************
    #     MRList[:,:2]    = ConnectionList[:,:2]
    #     MRList[:,3:]    = ConnectionList[:,2:4]
    #
    #     # **********************************************************************
    #     # check for selecting random value
    #     # **********************************************************************
    #     if self.MRInits is None:
    #         # self.MRInits = np.round(np.random.random(NumDev), self.RoundOffDec)
    #         InitStateVals = np.around(np.random.uniform(low=self.InitLo, \
    #                 high=self.InitHi, size=NumDev), decimals=self.RoundOffDec)
    #         self.MRInits = torch.from_numpy(InitStateVals)
    #
    #     # **********************************************************************
    #     # initial resistance
    #     # **********************************************************************
    #     InitRho = self.MemResObj.GetInitRho(self.MRInits)
    #     MRList[:,2] = self.MemResObj.GetInitRes(InitRho).Datasets.numpy()
    #
    #     return MRList
    #
    # # **************************************************************************
    # def _CreatMCList(self, ConnectionList=None):
    #     # **********************************************************************
    #     # set the function name
    #     # **********************************************************************
    #     FunctionName = "NetList::_CreatMCList()"
    #
    #     # **********************************************************************
    #     # check the list
    #     # **********************************************************************
    #     if ConnectionList is None:
    #         # Msg = "...%-25s: ConnectionList is empty..." % (FunctionName)
    #         # print(Msg)
    #         return ConnectionList
    #
    #     # **********************************************************************
    #     # display the message
    #     # **********************************************************************
    #     if self.Verbose:
    #         # display the information
    #         Msg = "...%-25s: creating memcapacitive NetList <MCapDev = %d>..." % \
    #                 (FunctionName, len(ConnectionList))
    #         print(Msg)
    #
    #     # **********************************************************************
    #     # Number of memcapacitive device
    #     # **********************************************************************
    #     NumDev = len(ConnectionList)
    #     """
    #     # * ********************************************************************
    #     # * 0, 1, 2 , 3
    #     # **********************************************************************
    #     # * a, b, Rw, Distance
    #     self.ConnectionList
    #
    #     # **********************************************************************
    #     # set the memcapacitive device list
    #     # **********************************************************************
    #     #      0     ,    1      , 2, 3 , 4 , 5, 6, 7
    #     # **********************************************************************
    #     #   plus node, minus node, C, dC, Rw, d, x, m
    #     # **********************************************************************
    #     """
    #     Fields = 8
    #     MCList = np.zeros((NumDev, Fields), dtype=float)
    #
    #     # **********************************************************************
    #     # set the memcapacitive list
    #     # **********************************************************************
    #     MCList[:,:2]    = ConnectionList[:,:2]
    #     MCList[:,4:6]   = ConnectionList[:,2:4]
    #
    #     # **********************************************************************
    #     # check for selecting random value
    #     # **********************************************************************
    #     if self.MCInits is None:
    #         # get random initial states
    #         # self.MCInits = np.round(np.random.random(NumDev), self.RoundOffDec)
    #         InitStateVals = np.around(np.random.uniform(low=self.InitLo, \
    #                 high=self.InitHi, size=NumDev), decimals=self.RoundOffDec)
    #         self.MCInits = torch.from_numpy(InitStateVals)
    #
    #     # **********************************************************************
    #     # initial capacitances
    #     # **********************************************************************
    #     MCList[:,2] = self.MemCapObj.GetInitRho(self.MCInits).Datasets.numpy()
    #
    #     # **********************************************************************
    #     # check the device model for x and m values
    #     # **********************************************************************
    #     if self.MemCapObj == "Mohamed":
    #         TensorX, TensorM    = self.MemCapObj.GetXandMInits()
    #         MCList[:,6] = TensorX.Datasets.numpy()
    #         MCList[:,7] = TensorM.Datasets.numpy()
    #
    #         # release memory
    #         del TensorX
    #         del TensorM
    #
    #     elif self.MemCapObj == "Najem":
    #         TensorR, TensorW    = self.MemCapObj.GetRAndWInits()
    #         MCList[:,6] = TensorR.Datasets.numpy()
    #         MCList[:,7] = TensorW.Datasets.numpy()
    #
    #         # release memory
    #         del TensorR
    #         del TensorW
    #
    #     return MCList
    #
    # # **************************************************************************
    # def _CreatRLeakageList(self, MCList=None):
    #     # **********************************************************************
    #     # set the function name
    #     # **********************************************************************
    #     FunctionName = "NetList::_CreatRLeakageList()"
    #
    #     # **********************************************************************
    #     # display the message
    #     # **********************************************************************
    #     if self.Verbose:
    #         # display the information
    #         Msg = "...%-25s: creating leakage R list <RLeakageList = %d>..." % \
    #                 (FunctionName, len(MCList))
    #         print(Msg)
    #
    #     # **********************************************************************
    #     # make copy of the new list
    #     # **********************************************************************
    #     MRLeakageList       = np.copy(MCList[:,:3])
    #     MRLeakageList[:,2]  = self.RLeakage
    #
    #     return MRLeakageList

    # **************************************************************************
    def _CreatVsList(self, InputNodes=None, VsInits=None):
        # **********************************************************************
        # set the function name
        # **********************************************************************
        FunctionName = "NetList::_CreatVsList()"

        # **********************************************************************
        # display the message
        # **********************************************************************
        if self.Verbose:
            # display the information
            Msg = "...%-25s: creating Vs NetList <Vs = %d>..." % \
                    (FunctionName, len(InputNodes))
            print(Msg)

        # **********************************************************************
        # Number of input sources
        # **********************************************************************
        NumVs = self.InNodes

        # **********************************************************************
        # set the memcapacitive device list
        # **********************************************************************
        Fields = 3
        VsList = np.zeros((NumVs, Fields))

        # **********************************************************************
        # set the Vs list
        # **********************************************************************
        VsList[:,0] = InputNodes

        # **********************************************************************
        # set the initial values if there are any
        # **********************************************************************
        if VsInits is not None:
            VsList[:,2] = VsInits

        return VsList

    # **************************************************************************
    def _CreatRList(self, Nodes, RVal=None):
        # **********************************************************************
        # set the function name
        # **********************************************************************
        FunctionName = "NetList::_CreatRList()"

        # **********************************************************************
        # display the message
        # **********************************************************************
        if self.Verbose:
            # display the information
            Msg = "...%-25s: creating resistor NetList <ResDev = %d>..." % \
                    (FunctionName, len(Nodes))
            print(Msg)

        # **********************************************************************
        # reset the RList
        # **********************************************************************
        RList = np.zeros((Nodes.size, 3))

        # **********************************************************************
        # set the node
        # **********************************************************************
        RList[:,0] = Nodes

        # **********************************************************************
        # check the value of resistance
        # **********************************************************************
        if RVal is None:
            RList[:,2] = self.GndResistor
        else:
            RList[:,2] = RVal
        return RList

    # # **************************************************************************
    # def _CalTotalPower(self):
    #     # get the input nodes
    #     InputNodes = self.InputNodes
    #
    #     # get the number of nodes, including the ground node
    #     Nodes   = self.TotalNumNodes + 1
    #
    #     # get voltage sources and current measurements from tensors
    #     # Vs  = (self.VNodes[:,InputNodes].Datasets).cpu().numpy()
    #     # Is  = (self.VNodes[:,Nodes:Nodes+self.InNodes].Datasets).cpu().numpy()
    #     Vs  = self.VNodes[:,InputNodes]
    #     Is  = self.VNodes[:,Nodes:Nodes+self.InNodes]
    #
    #     # **********************************************************************
    #     # calculate the average power of the sources:
    #     #       Pavg    = 1 / (Length * dt) * int_0^(Length*dt)(Vi * Ii) dt
    #     # **********************************************************************
    #     PInst       = torch.mul(Vs, Is)
    #     TotalPInst  = self._CalInstPower(PInst, self.dt)
    #     PTotal      = np.absolute(TotalPInst)
    #
    #     # **********************************************************************
    #     # calculate the power consumption of nano wire
    #     # **********************************************************************
    #     if self.NanoWireFlag:
    #         # print("...calculating nanowire power...")
    #         # calculate average power of RRs
    #         PnwRR, VSqOverR = self._CalRRPower()
    #
    #         # calculate average power of RCs
    #         PnwRC   = self._CalRCPower(VSqOverR=VSqOverR)
    #
    #         #  the total power dissipated in nanowires
    #         self.Pnw = PnwRR + PnwRC
    #
    #     return PTotal + self.Pnw
    #
    # # **************************************************************************
    # def AveragePower(self):
    #     # **********************************************************************
    #     # check the flag for calculating memristive power
    #     # **********************************************************************
    #     if self.PowerMRFlag:
    #         self.MemResPower    = self._CalMemResPower()
    #
    #     # **********************************************************************
    #     # return total power
    #     # **********************************************************************
    #     return self._CalTotalPower()
    #
    # # **************************************************************************
    # def ResetPowerCalculation(self):
    #     # reset the variables for power calculations
    #     self.TotalMemResPower   = 0.0
    #     self.TotalMemCapPower   = 0.0
    #     self.TotalRun           = 0.0
    #     self.Pnw                = 0.0
    #     self.MemResPower        = 0.0
    #
    # # **************************************************************************
    # def SetSelectInputs(self, SelectInputs):
    #     self.SelectInputs   = SelectInputs + self.TotalNumNodes
    #     self.NumSelInputs   = len(self.SelectInputs)
    #
    # # **************************************************************************
    # def SetInitMRAndInitMC(self, MRInits=None, MCInits=None):
    #     # **********************************************************************
    #     # check the initial setting
    #     # **********************************************************************
    #     if MRInits is not None:
    #         # ******************************************************************
    #         # save the initial values
    #         # ******************************************************************
    #         self.MRInits = MRInits
    #
    #         # ******************************************************************
    #         # reset the NetList matrices
    #         # ******************************************************************
    #         self._SetMemResMatrix()
    #
    #     # **********************************************************************
    #     # check the initial setting
    #     # **********************************************************************
    #     if MCInits is not None:
    #         # ******************************************************************
    #         # save the initial values
    #         # ******************************************************************
    #         self.MCInits = MCInits
    #
    #         # ******************************************************************
    #         # reset the NetList matrices
    #         # ******************************************************************
    #         self._SetMemCapMatrix()
    #
    # # **************************************************************************
    # def ResetNetList(self):
    #     # set the function name
    #     FunctionName = "NetList::ResetNetList()"
    #
    #     # display the message
    #     if self.Verbose:
    #         # display the information
    #         Msg = "...%-25s: reset NetList states..." % (FunctionName)
    #         print(Msg)
    #
    #     """
    #     # **********************************************************************
    #     # parameters for memristive matrix
    #     # **********************************************************************
    #     #      0     ,    1      , 2,  3 ,  4   ,  5  , 6 , 7
    #     # **********************************************************************
    #     #   plus node, minus node, R, Rho, PrevR, dAij, Rw, PreVRes
    #     # **********************************************************************
    #     """
    #     if self.MRFlag:
    #         # reset the resistive matrix
    #         self.MemResMatrix[:,2:] = 0.0
    #
    #         # check the initial state
    #         if self.GPU_Flag:
    #             self.MRInits = self.MRInits.to(self.Device)
    #
    #         # get the initial values of rho
    #         InitRho = self.MemResObj.GetInitRho(self.MRInits)
    #
    #         # set the initial resistance
    #         self.MemResMatrix[:,2] = torch.from_numpy(self.MRList[:,2])
    #
    #         # save the internal state Rho
    #         self.MemResMatrix[:,3] = InitRho
    #
    #     """
    #     # *************************************************************************
    #     # parameters for memcapacitive matrix
    #     # *************************************************************************
    #     #      0     ,    1      , 2, 3 ,    4    ,  5 ,  6 , 7 ,    8 ,  9  , 10, 11
    #     # *************************************************************************
    #     #   plus node, minus node, C, dC, PrevVcaps, In, x|R, m|W, PrevC, dAij, Rw, G
    #     # **************************************************************************
    #     """
    #     if self.MCFlag:
    #         # reset capacitive matrix
    #         self.MemCapMatrix[:]   = 0.0
    #
    #         # set the initial capacitive values
    #         self.MemCapMatrix[:,2] = torch.from_numpy(self.MCList[:,3])
    #
    #         # save the previous capacitance values
    #         self.MemCapMatrix[:,8] = self.MemCapMatrix[:,2] / self.dt
    #
    #         # reset x and m for Mohamed model
    #         if self.MemCapName == "Mohamed":
    #             self.SetInitValsForXAndM(self.MemCapObj, self.MemCapMatrix, \
    #                     self.MCInits)
    #         elif self.MemCapName == "Najem":
    #             self.SetInitValsForRAndW(self.MemCapObj, self.MemCapMatrix, \
    #                     self.MCInits)
    #
    #     # **********************************************************************
    #     # construct Matrix A and Z for NetList
    #     # **********************************************************************
    #     SetupMatrixDict = self.MatricesAandZ(self.ParMatrixAAndZDict)
    #
    #     # **********************************************************************
    #     # extract setup matrices: MatrixA, MatrixZ, and MCZCoeff are torch.tensor
    #     #       of double type
    #     # **********************************************************************
    #     MatrixA         = SetupMatrixDict["MatrixA"]
    #     MatrixZ         = SetupMatrixDict["MatrixZ"]
    #     CZCoeff         = SetupMatrixDict["CZCoeff"]
    #     MCZCoeff        = SetupMatrixDict["MCZCoeff"]
    #     self.MatrixA[:] = MatrixA
    #     self.MatrixZ[:] = MatrixZ.reshape(len(MatrixZ), 1)
    #     self.MCZCoeff[:] = MCZCoeff
    #     NumVs           = SetupMatrixDict["NumVs"]
    #
    #     # **********************************************************************
    #     # set the matrix Z
    #     # **********************************************************************
    #     # self.MatrixZ[self.TotalNumNodes:self.VsIndices] = self.VsList[:,2]
    #     self.MatrixZ[self.TotalNumNodes:self.VsIndices] = 0.0
    #
    #     # **********************************************************************
    #     # power calculations for memristive and memcapacitive devices
    #     # **********************************************************************
    #     self.TotalMemResPower   = 0.0
    #     self.TotalMemCapPower   = 0.0
    #     self.TotalRun           = 0.0
    #     self.NumTs              = 1.0
    #     self.Pnw                = 0.0
    #     self.MemResPower        = 0.0
    #
    # # **************************************************************************
    # def ResetNetListRandomStates(self):
    #     # set the function name
    #     FunctionName = "NetList::ResetNetList()"
    #
    #     # display the message
    #     if self.Verbose:
    #         # display the information
    #         Msg = "...%-25s: reset NetList states..." % (FunctionName)
    #         print(Msg)
    #
    #     """
    #     # **********************************************************************
    #     # parameters for memristive matrix
    #     # **********************************************************************
    #     #      0     ,    1      , 2,  3 ,  4   ,  5  , 6 , 7
    #     # **********************************************************************
    #     #   plus node, minus node, R, Rho, PrevR, dAij, Rw, PreVRes
    #     # **********************************************************************
    #     """
    #     if self.MRFlag:
    #         # reset the resistive matrix
    #         self.MemResMatrix[:,2:] = 0.0
    #
    #         # selecting random initial states
    #         self.MRInits = np.around(np.random.uniform(low=self.InitLo, \
    #                 high=self.InitHi, size=len(self.MRList)), decimals=self.RoundOffDec)
    #
    #         # initial resistance
    #         InitRho     = self.MemResObj.GetInitRho(self.MRInits)
    #         MRList[:,2] = self.MemResObj.GetInitRes(InitRho).Datasets.numpy()
    #
    #         # set the initial resistance
    #         self.MemResMatrix[:,2] = torch.from_numpy(self.MRList[:,2])
    #
    #         # save the internal state Rho
    #         self.MemResMatrix[:,3] = InitRho
    #
    #     """
    #     # *************************************************************************
    #     # parameters for memcapacitive matrix
    #     # *************************************************************************
    #     #      0     ,    1      , 2, 3 ,    4    ,  5 ,  6 , 7 ,    8 ,  9  , 10, 11
    #     # *************************************************************************
    #     #   plus node, minus node, C, dC, PrevVcaps, In, x|R, m|W, PrevC, dAij, Rw, G
    #     # **************************************************************************
    #     """
    #     if self.MCFlag:
    #         # selecting random initial states
    #         self.MCInits = np.around(np.random.uniform(low=self.InitLo, \
    #                 high=self.InitHi, size=len(self.MCList)), decimals=self.RoundOffDec)
    #
    #         # reset capacitive matrix
    #         self.MemCapMatrix[:]   = 0.0
    #
    #         # set the initial capacitive values
    #         InitCVals           = self.MemCapObj.RandomInitCVals(self.MCInits)
    #         self.MemCapMatrix[:,2] = InitCVals
    #         self.MCList[:,3]    = InitCVals.Datasets.cpu().numpy()
    #
    #         # save the previous capacitance values
    #         self.MemCapMatrix[:,8] = self.MemCapMatrix[:,2] / self.dt
    #
    #         # reset x and m for Mohamed model
    #         if self.MemCapName == "Mohamed":
    #             self.SetInitValsForXAndM(self.MemCapObj, self.MemCapMatrix, \
    #                     self.MCInits)
    #         elif self.MemCapName == "Najem":
    #             self.SetInitValsForRAndW(self.MemCapObj, self.MemCapMatrix, \
    #                     self.MCInits, RandomFlag=True)
    #
    #     # **********************************************************************
    #     # construct Matrix A and Z for NetList
    #     # **********************************************************************
    #     SetupMatrixDict = self.MatricesAandZ(self.ParMatrixAAndZDict)
    #
    #     # **********************************************************************
    #     # extract setup matrices: MatrixA, MatrixZ, and MCZCoeff are torch.tensor
    #     #       of double type
    #     # **********************************************************************
    #     MatrixA         = SetupMatrixDict["MatrixA"]
    #     MatrixZ         = SetupMatrixDict["MatrixZ"]
    #     MCZCoeff          = SetupMatrixDict["MCZCoeff"]
    #     self.MatrixA[:] = MatrixA
    #     self.MatrixZ[:] = MatrixZ.reshape(len(MatrixZ), 1)
    #     self.MCZCoeff[:] = MCZCoeff
    #     NumVs           = SetupMatrixDict["NumVs"]
    #
    #     # **********************************************************************
    #     # set the matrix Z
    #     # **********************************************************************
    #     # self.MatrixZ[self.TotalNumNodes:self.VsIndices] = self.VsList[:,2]
    #     self.MatrixZ[self.TotalNumNodes:self.VsIndices] = 0.0
    #
    #     # **********************************************************************
    #     # power calculations for memristive and memcapacitive devices
    #     # **********************************************************************
    #     self.TotalMemResPower   = 0.0
    #     self.TotalMemCapPower   = 0.0
    #     self.TotalRun           = 0.0
    #     self.NumTs              = 1.0
    #     self.Pnw                = 0.0
    #     self.MemResPower        = 0.0
    #
    # # **************************************************************************
    # def ResetVerboseFlag(self):
    #     # reset verbose flag
    #     self.Verbose = False
    #
    #     # reset the verbose flag of device models
    #     if self.MemResObj is not None:
    #         self.MemResObj.ResetVerboseFlag()
    #
    #     # reset the verbose flag of device models
    #     if self.MemCapObj is not None:
    #         self.MemCapObj.ResetVerboseFlag()
    #
    # # **************************************************************************
    # def SetVerboseFlag(self):
    #     # set verbose flag
    #     self.Verbose = True
    #
    # # **************************************************************************
    # def GetRemainNodes(self):
    #     return self.AvaiOutNodes
    #
    # # **************************************************************************
    # def GetVsList(self):
    #     return self.VsList
    #
    # # **************************************************************************
    # def GetInitMRAndInitMC(self):
    #     return self.MRInits, self.MCInits
    #
    # # **************************************************************************
    # def GetNetListNodeVoltages(self):
    #     return self.VNodes
    #
    # # **************************************************************************
    # def GetDeviceList(self):
    #     return self.DevList
    #
    # # **************************************************************************
    # def GetOutputNodes(self):
    #     if not torch.is_tensor(self.OutputNodes):
    #         self.OutputNodes = torch.from_numpy(self.OutputNodes.astype("int64"))
    #     return self.OutputNodes

    def GetNetParams(self):
        # **********************************************************************
        # convert to tensor
        # **********************************************************************
        if self.VsList is not None:
            self.VsList = torch.from_numpy(self.VsList)
        if self.IsList is not None:
            self.IsList = torch.from_numpy(self.IsList)
        if self.RList is not None:
            self.RList = torch.from_numpy(self.RList)
        if self.CList is not None:
            self.CList = torch.from_numpy(self.CList)
        if self.MRList is not None:
            self.MRList = torch.from_numpy(self.MRList[:,:3])
            self.MRList[:,2] = 1.0
        if self.MCList is not None:
            self.MCList = torch.from_numpy(self.MCList[:,:3])
            self.MCList[:,2] = 1.0
        if self.NodeList is not None:
            self.NodeList = torch.from_numpy(self.NodeList)
        if self.InputNodes is not None:
            self.InputNodes = torch.from_numpy(self.InputNodes)
        if self.AvaiOutNodes is not None:
            self.AvaiOutNodes = torch.from_numpy(self.AvaiOutNodes)

        # **********************************************************************
        # set the return dictionary
        # **********************************************************************
        NetParams   = { "VsList": self.VsList, "IsList": self.IsList, "RList": self.RList,\
            "CList": self.CList, "MRList": self.MRList, "MCList": self.MCList,\
            "NodeList": self.NodeList, "NumNodes": self.NumNodes, "InputNodes": self.InputNodes,
            "AvaiOutNodes": self.AvaiOutNodes
        }


        # self.NumNodes   = NetParms["NumNodes"]
        # self.dt         = NetParms["dt"]
        # self.MemResObj  = NetParms["MemResObj"]
        # self.MemCapObj  = NetParms["MemCapObj"]
        # self.Verbose    = Verbose
        #
        #
        # self.InputNodes = None
        # self.AvaiOutNodes = None
        return NetParams

# ******************************************************************************
if __name__ == "__main__":
    # **************************************************************************
    # import module
    # **************************************************************************
    import SmallWorldPowerLaw
    from MathUtils import FindFactors
    import pandas as pd
    import time

    # **************************************************************************
    # a temporary fix for OpenMP
    # **************************************************************************
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    # **************************************************************************
    # set the waring to generate exception on
    # **************************************************************************
    import warnings
    warnings.simplefilter("error", category=RuntimeWarning)
    warnings.simplefilter("error", category=ArithmeticError)

    # **************************************************************************
    # paramters for input signals
    # **************************************************************************
    start_time  = time.time()
    e = int(time.time() - start_time)
    print("Start time: {:02d}:{:02d}:{:02d}".format(e // 3600, (e % 3600 // 60), e % 60))

    # **************************************************************************
    # check for GPU
    # **************************************************************************
    GPU_Flag    = torch.cuda.is_available()
    Device      = torch.device("cuda:0" if GPU_Flag else "cpu")

    # **************************************************************************
    # parameters for small-world power-law graph
    # **************************************************************************
    # initial starting graph
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
    # Create SW graph
    # **************************************************************************
    GraphClass = SmallWorldPowerLaw.SmallWorldPowerLaw(InitGraph=InitGraph, L=L, Beta=Beta, \
            Alpha=Alpha, Gamma=Gamma, Delta=Delta, BoundaryFlag=BoundaryFlag, \
            GndNodeFlag=GndNodeFlag, Verbose=Verbose)

    # get the device list
    DevList = GraphClass.GetEdgeList()
    print("DevList  = ", DevList.shape)
    # exit()

    # **************************************************************************
    # create memcapacitive circuit
    # **************************************************************************
    InNodes     = 20
    PerMemC     = 0
    MRMCFlag    = False
    NetParams   = {"DevList": DevList, "InNodes": InNodes, "PerMemC": PerMemC, \
            "MRMCFlag": MRMCFlag, "Verbose": Verbose}

    NetListCircuit = NetList(NetParams)
    if GPU_Flag:
        NetListCircuit = NetListCircuit.to(Device)

    exit()
    # **************************************************************************
    # set the input signal
    # **************************************************************************
    NumStep = 1000
    Vin     = np.random.randn(NumStep, NumInputs)
    Vin     = torch.from_numpy(Vin)
    if GPU_Flag:
        Vin = Vin.to(Device)

    # **************************************************************************
    # set the time
    # **************************************************************************
    dt      = 2.7e-07
    t       = np.arange(0, NumStep * dt, dt)
    t       = torch.from_numpy(t)

    # **************************************************************************
    # set the time
    # **************************************************************************
    SelectIndices   = 20
    SelectInputs    = np.unique(np.random.choice(NumInputs, size=SelectIndices, replace=False))
    SelectInputs    = torch.from_numpy(SelectInputs.astype("int64"))
    if GPU_Flag:
        SelectInputs = SelectInputs.to(Device)
    # print("SelectIndices    = ", SelectIndices.shape)
    # print("SelectIndices    = ", SelectIndices)

    # **************************************************************************
    # set the selected inputs
    # **************************************************************************
    NetListCircuit.SetSelectInputs(SelectInputs)

    # **************************************************************************
    # set the time
    # **************************************************************************
    AppliedInputs   = Vin[:,SelectInputs]
    # print("AppliedInputs    = ", AppliedInputs.shape)
    # print("AppliedInputs    = ", AppliedInputs.dtype)
    # print("AppliedInputs    = ", AppliedInputs.device)
    #
    # print("SelectInputs     = ", SelectInputs.shape)
    # print("SelectInputs     = ", SelectInputs.dtype)
    # print("SelectInputs     = ", SelectInputs.device)
    # exit()

    # **************************************************************************
    # get the node voltages
    # **************************************************************************
    print("... Calculating Node Voltages ...")
    NetListCircuit.SetSelectInputs(SelectInputs)
    VNodeNetList    = NetListCircuit(t, AppliedInputs, SelectIndices=SelectIndices,\
            Verbose=True)
    print("AppliedInputs    = ", AppliedInputs.shape)
    print("VNodeNetList         = ", VNodeNetList.shape)

    # **************************************************************************
    # end time
    # **************************************************************************
    e = int(time.time() - start_time)
    print("\nElapsed time = {:02d}:{:02d}:{:02d}".format(e // 3600, (e % 3600 // 60), e % 60))
