# ******************************************************************************
# import modules
# ******************************************************************************
from scipy import signal
from os.path import join
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import sys

# ******************************************************************************
# Functions in this module
# ******************************************************************************
def _ProcessArguments():
    # process the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--Netlist", type=str, required=True,
            help = "Netlist file for circuit.")

    # # MemCapacitive model
    # ap.add_argument("-mc", "--McModel", type=str, required=True,
    #         help = "Memcapacitive model.")
    #
    # # memristive model
    # ap.add_argument("-mr", "--MrModel", type=str, required=True,
    #         help = "Memristive model.")
    #
    # # add argument for graph name
    # ap.add_argument("-app", "--AppType", type=str, required=True,
    #         help = "Application type < SpokenDigits | Mnist | Cifar10 | Timit >.")
    #
    # # add argument for one device option
    # ap.add_argument("-per", "--PerMemC", type=float, default=None,
    #         help = "either 0 or 100.")
    #
    # # add argument for number of training images
    # ap.add_argument("-ntr", "--NumTrains", type=int, default=None,
    #         help = "Number of training images. Default is None.")
    #
    # # add argument for number of testing images
    # ap.add_argument("-nts", "--NumTests", type=int, default=None,
    #         help = "Number of testing images. Default is None.")
    #
    # # add argument for a runtime in minutes
    # ap.add_argument("-tm", "--RunTimeM", type=float, default=0.0,
    #         help = "Run time in minutes.")
    #
    # # add argument for a runtime in hours
    # ap.add_argument("-th", "--RunTimeH", type=float, default=0.0,
    #         help = "Run time in hours.")
    #
    # # add argument for a runtime in days
    # ap.add_argument("-td", "--RunTimeD", type=float, default=0.0,
    #         help = "Run time in days.")
    #
    # # add argument for simulation option
    # ap.add_argument("-dtLo", "--dtLo", type=float, default=None,
    #         help = "Low limit of the timestep.")
    #
    # # add argument for simulation option
    # ap.add_argument("-dtHi", "--dtHi", type=float, default=None,
    #         help = "High limit of the timestep.")
    #
    # # add argument for simulation option
    # ap.add_argument("-dt", "--TimeStep", type=float, default=None,
    #         help = "Timestep pulse.")
    #
    # # add argument for simulation option
    # ap.add_argument("-AmpLo", "--AmpLo", type=float, default=None,
    #         help = "Low limit of signal amplitude.")
    #
    # # add argument for simulation option
    # ap.add_argument("-AmpHi", "--AmpHi", type=float, default=None,
    #         help = "High limit of signal amplitude.")
    #
    # # add argument for simulation option
    # ap.add_argument("-Pgpu", "--Pgpu", type=int, default=None,
    #         help = "Number of processes per GPU. The default value is none.")
    #
    # # add argument for simulation option
    # ap.add_argument("-nT", "--NumTrials", type=int, default=None,
    #         help = "Number of trials. The default value is none.")
    #
    # # add argument for heade node ip address
    # ap.add_argument("-ip", "--ClusterIp", type=str, default=None,
    #         help = "Cluster node IP address.")
    #
    # # add argument for trial folder name
    # ap.add_argument("-tf", "--TrialFolder", type=str, default=None,
    #         help = "Trial folder.")
    #
    # # add argument for instances
    # ap.add_argument("-nI", "--Instances", type=int, default=None,
    #         help = "Number of instances for average results. Default value is 5.")

    # get the arguments
    args = ap.parse_args()
    return args

# ******************************************************************************
def ReadCompInfo(Netlist):
    ColLbs      = ["DEV", "N1", "N2", "VAL"]
    CompInfo    = pd.read_csv(Netlist, delimiter=' ', header=None, index_col=False,\
            comment='*', skipinitialspace=True)
    CompInfo.columns = ColLbs
    CompInfo["DEV"] = CompInfo["DEV"].str.upper()
    return CompInfo

# ******************************************************************************
def ExtractNetlistSym(DfCompInfo):
    # **************************************************************************
    # get the heading names
    # **************************************************************************
    Header  = list(DfCompInfo.columns)

    # **************************************************************************
    # extract voltage source list
    # **************************************************************************
    VolInd  = np.asarray([ADev[0] == 'V' in ADev for ADev in DfCompInfo["DEV"].values])
    if VolInd.any():
        VList   = DfCompInfo.loc[VolInd].to_numpy()
    else:
        VList   = None

    # **************************************************************************
    # extract current source list
    # **************************************************************************
    CurInd  = np.asarray([ADev[0] == 'I' in ADev for ADev in DfCompInfo["DEV"].values])
    if CurInd.any():
        IList   = DfCompInfo.loc[CurInd].to_numpy()
    else:
        IList  = None

    # **************************************************************************
    # extract resistor list
    # **************************************************************************
    ResInd  = np.asarray([ADev[0] == 'R' for ADev in DfCompInfo["DEV"].values])
    if ResInd.any():
        RList   = DfCompInfo.loc[ResInd].to_numpy()
    else:
        RList  = None

    # **************************************************************************
    # extract capacitor list
    # **************************************************************************
    CapInd  = np.asarray([ADev[0] == 'C' for ADev in DfCompInfo["DEV"].values])
    if CapInd.any():
        CList   = DfCompInfo.loc[CapInd].to_numpy()
    else:
        CList  = None

    # **************************************************************************
    # extract memristor list
    # **************************************************************************
    MResInd = np.asarray([ADev[:2] == "MR" for ADev in DfCompInfo["DEV"].values])
    if MResInd.any():
        MRList  = DfCompInfo.loc[MResInd].to_numpy()
    else:
        MRList  = None

    # **************************************************************************
    # extract memcapacitor list
    # **************************************************************************
    MCapInd = np.asarray([ADev[:2] == "MC" for ADev in DfCompInfo["DEV"].values])
    if MCapInd.any():
        MCList  = DfCompInfo.loc[MCapInd].to_numpy()
    else:
        MCList  = None

    # **************************************************************************
    # extract nodes
    # **************************************************************************
    N1  = DfCompInfo["N1"].values
    N2  = DfCompInfo["N2"].values
    NodeList    = np.unique([N1, N2]).astype(int)
    NumNodes    = len(NodeList) - 1

    # **************************************************************************
    # set the return dictionary
    # **************************************************************************
    return {"Header": Header, "VList": VList, "IList": IList, "RList": RList, "CList": CList,\
            "MRList": MRList, "MCList": MCList, "NodeList": NodeList, "NumNodes": NumNodes,\
            "dt": None}

# ******************************************************************************
def ExtractNetlists(NetParms):
    # **************************************************************************
    # check the list
    # **************************************************************************
    if NetParms["VList"] is not None:
        VList   = torch.from_numpy(NetParms["VList"][:,1:].astype(np.double))
    else:
        VList   = None

    if NetParms["IList"] is not None:
        IList   = torch.from_numpy(NetParms["IList"][:,1:].astype(np.double))
    else:
        IList   = None

    if NetParms["RList"] is not None:
        RList   = torch.from_numpy(NetParms["RList"][:,1:].astype(np.double))
    else:
        RList   = None

    if NetParms["CList"] is not None:
        CList   = torch.from_numpy(NetParms["CList"][:,1:].astype(np.double))
    else:
        CList   = None

    if NetParms["MRList"] is not None:
        MRList   = torch.from_numpy(NetParms["MRList"][:,1:].astype(np.double))
    else:
        MRList   = None

    if NetParms["MCList"] is not None:
        MCList   = torch.from_numpy(NetParms["MCList"][:,1:].astype(np.double))
    else:
        MCList   = None

    if NetParms["MCList"] is not None:
        NodeList   = NetParms["NodeList"]
    else:
        NodeList   = None

    if NetParms["NumNodes"] is not None:
        NumNodes   = NetParms["NumNodes"]
    else:
        NumNodes   = None

    dt = NetParms["dt"]

    AllLists = {"VList": VList, "IList": IList, "RList": RList, "CList": CList,\
            "MRList": MRList, "MCList": MCList, "NodeList": NodeList, "NumNodes": NumNodes,\
            "dt": dt, "MemResObj": NetParms["MemResObj"], "MemCapObj": NetParms["MemCapObj"]}

    return AllLists

# ******************************************************************************
# MNA class to build Matrix A, vect X, and vector Z
# ******************************************************************************
class MNA(nn.Module):
    def __init__(self, NetParms=None, Verbose=False):
        # **********************************************************************
        super(MNA, self).__init__()

        # **********************************************************************
        # set the function name
        # **********************************************************************
        FunctionName = "MNA::__init__()"

        # **********************************************************************
        # Save the network parameters
        # **********************************************************************
        self.VList 	= NetParms["VList"]
        self.IList 	= NetParms["IList"]
        self.RList	= NetParms["RList"]
        self.CList	= NetParms["CList"]
        self.MRList	= NetParms["MRList"]
        self.MCList	= NetParms["MCList"]
        self.NodeList = NetParms["NodeList"]
        self.NumNodes = NetParms["NumNodes"]
        self.dt     = NetParms["dt"]
        self.Verbose = Verbose


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
    # process the arguments
    # **************************************************************************
    ap = argparse.ArgumentParser()

    # Manually set sys.argv to simulate command-line arguments
    sys.argv = ["MNA.py", "-f", "Assets/Figure1.cir"]

    ap.add_argument("-f", "--Netlist", type=str, required=True, help="Netlist file for circuit.")
    args = ap.parse_args()
    Verbose = True

    # **************************************************************************
    # read components from the netlist
    # **************************************************************************
    DfCompInfo  = ReadCompInfo(args.Netlist)
    print(" ")
    print(args.Netlist)
    print(DfCompInfo)

    # **************************************************************************
    # create input signal
    # **************************************************************************
    Type    = "Square"
    # Type    = "Sine"
    Amp     = 2.0
    Freq    = 1
    Offset  = 0
    Cycles  = 2
    v, t    = Signal(Type, Amp, Freq, Offset, NoCycles=Cycles, Verbose=Verbose)

    # **************************************************************************
    # extract the component list
    # **************************************************************************
    DefaultDt   = t[1] - t[0]
    SymFlag     = True
    NetParms    = ExtractNetlistSym(DfCompInfo)
    NetParms["dt"] = DefaultDt
    NetParms["MemResObj"] = None
    NetParms["MemCapObj"] = None
    NetParms    = ExtractNetlists(NetParms)
    # print(NetParms)

    # **************************************************************************
    # Create MNA object and build Matrix A, Vector X, and Vector Z
    # **************************************************************************
    print(" ")

    MNAObj  = MNA(NetParms=NetParms, Verbose=Verbose)
