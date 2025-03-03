# ******************************************************************************
# import modules
# ******************************************************************************
import numpy as np
import torch, os
import torch.nn as nn


class BiolekC4Memcapacitor(nn.Module):
    def __init__(self, InitVals=0.5, Verbose=False):
        # **********************************************************************
        super(BiolekC4Memcapacitor, self).__init__()

        # **********************************************************************
        # set the function name
        # **********************************************************************
        FunctionName = "BiolekC4::__init__()"

        # **********************************************************************
        # memristor model name
        # **********************************************************************
        self.ModelName = "BiolekC4"

        # **********************************************************************
        # save parameters
        # **********************************************************************
        self.InitVals   = InitVals
        self.Verbose    = Verbose

        # **********************************************************************
        # constants for the model
        # **********************************************************************
        self.Beta   = torch.tensor(70e-3, dtype=torch.float64)
        self.Vth    = torch.tensor(3, dtype=torch.float64)
        self.b1     = torch.tensor(10e-3, dtype=torch.float64)
        self.b2     = torch.tensor(1e-6, dtype=torch.float64)
        self.Clow   = torch.tensor(1e-12, dtype=torch.float64)
        self.Chigh  = torch.tensor(100e-12, dtype=torch.float64)
        self.Cinit  = (self.Clow+self.Chigh)/2
        self.x = self.Cinit

        ...

    # **************************************************************************
    def UpdateVals(self, Vc, dt):
        def STP(x, b):
            return 1/(1+np.exp(-x/b))
        def f(Vc):
            # *** Biolek C4 Equation ***
            return self.Beta * (Vc - (0.5 * (np.abs(Vc + self.Vth) - np.abs(Vc - self.Vth))))

            # *** Modified Working Equations ***
            # term1 = self.Beta * (Vc - self.Vth) * STP((Vc - self.Vth), self.b1)
            # term2 = self.Beta * (Vc + self.Vth) * STP((-Vc - self.Vth), self.b1)
            # return term1 - term2

        def W(Vc, x):

            # *** Biolek C4 Equations ***
            term1 = STP(Vc, self.b1) * STP((self.Chigh - x), self.b2)
            term2 = STP(-Vc, self.b1) * STP((x - self.Clow), self.b2)
            return term1 + term2

            # *** Modified Working Equation ***
            # return STP((Vc - self.Vth), self.b1) - STP((-Vc - self.Vth), self.b1)

        dx_dt = f(Vc) * W(Vc, self.x)
        self.x += dx_dt * dt
        ...

    # **************************************************************************
    def GetVals(self):
        return self.x
        ...

# ******************************************************************************
if __name__ == "__main__":
    # import module
    from os.path import join
    import sys
    from progressbar import progressbar
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update({"figure.autolayout": True})

    # **************************************************************************
    # set the function name
    # **************************************************************************
    FunctionName = "_main()_"

    # **************************************************************************
    # set the model name
    # **************************************************************************
    ModelName   = "BiolekC4"

    # **************************************************************************
    # append to the system path
    # **************************************************************************
    import WaveGenerator

    # **************************************************************************
    # set the parameters
    # **************************************************************************
    SignalType  = "Sine"
    Ampl        = 4.0
    Freq        = 50e6
    Offset      = 0
    N           = 1
    NumStep     = 1e3
    Verbose     = True

    # **************************************************************************
    # get the object
    # **************************************************************************
    Input = WaveGenerator.WaveGenerator(SignalType=SignalType, Ampl=Ampl, \
            Offset=Offset, Freq=Freq, Sample=NumStep, NumCycles=N, Verbose=Verbose)

    # **************************************************************************
    # create the sine wave
    # **************************************************************************
    (Vs, ts) = Input.Sine(Ampl, Offset, Freq, NumStep, N)

    # **************************************************************************
    # get the number of time step
    # **************************************************************************
    NumTs   = len(ts)
    Vs  = torch.from_numpy(Vs)
    ts  = torch.from_numpy(ts)
    dt  = ts[1] - ts[0]

    # **************************************************************************
    # initializing charge Q and current
    # **************************************************************************
    C   = torch.zeros(NumTs, dtype=torch.float64)

    # **************************************************************************
    # parameters for the memcapacitor model
    # **************************************************************************
    MemCap = BiolekC4Memcapacitor(Verbose=True)

    # **************************************************************************
    # display the information
    # **************************************************************************
    Msg = "...%-25s: calculate internal state ..." % (FunctionName)
    print(Msg)

    # **************************************************************************
    # calculate internal state
    # **************************************************************************
    C[0]    = MemCap.GetVals()
    for i in progressbar(range(1, NumTs)):
        # **********************************************************************
        # get the delta X or x
        # **********************************************************************
        MemCap.UpdateVals(Vs[i], dt)

        # **********************************************************************
        # save the resistance
        # **********************************************************************
        C[i]    = MemCap.GetVals()

    # **************************************************************************
    # calculate the current
    # **************************************************************************
    Q   = torch.multiply(Vs, C)

    # **************************************************************************
    # set the line width
    # **************************************************************************
    LineWidth = 1.5
    # LineWidth = 2.0
    # LineWidth = 2.5

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
    # plot the results
    # **************************************************************************
    Fig = plt.figure("Q-V Spice")
    plt.grid(linestyle="dotted")
    plt.plot(Vs, Q, "-", color="b", label="I-V", linewidth=LineWidth)
    plt.xlabel("Vin (V)")
    plt.ylabel("Charge (C)")
    plt.axis("tight")
    # legend = plt.legend(loc="best")

    # **************************************************************************
    # set the file name
    # **************************************************************************
    FileNameEps = "%sQV.eps" % ModelName
    FileNameJpg = "%sQV.jpg" % ModelName
    FileName    = FileNameJpg

    # **************************************************************************
    # save the figure
    # **************************************************************************
    print("...Saving figure to file = <%s> ..." % FileName)

    # **************************************************************************
    # save the figure
    # **************************************************************************
    plt.savefig(FileName)

    # **************************************************************************
    # get the figure handle
    # **************************************************************************
    Fig = plt.figure("C-V Spice")
    plt.grid(linestyle="dotted")
    scale = 1e12
    plt.plot(Vs, C*scale, "-", color="b", label="R-V", linewidth=LineWidth)
    plt.xlabel("Vin (V)")
    plt.ylabel(r"Capacitance (pF)")
    plt.axis("tight")
    # legend = plt.legend(loc="best")

    # **************************************************************************
    # set the file name
    # **************************************************************************
    FileNameEps = "%sCV.eps" % ModelName
    FileNameJpg = "%sCV.jpg" % ModelName
    FileName    = FileNameJpg

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
