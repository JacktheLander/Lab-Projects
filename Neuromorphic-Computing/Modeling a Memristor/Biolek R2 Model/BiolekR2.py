# ******************************************************************************
# import modules
# ******************************************************************************
import numpy as np
import torch, os
import torch.nn as nn

# ******************************************************************************
"""
Ideal Memristor Model R2
* D. Biolek, M. Di Ventra, and Y. V. Pershin, "Reliable SPICE simulations of
*       memristors, memcapacitors and meminductors," Radioengineering, vol. 22,
*       no. 4, pp. 945–968, 2013.
"""
class BiolekR2Memristor(nn.Module):
    def __init__(self, InitVals=0.5, Verbose=False):
        # **********************************************************************
        super(BiolekR2Memristor, self).__init__()

        # **********************************************************************
        # set the function name
        # **********************************************************************
        FunctionName = "BiolekR2::__init__()"

        # **********************************************************************
        # memristor model name
        # **********************************************************************
        self.ModelName = "BiolekR2"

        # **********************************************************************
        # save parameters
        # **********************************************************************
        self.InitVals   = InitVals
        self.Verbose    = Verbose

        # **********************************************************************
        # constants for the model
        # **********************************************************************
        self.Beta   = torch.tensor(1e13, dtype=torch.float64)
        self.Ron    = torch.tensor(1e3, dtype=torch.float64)
        self.Roff   = torch.tensor(10e3, dtype=torch.float64)
        self.Vth    = torch.tensor(4.6, dtype=torch.float64)
        self.b1     = torch.tensor(10e-6, dtype=torch.float64)
        self.b2     = torch.tensor(10e-6, dtype=torch.float64)
        self.DeltaX = self.Roff - self.Ron

        self.Rinit = (self.Ron + self.Roff)/2

        def theta(x, b):
            return 1 / (1 + np.exp(-x / b))

        def gamma(x, b):
            return x * (self.theta(x, b) - self.theta(-x, b))


    # **************************************************************************
    def UpdateVals(self, Vin, dt):
        f = self.Beta * (Vin - 0.5 * (self.gamma(Vin + self.Vth, self.b1) - self.gamma(Vin - self.Vth, self.b1)))

        term1 = self.theta(Vin, self.b1) * self.theta(1 - x / self.Roff, self.b2)
        term2 = self.theta(-Vin, self.b1) * self.theta(x / self.Ron - 1, self.b2)
        W = term1 + term2   # here we separated the W function for readability

        ...

    # **************************************************************************
    def GetVals(self):
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
    ModelName   = "BiolekR2"

    # **************************************************************************
    # append to the system path
    # **************************************************************************
    import WaveGenerator

    # **************************************************************************
    # set the parameters
    # **************************************************************************
    SignalType  = "Sine"
    Ampl        = 5.0
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
    R   = torch.zeros(NumTs, dtype=torch.float64)

    # **************************************************************************
    # parameters for the memristive model
    # **************************************************************************
    MemRes = BiolekR2Memristor(Verbose=True)

    # **************************************************************************
    # display the information
    # **************************************************************************
    Msg = "...%-25s: calculate internal state ..." % (FunctionName)
    print(Msg)

    # **************************************************************************
    # calculate internal state
    # **************************************************************************
    R[0]    = MemRes.GetVals()
    for i in progressbar(range(1, NumTs)):
        # **********************************************************************
        # get the delta X or x
        # **********************************************************************
        MemRes.UpdateVals(Vs[i], dt)

        # **********************************************************************
        # save the resistance
        # **********************************************************************
        R[i]    = MemRes.GetVals()

    # **************************************************************************
    # calculate the current
    # **************************************************************************
    I   = torch.div(Vs, R)

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
    Fig = plt.figure("I-V Spice")
    plt.grid(linestyle="dotted")
    Scale = 1e3
    plt.plot(Vs, I * Scale, "-", color="b", label="I-V", linewidth=LineWidth)
    plt.xlabel("Vin (V)")
    plt.ylabel("Current (mA)")
    plt.axis("tight")
    # legend = plt.legend(loc="best")

    # **************************************************************************
    # set the file name
    # **************************************************************************
    FileNameEps = "MemRes%sIV.eps" % ModelName
    FileNameJpg = "MemRes%sIV.jpg" % ModelName
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
    Fig = plt.figure("R-V Spice")
    plt.grid(linestyle="dotted")
    Scale = 1e-3
    plt.plot(Vs, R * Scale, "-", color="b", label="R-V", linewidth=LineWidth)
    plt.xlabel("Vin (V)")
    plt.ylabel(r"Resistance ($k\Omega$)")
    plt.axis("tight")
    # legend = plt.legend(loc="best")

    # **************************************************************************
    # set the file name
    # **************************************************************************
    FileNameEps = "MemRes%sRV.eps" % ModelName
    FileNameJpg = "MemRes%sRV.jpg" % ModelName
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
